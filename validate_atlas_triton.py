import time
import math
import statistics
import torch
import triton

from atlas_triton import (
    ProductionAtlasEngine,
    _centroid_partial_kernel,
    _centroid_finalize_kernel,
)


FORWARD_MEAN_TOL = 1e-3
FORWARD_MAX_TOL = 1e-2
CENTROID_MEAN_TOL = 1e-5
CENTROID_MAX_TOL = 1e-4


def make_metadata(device, B=1, T=128, C=3, LMAX=64, overlap=True):
    cluster_positions = torch.zeros((B, C, LMAX), device=device, dtype=torch.int32)
    cluster_lengths = torch.full((B, C), LMAX, device=device, dtype=torch.int32)

    if overlap:
        spans = [
            (0, 64),
            (48, 112),
            (64, 128),
        ]
    else:
        spans = [
            (0, 64),
            (64, 128),
            (0, 64),
        ]

    for b in range(B):
        for c, (s, e) in enumerate(spans[:C]):
            vals = torch.arange(s, e, device=device, dtype=torch.int32)
            cluster_positions[b, c, : vals.numel()] = vals
            cluster_lengths[b, c] = vals.numel()

    visible_centroids = torch.zeros((B, C, max(C - 1, 1)), device=device, dtype=torch.int32)
    visible_counts = torch.full((B, C), C - 1, device=device, dtype=torch.int32)

    for b in range(B):
        for c in range(C):
            ids = [x for x in range(C) if x != c]
            visible_centroids[b, c, : len(ids)] = torch.tensor(ids, device=device, dtype=torch.int32)

    membership_counts = torch.zeros((B, T), device=device, dtype=torch.int32)
    for b in range(B):
        for c in range(C):
            n = int(cluster_lengths[b, c].item())
            toks = cluster_positions[b, c, :n].long()
            membership_counts[b, toks] += 1

    return cluster_positions, cluster_lengths, membership_counts, visible_centroids, visible_counts


def reference_centroids(k, v, cluster_positions, cluster_lengths):
    B, H, T, D = k.shape
    C = cluster_positions.shape[1]

    ck = torch.empty((B, C, D), device=k.device, dtype=torch.float32)
    cv = torch.empty((B, C, D), device=v.device, dtype=torch.float32)

    for b in range(B):
        for c in range(C):
            n = int(cluster_lengths[b, c].item())
            toks = cluster_positions[b, c, :n].long()
            ck[b, c] = k[b, :, toks, :].float().mean(dim=(0, 1))
            cv[b, c] = v[b, :, toks, :].float().mean(dim=(0, 1))

    return ck, cv


def triton_centroids(k, v, cluster_positions, cluster_lengths, block_d=64, centroid_r_block=64):
    B, H, T, D = k.shape
    C = cluster_positions.shape[1]
    LMAX = cluster_positions.shape[2]

    R = triton.cdiv(LMAX * H, centroid_r_block)
    num_d_blocks = triton.cdiv(D, block_d)

    partial_k = torch.empty((B, C, R, D), device=k.device, dtype=torch.float32)
    partial_v = torch.empty((B, C, R, D), device=v.device, dtype=torch.float32)
    centroid_k = torch.empty((B, C, D), device=k.device, dtype=torch.float32)
    centroid_v = torch.empty((B, C, D), device=v.device, dtype=torch.float32)

    _centroid_partial_kernel[(B, C, R * num_d_blocks)](
        k,
        v,
        cluster_positions,
        cluster_lengths,
        partial_k,
        partial_v,
        H,
        T,
        D,
        C,
        LMAX,
        R_BLOCK=centroid_r_block,
        BLOCK_D=block_d,
        num_warps=4,
    )

    _centroid_finalize_kernel[(B, C, num_d_blocks)](
        partial_k,
        partial_v,
        cluster_lengths,
        centroid_k,
        centroid_v,
        H,
        D,
        C,
        R,
        BLOCK_R=64,
        BLOCK_D=block_d,
        num_warps=4,
    )

    return centroid_k, centroid_v


def reference_atlas(q, k, v, cluster_positions, cluster_lengths, membership_counts, visible_centroids, visible_counts):
    B, H, T, D = q.shape
    C = cluster_positions.shape[1]
    scale = 1.0 / math.sqrt(D)

    centroid_k, centroid_v = reference_centroids(k, v, cluster_positions, cluster_lengths)

    y_sum = torch.zeros((B, H, T, D), device=q.device, dtype=torch.float32)
    y_count = torch.zeros((B, T), device=q.device, dtype=torch.float32)

    for b in range(B):
        for h in range(H):
            for c in range(C):
                n = int(cluster_lengths[b, c].item())
                toks = cluster_positions[b, c, :n].long()

                vc_n = int(visible_counts[b, c].item())
                vc_ids = visible_centroids[b, c, :vc_n].long()

                for slot in range(n):
                    tok = int(toks[slot].item())

                    local_mask = toks <= tok
                    local_toks = toks[local_mask]

                    k_parts = [k[b, h, local_toks, :].float()]
                    v_parts = [v[b, h, local_toks, :].float()]

                    if vc_n > 0:
                        k_parts.append(centroid_k[b, vc_ids].float())
                        v_parts.append(centroid_v[b, vc_ids].float())

                    kk = torch.cat(k_parts, dim=0)
                    vv = torch.cat(v_parts, dim=0)

                    scores = (q[b, h, tok, :].float() @ kk.t()) * scale
                    attn = torch.softmax(scores, dim=-1)
                    out = attn @ vv

                    y_sum[b, h, tok] += out
                    if h == 0:
                        y_count[b, tok] += 1.0

    y = y_sum / y_count.clamp_min(1.0)[:, None, :, None]
    return y.to(dtype=q.dtype)


def error_stats(a, b):
    diff = (a.float() - b.float()).abs()
    rmse = torch.sqrt(((a.float() - b.float()) ** 2).mean())
    return diff.max().item(), diff.mean().item(), rmse.item()


def make_inputs(seed, B=1, H=2, T=128, D=64, dtype=torch.bfloat16):
    torch.manual_seed(seed)
    device = "cuda"
    q = torch.randn((B, H, T, D), device=device, dtype=dtype)
    k = torch.randn((B, H, T, D), device=device, dtype=dtype)
    v = torch.randn((B, H, T, D), device=device, dtype=dtype)
    return q, k, v


def make_engine():
    return ProductionAtlasEngine(
        block_q=32,
        block_k=32,
        block_c=16,
        block_d=64,
        centroid_r_block=64,
        merge_avg=True,
    ).cuda()


def test_forward_equivalence():
    q, k, v = make_inputs(1337)
    meta = make_metadata(q.device)

    y_ref = reference_atlas(q, k, v, *meta)
    y_tri = make_engine()(q, k, v, *meta)
    torch.cuda.synchronize()

    max_err, mean_err, rmse = error_stats(y_ref, y_tri)
    passed = mean_err < FORWARD_MEAN_TOL and max_err < FORWARD_MAX_TOL

    print("FORWARD CORRECTNESS")
    print("-------------------")
    print("max error:", max_err)
    print("mean error:", mean_err)
    print("rmse:", rmse)
    print("PASS" if passed else "FAIL")
    return passed


def test_overlap_semantics():
    q, k, v = make_inputs(1337)
    meta = make_metadata(q.device, overlap=True)

    y_ref = reference_atlas(q, k, v, *meta)
    y_tri = make_engine()(q, k, v, *meta)
    torch.cuda.synchronize()

    membership_counts = meta[2]
    overlap_tokens = torch.nonzero(membership_counts[0] > 1, as_tuple=False).flatten()

    print("OVERLAP TEST")
    print("------------")

    passed = True
    for tok in overlap_tokens[:8]:
        t = int(tok.item())
        diff = (y_ref[0, 0, t].float() - y_tri[0, 0, t].float()).abs()
        max_tok = diff.max().item()
        print("token:", t)
        print("reference value:", y_ref[0, 0, t, :8].float().tolist())
        print("triton value:", y_tri[0, 0, t, :8].float().tolist())
        print("absolute max diff:", max_tok)
        if max_tok >= FORWARD_MAX_TOL:
            passed = False

    print("PASS" if passed else "FAIL")
    return passed


def test_centroid_attention():
    q, k, v = make_inputs(1337)
    cluster_positions, cluster_lengths, *_ = make_metadata(q.device)

    ck_ref, cv_ref = reference_centroids(k, v, cluster_positions, cluster_lengths)
    ck_tri, cv_tri = triton_centroids(k, v, cluster_positions, cluster_lengths)
    torch.cuda.synchronize()

    k_max, k_mean, _ = error_stats(ck_ref, ck_tri)
    v_max, v_mean, _ = error_stats(cv_ref, cv_tri)

    passed = (
        k_mean < CENTROID_MEAN_TOL
        and v_mean < CENTROID_MEAN_TOL
        and k_max < CENTROID_MAX_TOL
        and v_max < CENTROID_MAX_TOL
    )

    print("CENTROID TEST")
    print("-------------")
    print("centroid_k max:", k_max)
    print("centroid_k mean:", k_mean)
    print("centroid_v max:", v_max)
    print("centroid_v mean:", v_mean)
    print("PASS" if passed else "FAIL")
    return passed


def test_causal_mask_correctness():
    B, H, T, D = 1, 2, 16, 64
    q, k, v = make_inputs(42, B=B, H=H, T=T, D=D)

    C = 2
    LMAX = 8
    cluster_positions = torch.zeros((B, C, LMAX), device=q.device, dtype=torch.int32)
    cluster_lengths = torch.full((B, C), LMAX, device=q.device, dtype=torch.int32)
    cluster_positions[0, 0] = torch.arange(0, 8, device=q.device, dtype=torch.int32)
    cluster_positions[0, 1] = torch.arange(8, 16, device=q.device, dtype=torch.int32)

    membership_counts = torch.ones((B, T), device=q.device, dtype=torch.int32)

    # Disable centroid attention here so this isolates local causal masking.
    visible_centroids = torch.zeros((B, C, 1), device=q.device, dtype=torch.int32)
    visible_counts = torch.zeros((B, C), device=q.device, dtype=torch.int32)

    meta = (
        cluster_positions,
        cluster_lengths,
        membership_counts,
        visible_centroids,
        visible_counts,
    )

    engine = ProductionAtlasEngine(
        block_q=4,
        block_k=16,
        block_c=16,
        block_d=64,
        centroid_r_block=16,
        merge_avg=True,
    ).cuda()

    y_ref = reference_atlas(q, k, v, *meta)
    y_tri = engine(q, k, v, *meta)

    k2 = k.clone()
    v2 = v.clone()
    k2[:, :, 12:, :] += 100.0
    v2[:, :, 12:, :] += 100.0

    y_ref_2 = reference_atlas(q, k2, v2, *meta)
    y_tri_2 = engine(q, k2, v2, *meta)
    torch.cuda.synchronize()

    past = slice(0, 8)
    ref_future_effect = (y_ref[:, :, past] - y_ref_2[:, :, past]).abs().max().item()
    tri_future_effect = (y_tri[:, :, past] - y_tri_2[:, :, past]).abs().max().item()
    ref_tri_max, ref_tri_mean, _ = error_stats(y_ref, y_tri)

    passed = ref_future_effect == 0.0 and tri_future_effect == 0.0 and ref_tri_max < FORWARD_MAX_TOL

    print("CAUSAL TEST")
    print("-----------")
    print("reference future effect on past:", ref_future_effect)
    print("triton future effect on past:", tri_future_effect)
    print("reference/triton max error:", ref_tri_max)
    print("reference/triton mean error:", ref_tri_mean)
    print("PASS" if passed else "FAIL")
    return passed


def test_multiple_random_seeds():
    print("MULTIPLE RANDOM SEEDS")
    print("---------------------")
    print("seed | max_error | mean_error | rmse | status")

    all_passed = True
    for seed in [0, 1, 42, 1337]:
        q, k, v = make_inputs(seed)
        meta = make_metadata(q.device)

        y_ref = reference_atlas(q, k, v, *meta)
        y_tri = make_engine()(q, k, v, *meta)
        torch.cuda.synchronize()

        max_err, mean_err, rmse = error_stats(y_ref, y_tri)
        passed = mean_err < FORWARD_MEAN_TOL and max_err < FORWARD_MAX_TOL
        all_passed = all_passed and passed

        print(seed, "|", max_err, "|", mean_err, "|", rmse, "|", "PASS" if passed else "FAIL")

    return all_passed


def timed_runs(fn, warmup=100, iters=500):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg = sum(times) / len(times)
    p50 = statistics.median(times)
    p95 = sorted(times)[int(0.95 * len(times)) - 1]
    return avg, p50, p95


def test_microbenchmark():
    q, k, v = make_inputs(1337)
    meta = make_metadata(q.device)
    engine = make_engine()

    def ref_fn():
        reference_atlas(q, k, v, *meta)

    def tri_fn():
        engine(q, k, v, *meta)

    ref_avg, ref_p50, ref_p95 = timed_runs(ref_fn)
    tri_avg, tri_p50, tri_p95 = timed_runs(tri_fn)

    print("MICROBENCHMARK")
    print("--------------")
    print("Reference avg/p50/p95:", ref_avg, ref_p50, ref_p95)
    print("Triton avg/p50/p95:", tri_avg, tri_p50, tri_p95)
    print("Speedup:", ref_avg / tri_avg)
    return ref_avg, tri_avg


def test_memory_usage():
    q, k, v = make_inputs(1337)
    meta = make_metadata(q.device)
    engine = make_engine()

    torch.cuda.reset_peak_memory_stats()
    reference_atlas(q, k, v, *meta)
    torch.cuda.synchronize()
    ref_alloc = torch.cuda.max_memory_allocated()
    ref_reserved = torch.cuda.max_memory_reserved()

    torch.cuda.reset_peak_memory_stats()
    engine(q, k, v, *meta)
    torch.cuda.synchronize()
    tri_alloc = torch.cuda.max_memory_allocated()
    tri_reserved = torch.cuda.max_memory_reserved()

    print("MEMORY")
    print("------")
    print("Reference allocated/reserved:", ref_alloc, ref_reserved)
    print("Triton allocated/reserved:", tri_alloc, tri_reserved)
    return (ref_alloc, ref_reserved), (tri_alloc, tri_reserved)


def run_all():
    assert torch.cuda.is_available(), "CUDA is required"

    results = {}
    results["forward"] = test_forward_equivalence()
    results["overlap"] = test_overlap_semantics()
    results["centroid"] = test_centroid_attention()
    results["causal"] = test_causal_mask_correctness()
    results["seeds"] = test_multiple_random_seeds()

    ref_time, tri_time = test_microbenchmark()
    test_memory_usage()

    equivalent = all(results.values())

    print("FINAL VERDICT")
    print("-------------")
    print("Equivalent:", "YES" if equivalent else "NO")
    print("Reference time:", ref_time)
    print("Triton time:", tri_time)
    print("Speedup:", ref_time / tri_time)


if __name__ == "__main__":
    run_all()

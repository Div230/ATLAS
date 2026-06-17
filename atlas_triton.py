import torch
import triton
import triton.language as tl


@triton.jit
def _centroid_partial_kernel(
    K,
    V,
    cluster_positions,
    cluster_lengths,
    partial_k,
    partial_v,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    LMAX: tl.constexpr,
    R_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    pid = tl.program_id(2)

    num_d_blocks = tl.cdiv(D, BLOCK_D)
    r_blk = pid // num_d_blocks
    d_blk = pid - r_blk * num_d_blocks

    offs_r = r_blk * R_BLOCK + tl.arange(0, R_BLOCK)
    offs_d = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)

    length = tl.load(cluster_lengths + b * C + c)
    total = length * H

    valid_r = offs_r < total
    slot = offs_r // H
    h = offs_r - slot * H

    tok = tl.load(
        cluster_positions + (b * C + c) * LMAX + slot,
        mask=valid_r,
        other=0,
    )

    k = tl.load(
        K + ((b * H + h[:, None]) * T + tok[:, None]) * D + offs_d[None, :],
        mask=valid_r[:, None] & (offs_d[None, :] < D),
        other=0.0,
    ).to(tl.float32)

    v = tl.load(
        V + ((b * H + h[:, None]) * T + tok[:, None]) * D + offs_d[None, :],
        mask=valid_r[:, None] & (offs_d[None, :] < D),
        other=0.0,
    ).to(tl.float32)

    sum_k = tl.sum(k, axis=0)
    sum_v = tl.sum(v, axis=0)

    R = tl.cdiv(LMAX * H, R_BLOCK)
    out = (((b * C + c) * R + r_blk) * D + offs_d)

    tl.store(partial_k + out, sum_k, mask=offs_d < D)
    tl.store(partial_v + out, sum_v, mask=offs_d < D)


@triton.jit
def _centroid_finalize_kernel(
    partial_k,
    partial_v,
    cluster_lengths,
    centroid_k,
    centroid_v,
    H: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    d_blk = tl.program_id(2)

    offs_r = tl.arange(0, BLOCK_R)
    offs_d = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)

    acc_k = tl.zeros((BLOCK_R, BLOCK_D), tl.float32)
    acc_v = tl.zeros((BLOCK_R, BLOCK_D), tl.float32)

    for r0 in range(0, R, BLOCK_R):
        r = r0 + offs_r
        valid_r = r < R
        ptr = (((b * C + c) * R + r[:, None]) * D + offs_d[None, :])

        acc_k += tl.load(
            partial_k + ptr,
            mask=valid_r[:, None] & (offs_d[None, :] < D),
            other=0.0,
        )
        acc_v += tl.load(
            partial_v + ptr,
            mask=valid_r[:, None] & (offs_d[None, :] < D),
            other=0.0,
        )

    length = tl.load(cluster_lengths + b * C + c)
    denom = tl.maximum(length * H, 1)

    out = (b * C + c) * D + offs_d

    tl.store(
        centroid_k + out,
        tl.sum(acc_k, axis=0) / denom,
        mask=offs_d < D,
    )
    tl.store(
        centroid_v + out,
        tl.sum(acc_v, axis=0) / denom,
        mask=offs_d < D,
    )


@triton.jit
def _atlas_attn_atomic_kernel(
    Q,
    K,
    V,
    centroid_k,
    centroid_v,
    cluster_positions,
    cluster_lengths,
    visible_centroids,
    visible_counts,
    Y_SUM,
    Y_COUNT,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    LMAX: tl.constexpr,
    VMAX: tl.constexpr,
    QBLOCKS: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MERGE_AVG: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    pid = tl.program_id(2)

    c = pid // QBLOCKS
    qb = pid - c * QBLOCKS

    offs_q = tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)

    q_len = tl.load(cluster_lengths + b * C + c)

    q_slot = qb * BLOCK_Q + offs_q
    q_valid = q_slot < q_len

    q_tok = tl.load(
        cluster_positions + (b * C + c) * LMAX + q_slot,
        mask=q_valid,
        other=0,
    )

    q = tl.load(
        Q + ((b * H + h) * T + q_tok[:, None]) * D + offs_d[None, :],
        mask=q_valid[:, None] & (offs_d[None, :] < D),
        other=0.0,
    ).to(tl.float32)

    scale = 1.0 / tl.sqrt(tl.full((), D, tl.float32))

    m_i = tl.full((BLOCK_Q,), -3.4028234663852886e38, tl.float32)
    l_i = tl.zeros((BLOCK_Q,), tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_D), tl.float32)

    for k0 in range(0, LMAX, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        k_valid = ks < q_len

        k_tok = tl.load(
            cluster_positions + (b * C + c) * LMAX + ks,
            mask=k_valid,
            other=0,
        )

        kt = tl.load(
            K + ((b * H + h) * T + k_tok[:, None]) * D + offs_d[None, :],
            mask=k_valid[:, None] & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        vt = tl.load(
            V + ((b * H + h) * T + k_tok[:, None]) * D + offs_d[None, :],
            mask=k_valid[:, None] & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        scores = tl.dot(q, tl.trans(kt)) * scale
        causal = q_tok[:, None] >= k_tok[None, :]

        scores = tl.where(
            q_valid[:, None] & k_valid[None, :] & causal,
            scores,
            -3.4028234663852886e38,
        )

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        alpha = tl.exp(m_i - m_new)

        acc = acc * alpha[:, None] + tl.dot(p, vt)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    vc_len = tl.load(visible_counts + b * C + c)

    for c0 in range(0, VMAX, BLOCK_C):
        cs = c0 + tl.arange(0, BLOCK_C)
        c_valid = cs < vc_len

        cid = tl.load(
            visible_centroids + (b * C + c) * VMAX + cs,
            mask=c_valid,
            other=0,
        )

        ck = tl.load(
            centroid_k + (b * C + cid[:, None]) * D + offs_d[None, :],
            mask=c_valid[:, None] & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        cv = tl.load(
            centroid_v + (b * C + cid[:, None]) * D + offs_d[None, :],
            mask=c_valid[:, None] & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        scores = tl.dot(q, tl.trans(ck)) * scale

        scores = tl.where(
            q_valid[:, None] & c_valid[None, :],
            scores,
            -3.4028234663852886e38,
        )

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        alpha = tl.exp(m_i - m_new)

        acc = acc * alpha[:, None] + tl.dot(p, cv)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    out = acc / tl.maximum(l_i[:, None], 1.0)

    out_ptr = ((b * H + h) * T + q_tok[:, None]) * D + offs_d[None, :]

    tl.atomic_add(
        Y_SUM + out_ptr,
        out,
        sem="relaxed",
        mask=q_valid[:, None] & (offs_d[None, :] < D),
    )

    if MERGE_AVG:
        tl.atomic_add(
            Y_COUNT + b * T + q_tok,
            tl.full((BLOCK_Q,), 1.0, tl.float32),
            sem="relaxed",
            mask=(h == 0) & q_valid,
        )


@triton.jit
def _atlas_normalize_kernel(
    Y_SUM,
    Y_COUNT,
    Y,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.program_id(2)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    count = tl.load(Y_COUNT + b * T + t)

    val = tl.load(
        Y_SUM + ((b * H + h) * T + t) * D + offs_d,
        mask=mask_d,
        other=0.0,
    )

    val = val / tl.maximum(count, 1.0)

    tl.store(
        Y + ((b * H + h) * T + t) * D + offs_d,
        val,
        mask=mask_d,
    )


class ProductionAtlasEngine(torch.nn.Module):
    def __init__(
        self,
        block_q=32,
        block_k=64,
        block_c=32,
        block_d=64,
        centroid_r_block=128,
        merge_avg=True,
    ):
        super().__init__()
        self.block_q = block_q
        self.block_k = block_k
        self.block_c = block_c
        self.block_d = block_d
        self.centroid_r_block = centroid_r_block
        self.merge_avg = merge_avg

    def forward(
        self,
        q,
        k,
        v,
        cluster_positions,
        cluster_lengths,
        membership_counts,
        visible_centroids,
        visible_counts,
    ):
        B, H, T, D = q.shape
        C = cluster_positions.shape[1]
        LMAX = cluster_positions.shape[2]
        VMAX = visible_centroids.shape[2]

        assert D <= self.block_d, "V1 requires block_d >= D"
        assert self.block_c >= 16, "Triton tl.dot requires block_c >= 16"
        assert self.block_k >= 16, "Triton tl.dot requires block_k >= 16"

        qblocks = triton.cdiv(LMAX, self.block_q)
        R = triton.cdiv(LMAX * H, self.centroid_r_block)
        num_d_blocks = triton.cdiv(D, self.block_d)

        cluster_positions = cluster_positions.to(device=q.device, dtype=torch.int32)
        cluster_lengths = cluster_lengths.to(device=q.device, dtype=torch.int32)
        visible_centroids = visible_centroids.to(device=q.device, dtype=torch.int32)
        visible_counts = visible_counts.to(device=q.device, dtype=torch.int32)

        partial_k = torch.empty((B, C, R, D), device=q.device, dtype=torch.float32)
        partial_v = torch.empty((B, C, R, D), device=q.device, dtype=torch.float32)
        centroid_k = torch.empty((B, C, D), device=q.device, dtype=torch.float32)
        centroid_v = torch.empty((B, C, D), device=q.device, dtype=torch.float32)

        y_sum = torch.zeros((B, H, T, D), device=q.device, dtype=torch.float32)
        y_count = torch.zeros((B, T), device=q.device, dtype=torch.float32)
        y = torch.empty_like(q)

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
            R_BLOCK=self.centroid_r_block,
            BLOCK_D=self.block_d,
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
            BLOCK_D=self.block_d,
            num_warps=4,
        )

        _atlas_attn_atomic_kernel[(B, H, C * qblocks)](
            q,
            k,
            v,
            centroid_k,
            centroid_v,
            cluster_positions,
            cluster_lengths,
            visible_centroids,
            visible_counts,
            y_sum,
            y_count,
            H,
            T,
            D,
            C,
            LMAX,
            VMAX,
            qblocks,
            BLOCK_Q=self.block_q,
            BLOCK_K=self.block_k,
            BLOCK_C=self.block_c,
            BLOCK_D=self.block_d,
            MERGE_AVG=self.merge_avg,
            num_warps=4,
            num_stages=3,
        )

        if self.merge_avg:
            _atlas_normalize_kernel[(B, H, T)](
                y_sum,
                y_count,
                y,
                H,
                T,
                D,
                BLOCK_D=self.block_d,
                num_warps=4,
            )
            return y

        return y_sum.to(dtype=q.dtype)


def _make_test_metadata(device):
    B = 1
    T = 128
    C = 3
    LMAX = 64

    cluster_positions = torch.zeros((B, C, LMAX), device=device, dtype=torch.int32)
    cluster_lengths = torch.full((B, C), LMAX, device=device, dtype=torch.int32)

    cluster_positions[0, 0] = torch.arange(0, 64, device=device, dtype=torch.int32)
    cluster_positions[0, 1] = torch.arange(48, 112, device=device, dtype=torch.int32)
    cluster_positions[0, 2] = torch.arange(64, 128, device=device, dtype=torch.int32)

    visible_centroids = torch.zeros((B, C, C - 1), device=device, dtype=torch.int32)
    visible_counts = torch.full((B, C), C - 1, device=device, dtype=torch.int32)

    visible_centroids[0, 0] = torch.tensor([1, 2], device=device, dtype=torch.int32)
    visible_centroids[0, 1] = torch.tensor([0, 2], device=device, dtype=torch.int32)
    visible_centroids[0, 2] = torch.tensor([0, 1], device=device, dtype=torch.int32)

    membership_counts = torch.ones((B, T), device=device, dtype=torch.int32)
    membership_counts[0, 48:64] = 2
    membership_counts[0, 64:112] = 2

    return (
        cluster_positions,
        cluster_lengths,
        membership_counts,
        visible_centroids,
        visible_counts,
    )


def test_atlas_triton_v1():
    assert torch.cuda.is_available(), "CUDA is required"

    device = "cuda"
    B = 1
    H = 2
    T = 128
    D = 64
    dtype = torch.bfloat16

    q = torch.randn((B, H, T, D), device=device, dtype=dtype)
    k = torch.randn((B, H, T, D), device=device, dtype=dtype)
    v = torch.randn((B, H, T, D), device=device, dtype=dtype)

    (
        cluster_positions,
        cluster_lengths,
        membership_counts,
        visible_centroids,
        visible_counts,
    ) = _make_test_metadata(device)

    engine = ProductionAtlasEngine(
        block_q=32,
        block_k=32,
        block_c=2,
        block_d=64,
        centroid_r_block=64,
        merge_avg=True,
    ).to(device)

    y = engine(
        q,
        k,
        v,
        cluster_positions,
        cluster_lengths,
        membership_counts,
        visible_centroids,
        visible_counts,
    )

    torch.cuda.synchronize()

    print("y.shape:", y.shape)
    print("y.dtype:", y.dtype)
    print("has_nan:", torch.isnan(y).any().item())

    assert y.shape == q.shape
    assert y.dtype == q.dtype
    assert not torch.isnan(y).any()


if __name__ == "__main__":
    test_atlas_triton_v1()

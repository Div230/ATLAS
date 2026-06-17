from __future__ import annotations
from typing import Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def build_clusters_gpu(
    k_norm:       torch.Tensor,   # (B, T, Dh)  float32, L2-normalised
    cluster_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-only clustering.  Zero CPU bookkeeping inside the per-cluster loop.

    Parameters
    ----------
    k_norm : (B, T, Dh) float32, already L2-normalised
    cluster_size : int

    Returns
    -------
    cluster_idx   : (B, N, S) torch.long
    cluster_valid : (B, N, S) torch.bool
    """
    B, T, Dh = k_norm.shape
    S    = cluster_size
    dev  = k_norm.device

    max_clusters = (T + S - 1) // S

    # Pre-allocate outputs.
    cluster_idx_all   = torch.zeros(B, max_clusters, S, device=dev, dtype=torch.long)
    cluster_valid_all = torch.zeros(B, max_clusters, S, device=dev, dtype=torch.bool)
     

    # assigned_bias[b, t] = 0.0 (free) or -inf (assigned).
    # Adding this to sims suppresses assigned tokens without a separate mask op.
    assigned_bias = torch.zeros(B, T, device=dev, dtype=torch.float32)

    # Per-batch remaining token count (GPU tensor, updated by subtraction).
    remaining = torch.full((B,), T, device=dev, dtype=torch.int32)

    # Precompute negative position scores for seed finding (reused every step).
    # neg_pos[t] = -t  →  argmax over free tokens gives the smallest index.
    # float32 is exact for t in [0, 2^24), so T ≤ 4096 is safe.
    neg_pos = -torch.arange(T, device=dev, dtype=torch.float32)  # (T,)

    # Cached batch-index vector for advanced indexing.
    b_idx = torch.arange(B, device=dev, dtype=torch.long)        # (B,)

    actual_steps = 0

    for step in range(max_clusters):
        # ── Early exit ────────────────────────────────────────────────
        # Single DtoH transfer: one int32 scalar per step.
        if remaining.max().item() <= 0:
            break
        actual_steps += 1

        # ── Seed finding (GPU, no alloc beyond one (B,T) temp) ────────
        # seed_score[b, t] = neg_pos[t] + assigned_bias[b, t]
        #   = -t          if free  (argmax → smallest free index)
        #   = -t + (-inf) if assigned (= -inf, never chosen)
        # This is mathematically identical to the original cursor scan.
        seed_score = neg_pos.unsqueeze(0) + assigned_bias        # (B, T)

        # Suppress done batch elements so their seed doesn't corrupt output.
        done_mask = (remaining <= 0).unsqueeze(1)                # (B, 1)
        if done_mask.any():
            seed_score = seed_score.masked_fill(done_mask, float('-inf'))

        seeds = seed_score.argmax(dim=1)                         # (B,) long

        # ── Batched cosine similarity (GEMM via bmm) ──────────────────
        seed_vecs = k_norm[b_idx, seeds]                         # (B, Dh)
        # bmm: (B, T, Dh) × (B, Dh, 1) → (B, T, 1) → (B, T)
        sims = torch.bmm(k_norm, seed_vecs.unsqueeze(2)).squeeze(2)  # (B, T)

        # ── Suppress assigned tokens and done batches ─────────────────
        sims = sims + assigned_bias
        if done_mask.any():
            sims = sims.masked_fill(done_mask, float('-inf'))

        # ── Top-K selection ───────────────────────────────────────────
        k_take = min(S, T)
        top_vals, top_inds = torch.topk(sims, k=k_take, largest=True, sorted=True)

        # Sort indices to match original's canonical (ascending) ordering.
        top_inds_sorted, sort_perm = top_inds.sort(dim=1)        # (B, S)
        top_sims_sorted = top_vals.gather(1, sort_perm)          # (B, S)

        # ── Valid mask ────────────────────────────────────────────────
        # A slot is valid if its similarity was not -inf
        # (i.e. the token was actually free, not a padding artefact from
        # a done batch element where all sims are -inf).
        valid_mask = top_sims_sorted > (float('-inf') + 1.0)     # (B, S) bool

        # ── Update assigned_bias — Bug #1 fix ─────────────────────────
        # WRONG:  assigned_bias.scatter_(1, idx, neg_inf * valid.float())
        #         because -inf * 0.0 = NaN (IEEE 754).
        # CORRECT: use torch.where to produce exactly -inf or 0.0.
        src = torch.where(
            valid_mask,
            torch.full((B, k_take), float('-inf'), device=dev),
            torch.zeros((B, k_take), device=dev),
        )
        assigned_bias.scatter_(1, top_inds_sorted, src)

        # ── Store results ─────────────────────────────────────────────
        # Zero-fill the index slots that aren't valid (valid_mask=False).
        cluster_idx_all[:, step, :k_take]   = top_inds_sorted.masked_fill(~valid_mask, 0)
        cluster_valid_all[:, step, :k_take] = valid_mask

        # ── Update remaining ──────────────────────────────────────────
        per_batch_taken = valid_mask.sum(dim=1).to(torch.int32)  # (B,)
        remaining -= per_batch_taken

    N = max(actual_steps, 1)
    return cluster_idx_all[:, :N], cluster_valid_all[:, :N]


@torch.no_grad()
def build_clusters_drop_in(
    k_shared:     torch.Tensor,   # (B, T, Dh)  raw (un-normalised)
    cluster_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact drop-in for ClusterSelfAttention12._build_clusters().

    Normalises k_shared in float32 (identical to original) then delegates
    to build_clusters_gpu().
    """
    k_norm = F.normalize(k_shared.float(), dim=-1)
    return build_clusters_gpu(k_norm, cluster_size=cluster_size)

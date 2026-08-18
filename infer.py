"""
infer.py — Production inference script for ATLAS / Dense GPT checkpoints.

Supported checkpoints
---------------------
  latest.pt            – symlink written by CheckpointManager
  best.pt              – best-val-loss snapshot
  checkpoint_step_X.pt – periodic snapshots
  final_model.pt       – raw BF16 state-dict (training/main() export)
  final_model.int8.ptz – int8 + zlib quantised (training/main() export)

Supported generation modes
--------------------------
  --interactive        – REPL loop (default when no prompts given)
  --prompt "text"      – single prompt (repeatable)
  --prompt-file path   – one prompt per line
  --batch-file path    – same as prompt-file, alias kept for clarity

Usage examples
--------------
  python infer.py --checkpoint outputs/dense/checkpoints/best.pt \
                  --tokenizer data/tokenizers/fineweb_1024_bpe.model \
                  --interactive

  python infer.py --checkpoint final_model.pt \
                  --tokenizer data/tokenizers/fineweb_1024_bpe.model \
                  --prompt "Once upon a time" \
                  --max-new-tokens 200 --temperature 0.8 --top-p 0.9

  python infer.py --checkpoint final_model.int8.ptz \
                  --tokenizer data/tokenizers/fineweb_1024_bpe.model \
                  --prompt-file prompts.txt --stream
"""

from __future__ import annotations

import argparse
import io
import math
import os
import random
import sys
import zlib
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

# ──────────────────────────────────────────────────────────────────────────────
# OPTIONAL: try importing the Triton ATLAS engine (not needed for inference).
# We silently skip it so the script runs even without atlas_triton_2 installed.
# ──────────────────────────────────────────────────────────────────────────────
try:
    from atlas_triton_2 import ProductionAtlasEngine  # noqa: F401
    _HAS_ATLAS_TRITON = True
except ImportError:
    _HAS_ATLAS_TRITON = False


# =============================================================================
# QUANTISATION HELPERS  (identical to training script)
# =============================================================================

CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights",
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = CONTROL_TENSOR_NAME_PATTERNS
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16


def dequantize_state_dict_int8(obj: dict) -> dict[str, Tensor]:
    """Restore a full-precision state-dict from the int8 bundle."""
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})

    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (
                q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))
            ).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()

    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t

    return out


# =============================================================================
# MODEL ARCHITECTURE  (exact copy from training script)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    """Weight stays in fp32; cast to activation dtype at matmul time."""
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                is_causal=True,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class ClusterSelfAttention16(nn.Module):
    """
    ATLAS ClusterSelfAttention — vectorised clustering, no per-cluster DtoH sync
    on the hot path (only the last-cluster fallback does one small transfer).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.c_q = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.c_k = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.c_v = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init))
        self.rotary = Rotary(self.head_dim, base=rope_base)

        self.cluster_size = 64
        self.last_cluster_count = 0
        self.last_avg_cluster_size = 0.0
        self.profile_counter = 0
        self.enable_torch_profiler = False
        self._arange_cache: dict = {}

    # ------------------------------------------------------------------
    def _get_arange(self, n: int, device: torch.device) -> torch.Tensor:
        key = (n, str(device))
        if key not in self._arange_cache:
            self._arange_cache[key] = torch.arange(n, device=device)
        return self._arange_cache[key]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _build_clusters(
        self, k_shared: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, Dh = k_shared.shape
        device = k_shared.device
        S = self.cluster_size

        k_norm = F.normalize(k_shared.float(), dim=-1)
        max_clusters = (T + S - 1) // S

        cluster_idx = torch.zeros((B, max_clusters, S), device=device, dtype=torch.long)
        cluster_valid = torch.zeros((B, max_clusters, S), device=device, dtype=torch.bool)
        total_cluster_size = 0
        total_cluster_count = 0
        max_cluster_count = 0
        ownership = torch.empty((B, T), device=device, dtype=torch.long)

        for b in range(B):
            k_norm_b = k_norm[b]
            assigned_bias = k_norm_b.new_zeros(T)
            assigned_cpu = np.zeros(T, dtype=np.bool_)
            remaining_count = T
            seed_cursor = 0
            cluster_count_b = 0

            while remaining_count > 0:
                while seed_cursor < T and assigned_cpu[seed_cursor]:
                    seed_cursor += 1
                seed = seed_cursor

                if remaining_count <= S:
                    remaining_idx = np.flatnonzero(~assigned_cpu)
                    cluster = torch.tensor(remaining_idx, device=device, dtype=torch.long)
                    cluster_cpu = remaining_idx
                else:
                    seed_vec = k_norm_b[seed]
                    sims = k_norm_b @ seed_vec
                    sims = sims + assigned_bias
                    top_result = torch.topk(sims, k=S, largest=True)
                    cluster = top_result.indices
                    cluster_cpu = cluster.cpu().numpy()

                if remaining_count > S:
                    assigned_bias.scatter_(0, cluster, float("-inf"))

                assigned_cpu[cluster_cpu] = True
                remaining_count -= len(cluster_cpu)

                n = cluster.numel()
                cluster_idx[b, cluster_count_b, :n] = cluster
                cluster_valid[b, cluster_count_b, :n] = True
                ownership[b, cluster] = cluster_count_b
                total_cluster_size += n
                total_cluster_count += 1
                cluster_count_b += 1

            max_cluster_count = max(max_cluster_count, cluster_count_b)

        N = max_cluster_count
        cluster_idx = cluster_idx[:, :N]
        cluster_valid = cluster_valid[:, :N]
        self._last_cluster_assignments = ownership.detach().cpu()
        self.last_cluster_count = float(total_cluster_count) / max(B, 1)
        self.last_avg_cluster_size = float(total_cluster_size) / max(total_cluster_count, 1)
        self._last_cluster_idx = cluster_idx.detach().cpu()
        self._last_cluster_valid = cluster_valid.detach().cpu()
        return cluster_idx, cluster_valid

    # ------------------------------------------------------------------
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H = self.num_heads
        H_kv = self.num_kv_heads
        Dh = self.head_dim
        S = self.cluster_size
        device = x.device

        q = self.c_q(x).view(B, T, H, Dh).transpose(1, 2)
        k = self.c_k(x).view(B, T, H_kv, Dh).transpose(1, 2)
        v = self.c_v(x).view(B, T, H_kv, Dh).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        cos, sin = self.rotary(T, device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        if H_kv != H:
            repeat = H // H_kv
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        k_shared = k.mean(dim=1)

        cluster_idx, cluster_valid = self._build_clusters(k_shared.detach())
        N = cluster_idx.size(1)

        flat_idx = cluster_idx.reshape(B, N * S)
        gather_idx = flat_idx[:, None, :, None].expand(B, H, N * S, Dh)

        Q_cluster = torch.gather(q, dim=2, index=gather_idx).view(B, H, N, S, Dh)
        K_cluster = torch.gather(k, dim=2, index=gather_idx).view(B, H, N, S, Dh)
        V_cluster = torch.gather(v, dim=2, index=gather_idx).view(B, H, N, S, Dh)

        b_idx = self._get_arange(B, device)[:, None, None]
        k_shared_cluster = k_shared[b_idx, cluster_idx]

        valid_f = cluster_valid.to(dtype=k_shared.dtype)
        valid_f_BNS1 = valid_f.unsqueeze(-1)
        counts = valid_f_BNS1.sum(dim=2).clamp_min_(1.0)
        centroids_k = (k_shared_cluster * valid_f_BNS1).sum(dim=2) / counts

        cluster_min_pos = cluster_idx.masked_fill(~cluster_valid, T).min(dim=2).values
        cluster_active = cluster_valid.any(dim=2)

        if N > 1:
            all_ids = self._get_arange(N, device)
            external_ids = all_ids[None, :].expand(N, N)
            external_ids = external_ids[external_ids != all_ids[:, None]].view(N, N - 1)

            b_idx_BN1 = self._get_arange(B, device)[:, None, None]
            ext_ids_BNN1 = external_ids[None, :, :].expand(B, N, N - 1)
            external_k = centroids_k[b_idx_BN1, ext_ids_BNN1]
            external_k_Bh = external_k[:, None, :, :, :].expand(B, H, N, N - 1, Dh)

            K_star = torch.cat([K_cluster, external_k_Bh], dim=3)
            V_star = torch.cat(
                [V_cluster, V_cluster.new_zeros(B, H, N, N - 1, Dh)], dim=3
            )

            external_min_pos = cluster_min_pos[b_idx_BN1, ext_ids_BNN1]
            external_active = cluster_active[
                self._get_arange(B, device)[:, None, None], ext_ids_BNN1
            ]
            L = S + N - 1
        else:
            K_star = K_cluster
            V_star = V_cluster
            L = S
            external_min_pos = cluster_min_pos.new_empty((B, N, 0))
            external_active = cluster_active.new_empty((B, N, 0))

        query_pos = cluster_idx
        key_pos = cluster_idx

        local_mask = (
            cluster_valid[:, :, :, None]
            & cluster_valid[:, :, None, :]
            & (query_pos[:, :, :, None] >= key_pos[:, :, None, :])
        )

        external_mask = (
            cluster_valid[:, :, :, None]
            & external_active[:, :, None, :]
            & (query_pos[:, :, :, None] >= external_min_pos[:, :, None, :])
        )

        if L > S:
            attn_mask = torch.cat([local_mask, external_mask], dim=3)
        else:
            attn_mask = local_mask

        attn_mask[..., 0] = attn_mask[..., 0] | ~cluster_valid

        q_sdpa = Q_cluster.reshape(B * H * N, S, Dh)
        k_sdpa = K_star.reshape(B * H * N, L, Dh)
        v_sdpa = V_star.reshape(B * H * N, L, Dh)

        mask_sdpa = (
            attn_mask[:, None, :, :, :]
            .expand(B, H, N, S, L)
            .reshape(B * H * N, S, L)
        )

        with sdpa_kernel(SDPBackend.MATH):
            output = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=mask_sdpa,
                dropout_p=0.0,
                is_causal=False,
            ).view(B, H, N, S, Dh)

        output = output * valid_f[:, None, :, :, None]

        y = torch.zeros_like(q)
        flat_valid_f = valid_f.reshape(B, N * S)
        flat_output = output.reshape(B, H, N * S, Dh)
        flat_output = flat_output * flat_valid_f[:, None, :, None]
        scatter_idx = flat_idx[:, None, :, None].expand(B, H, N * S, Dh)
        y.scatter_add_(dim=2, index=scatter_idx, src=flat_output)

        y = y.transpose(1, 2).reshape(B, T, C)
        y = self.proj(y)

        self.profile_counter += 1
        self._last_attn_mask = attn_mask.detach().cpu()
        self._last_N = N
        self._last_L = L
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.w1 = CastedLinear(dim, hidden, bias=False)
        self.w2 = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        return self.proj(x1 * F.silu(x2))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        use_atlas: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        attn_cls = ClusterSelfAttention16 if use_atlas else CausalSelfAttention
        self.attn = attn_cls(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        use_atlas: bool = False,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32)
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim, num_heads, num_kv_heads, mlp_mult,
                    rope_base, qk_gain_init, use_atlas=use_atlas,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = (
            None if tie_embeddings
            else CastedLinear(model_dim, vocab_size, bias=False)
        )
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    @torch.no_grad()
    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)

        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x)

        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)

        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return logits

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Cross-entropy loss — kept for API compatibility, not used during inference."""
        logits = self.forward_logits(input_ids)
        logits = logits.reshape(-1, logits.size(-1))
        targets = target_ids.reshape(-1)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================

def _detect_checkpoint_type(ckpt: dict) -> str:
    """
    Inspect state_dict keys to determine whether this is a Dense or ATLAS run.

    Heuristic: any Block whose .attn module is a ClusterSelfAttention will have
    parameter names containing "c_q" *without* being wrapped under "attn.c_q"
    in a CastedLinear (i.e. it will be a plain nn.Linear, but the naming is the
    same).  The reliable signal is whether *any* block has "._arange_cache" or
    "_last_cluster" cached attributes — but those aren't in the state-dict.

    Instead we look for whether the checkpoint config (if present) mentions
    "atlas", or we fall back to key inspection of the "blocks.*.attn" subtree.
    Nothing in the Dense CausalSelfAttention is *absent* from ClusterSelfAttention16,
    so the reverse check (is ClusterSelfAttention16?) is easier:
      ClusterSelfAttention16 uses plain nn.Linear, not CastedLinear.
      CastedLinear stores its weight under the same name.
    Both have the same set of weight names, so we rely on the saved config.
    """
    # 1. Explicit config key
    config = ckpt.get("config", {})
    if isinstance(config, dict):
        experiment = config.get("experiment_name", "")
        if "atlas" in str(experiment).lower():
            return "atlas"
        if "dense" in str(experiment).lower():
            return "dense"

    # 2. checkpoint_state may carry a hint
    cs = ckpt.get("checkpoint_state", {})
    if isinstance(cs, dict):
        for v in cs.values():
            if isinstance(v, str) and "atlas" in v.lower():
                return "atlas"

    # 3. Fallback: both architectures have identical parameter names.
    #    We assume Dense unless the user explicitly requested ATLAS via args.
    return "dense"


def load_checkpoint(
    path: str,
    device: torch.device,
    use_atlas: Optional[bool] = None,
) -> tuple[dict, dict, str]:
    """
    Load a checkpoint from disk.

    Returns
    -------
    state_dict : dict[str, Tensor]
    ckpt_meta  : dict   — everything else (step, val metrics, config …)
    arch       : str    — "dense" | "atlas"
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    print(f"[load] reading {path} …", flush=True)

    # ── int8+zlib quantised bundle ─────────────────────────────────────
    if path.suffix == ".ptz":
        with open(path, "rb") as f:
            blob = f.read()
        raw = zlib.decompress(blob)
        quant_obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        state_dict = dequantize_state_dict_int8(quant_obj)
        # No checkpoint metadata available in .ptz files
        ckpt_meta = {"step": None, "checkpoint_state": {}, "config": {}}
        arch = "atlas" if use_atlas else "dense"
        return state_dict, ckpt_meta, arch

    # ── Normal .pt checkpoint ──────────────────────────────────────────
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Determine state-dict location
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and all(isinstance(v, Tensor) for v in ckpt.values()):
        # Raw state-dict (e.g. final_model.pt saved with torch.save(model.state_dict(), ...))
        state_dict = ckpt
        ckpt = {"step": None, "checkpoint_state": {}, "config": {}}
    else:
        raise ValueError(
            f"Cannot identify state_dict in checkpoint. "
            f"Top-level keys: {list(ckpt.keys())}"
        )

    # Detect architecture
    if use_atlas is True:
        arch = "atlas"
    elif use_atlas is False:
        arch = "dense"
    else:
        arch = _detect_checkpoint_type(ckpt)

    return state_dict, ckpt, arch


def build_model(
    arch: str,
    device: torch.device,
    # Model config — matches training defaults
    vocab_size: int = 1024,
    num_layers: int = 12,
    model_dim: int = 896,
    num_heads: int = 14,
    num_kv_heads: int = 7,
    mlp_mult: int = 3,
    tie_embeddings: bool = True,
    tied_embed_init_std: float = 0.005,
    logit_softcap: float = 30.0,
    rope_base: float = 50000.0,
    qk_gain_init: float = 1.5,
) -> GPT:
    use_atlas = arch == "atlas"
    model = GPT(
        vocab_size=vocab_size,
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_mult=mlp_mult,
        tie_embeddings=tie_embeddings,
        tied_embed_init_std=tied_embed_init_std,
        logit_softcap=logit_softcap,
        rope_base=rope_base,
        qk_gain_init=qk_gain_init,
        use_atlas=use_atlas,
    )
    model = model.to(device)
    return model


# =============================================================================
# GENERATION
# =============================================================================

def _apply_repetition_penalty(logits: Tensor, ids: Tensor, penalty: float) -> Tensor:
    """Standard multiplicative repetition penalty applied in-place."""
    if penalty == 1.0:
        return logits
    # ids: (1, T) — gather unique token ids
    unique_ids = ids[0].unique()
    score = logits[0, unique_ids]
    # Penalise tokens that have already appeared
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits[0, unique_ids] = score
    return logits


def _sample_next(
    logits_last: Tensor,   # (1, vocab_size)
    temperature: float,
    top_k: int,
    top_p: float,
) -> Tensor:
    """
    Sample one token from logits with temperature / top-k / top-p.
    Returns a (1,) long tensor.
    """
    if temperature == 0.0:
        # Greedy
        return logits_last.argmax(dim=-1)

    logits = logits_last / temperature

    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, -1:]] = float("-inf")

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumprob = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        # (shift right: first token always kept)
        sorted_remove = cumprob - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[sorted_remove] = float("-inf")
        logits = torch.gather(sorted_logits, dim=1, index=sorted_idx.argsort(dim=1))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def generate(
    model: GPT,
    tokenizer: spm.SentencePieceProcessor,
    device: torch.device,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    max_context: int = 1024,
    stream: bool = False,
) -> str | Iterator[str]:
    """
    Full-featured generation with temperature / top-k / top-p / rep-penalty.

    If stream=True returns a generator that yields decoded tokens one-by-one.
    Otherwise returns the full decoded string.
    """
    model.eval()
    ids = torch.tensor(
        [tokenizer.encode(prompt)], dtype=torch.long, device=device
    )

    def _generate_iter() -> Iterator[str]:
        nonlocal ids
        for _ in range(max_new_tokens):
            context = ids[:, -max_context:]
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
            ):
                logits = model.forward_logits(context)  # (1, T, V)

            logits_last = logits[:, -1, :]  # (1, V)

            if repetition_penalty != 1.0:
                logits_last = _apply_repetition_penalty(logits_last, ids, repetition_penalty)

            next_token = _sample_next(logits_last, temperature, top_k, top_p)
            next_token = next_token.unsqueeze(0) if next_token.dim() == 1 else next_token.reshape(1, 1)
            ids = torch.cat([ids, next_token], dim=1)

            # Decode just the new token
            piece = tokenizer.id_to_piece(int(next_token.item()))
            # SentencePiece uses ▁ for word-start space; decode the whole
            # running sequence for correctness on multi-byte tokens.
            token_text = tokenizer.decode([int(next_token.item())])
            yield token_text

    if stream:
        return _generate_iter()
    else:
        all_text = "".join(_generate_iter())
        return all_text


# =============================================================================
# CHECKPOINT METADATA PRINTER
# =============================================================================

def print_checkpoint_info(ckpt_meta: dict, arch: str, model: GPT) -> None:
    """Pretty-print checkpoint metadata and model info."""
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"  Architecture : {arch.upper()}")
    print(f"  Parameters   : {n_params:,}  ({n_params / 1e6:.2f}M)")

    step = ckpt_meta.get("step")
    if step is not None:
        print(f"  Step         : {step:,}")

    cs = ckpt_meta.get("checkpoint_state", {})
    if cs:
        last_loss = cs.get("last_val_loss")
        last_bpb = cs.get("last_val_bpb")
        last_ppl = cs.get("last_val_ppl")
        best_loss = cs.get("best_val_loss")
        training_ms = cs.get("training_time_ms")

        if last_loss is not None and last_loss < float("inf"):
            print(f"  Val loss     : {last_loss:.4f}")
        if last_bpb is not None and last_bpb < float("inf"):
            print(f"  Val BPB      : {last_bpb:.4f}")
        if last_ppl is not None and last_ppl < float("inf"):
            print(f"  Val PPL      : {last_ppl:.4f}")
        if best_loss is not None and best_loss < float("inf"):
            print(f"  Best val loss: {best_loss:.4f}")
        if training_ms is not None:
            hours = training_ms / (1000 * 3600)
            print(f"  Train time   : {hours:.2f} h")

        bmt = cs.get("best_model_tracking", {})
        if isinstance(bmt, dict) and bmt.get("best_val_loss", float("inf")) < float("inf"):
            print(f"  Best tracked : loss={bmt['best_val_loss']:.4f} "
                  f"@ step {bmt.get('step_of_best_loss', '?')}")

    config = ckpt_meta.get("config", {})
    if config:
        print(f"\n  Saved config :")
        skip_keys = {"run_id", "wandb_project", "wandb_entity", "wandb_mode",
                     "train_files", "val_files", "data_path"}
        for k, v in config.items():
            if k not in skip_keys:
                print(f"    {k:30s} = {v}")

    print(f"{'='*60}\n")


# =============================================================================
# INTERACTIVE / BATCH MODES
# =============================================================================

def run_interactive(
    model: GPT,
    tokenizer: spm.SentencePieceProcessor,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    print("Entering interactive mode. Type your prompt and press Enter.")
    print("Commands: :quit  :seed <n>  :temp <f>  :topk <n>  :topp <f>")
    print("          :rep <f>  :tokens <n>  :stream on|off\n")

    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    rep_penalty = args.repetition_penalty
    max_new_tokens = args.max_new_tokens
    stream = args.stream

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt:
            continue

        # In-session command parser
        if prompt.startswith(":"):
            parts = prompt.split()
            cmd = parts[0]
            val = parts[1] if len(parts) > 1 else ""
            if cmd == ":quit":
                break
            elif cmd == ":seed":
                seed = int(val)
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                print(f"  [seed set to {seed}]")
            elif cmd == ":temp":
                temperature = float(val)
                print(f"  [temperature = {temperature}]")
            elif cmd == ":topk":
                top_k = int(val)
                print(f"  [top_k = {top_k}]")
            elif cmd == ":topp":
                top_p = float(val)
                print(f"  [top_p = {top_p}]")
            elif cmd == ":rep":
                rep_penalty = float(val)
                print(f"  [repetition_penalty = {rep_penalty}]")
            elif cmd == ":tokens":
                max_new_tokens = int(val)
                print(f"  [max_new_tokens = {max_new_tokens}]")
            elif cmd == ":stream":
                stream = val.lower() in ("on", "true", "1", "yes")
                print(f"  [stream = {stream}]")
            else:
                print(f"  Unknown command: {cmd}")
            continue

        print()
        if stream:
            sys.stdout.write(prompt)
            sys.stdout.flush()
            gen = generate(
                model, tokenizer, device, prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=rep_penalty,
                stream=True,
            )
            for piece in gen:
                sys.stdout.write(piece)
                sys.stdout.flush()
            print("\n")
        else:
            full = generate(
                model, tokenizer, device, prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=rep_penalty,
                stream=False,
            )
            print(full)
            print()


def run_batch(
    prompts: List[str],
    model: GPT,
    tokenizer: spm.SentencePieceProcessor,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    for i, prompt in enumerate(prompts):
        print(f"\n{'─'*60}")
        print(f"[{i+1}/{len(prompts)}] Prompt: {prompt[:80]}{'…' if len(prompt)>80 else ''}")
        print(f"{'─'*60}")

        if args.stream:
            sys.stdout.write(prompt)
            sys.stdout.flush()
            gen = generate(
                model, tokenizer, device, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stream=True,
            )
            for piece in gen:
                sys.stdout.write(piece)
                sys.stdout.flush()
            print()
        else:
            full = generate(
                model, tokenizer, device, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stream=False,
            )
            print(full)


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inference script for ATLAS / Dense GPT checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Paths ──────────────────────────────────────────────────────────
    p.add_argument(
        "--checkpoint", "-c", required=True,
        help=(
            "Path to checkpoint. Accepts: latest.pt, best.pt, "
            "checkpoint_step_X.pt, final_model.pt, final_model.int8.ptz"
        ),
    )
    p.add_argument(
        "--tokenizer", "-t",
        default="data/tokenizers/fineweb_1024_bpe.model",
        help="Path to SentencePiece .model file.",
    )

    # ── Architecture override ──────────────────────────────────────────
    p.add_argument(
        "--arch", choices=["auto", "dense", "atlas"], default="auto",
        help=(
            "Force architecture type. 'auto' detects from checkpoint. "
            "Override with 'atlas' or 'dense' if auto-detection is wrong."
        ),
    )

    # ── Model config (match training defaults) ─────────────────────────
    mg = p.add_argument_group("Model config (must match training)")
    mg.add_argument("--vocab-size",          type=int,   default=1024)
    mg.add_argument("--num-layers",          type=int,   default=12)
    mg.add_argument("--model-dim",           type=int,   default=896)
    mg.add_argument("--num-heads",           type=int,   default=14)
    mg.add_argument("--num-kv-heads",        type=int,   default=7)
    mg.add_argument("--mlp-mult",            type=int,   default=3)
    mg.add_argument("--no-tie-embeddings",   action="store_true",
                    help="Disable tied embeddings (default: tied).")
    mg.add_argument("--tied-embed-init-std", type=float, default=0.005)
    mg.add_argument("--logit-softcap",       type=float, default=30.0)
    mg.add_argument("--rope-base",           type=float, default=50000.0)
    mg.add_argument("--qk-gain-init",        type=float, default=1.5)

    # ── Generation ─────────────────────────────────────────────────────
    gg = p.add_argument_group("Generation")
    gg.add_argument("--max-new-tokens",      type=int,   default=128)
    gg.add_argument("--temperature",         type=float, default=0.0,
                    help="Sampling temperature. 0 = greedy.")
    gg.add_argument("--top-k",               type=int,   default=0,
                    help="Top-k sampling. 0 = disabled.")
    gg.add_argument("--top-p",               type=float, default=1.0,
                    help="Nucleus (top-p) sampling. 1.0 = disabled.")
    gg.add_argument("--repetition-penalty",  type=float, default=1.0,
                    help="Repetition penalty. 1.0 = no penalty.")
    gg.add_argument("--seed",                type=int,   default=None,
                    help="RNG seed for deterministic sampling.")
    gg.add_argument("--stream",              action="store_true",
                    help="Print tokens as they are generated.")

    # ── Input ──────────────────────────────────────────────────────────
    ig = p.add_argument_group("Input")
    ig.add_argument("--prompt",              action="append", dest="prompts", default=[],
                    metavar="TEXT",
                    help="Prompt string. Can be given multiple times.")
    ig.add_argument("--prompt-file",         type=str, default=None,
                    help="File with one prompt per line.")
    ig.add_argument("--batch-file",          type=str, default=None,
                    help="Alias for --prompt-file.")
    ig.add_argument("--interactive",         action="store_true",
                    help="Launch interactive REPL (default when no prompts given).")

    # ── Device ─────────────────────────────────────────────────────────
    p.add_argument("--device", default="auto",
                   help="'auto', 'cuda', 'cuda:0', 'cpu', etc.")
    p.add_argument("--no-bf16", action="store_true",
                   help="Disable BF16 autocast (use FP32 everywhere).")

    return p


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # ── Device setup ───────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] using {device}", flush=True)

    # ── Seed ───────────────────────────────────────────────────────────
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        print(f"[seed] {args.seed}")

    # ── CUDA perf flags ────────────────────────────────────────────────
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Flash attention for Dense; MATH for ATLAS (cluster attention needs it)
        from torch.backends.cuda import (
            enable_cudnn_sdp, enable_flash_sdp,
            enable_math_sdp, enable_mem_efficient_sdp,
        )
        enable_cudnn_sdp(False)
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(False)
        enable_math_sdp(True)  # ATLAS needs MATH backend; Flash used by Dense blocks

    # ── Tokenizer ──────────────────────────────────────────────────────
    tok_path = args.tokenizer
    if not Path(tok_path).exists():
        parser.error(f"Tokenizer not found: {tok_path}")
    tokenizer = spm.SentencePieceProcessor(model_file=tok_path)
    print(f"[tokenizer] loaded {tok_path} (vocab={tokenizer.vocab_size()})")

    # ── Load checkpoint ────────────────────────────────────────────────
    use_atlas: Optional[bool] = None
    if args.arch == "atlas":
        use_atlas = True
    elif args.arch == "dense":
        use_atlas = False

    state_dict, ckpt_meta, arch = load_checkpoint(
        args.checkpoint, device, use_atlas=use_atlas
    )

    # ── Build model ────────────────────────────────────────────────────
    model = build_model(
        arch=arch,
        device=device,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=not args.no_tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    )

    # ── BF16 cast (matches training) ───────────────────────────────────
    if not args.no_bf16 and device.type == "cuda":
        model = model.bfloat16()
        # Keep small/control params in fp32 (same as training)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if (
                    param.ndim < 2
                    or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
                ) and param.dtype != torch.float32:
                    param.data = param.data.float()
        # CastedLinear matrices stay fp32 weight (cast at matmul time)
        for module in model.modules():
            if isinstance(module, CastedLinear):
                module.float()

    # ── Load weights ───────────────────────────────────────────────────
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys ({len(missing)}): {missing[:5]}{'…' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys ({len(unexpected)}): {unexpected[:5]}{'…' if len(unexpected)>5 else ''}")
    if not missing and not unexpected:
        print(f"[load] state_dict loaded perfectly (strict).")

    model.eval()

    # ── Print info ─────────────────────────────────────────────────────
    print_checkpoint_info(ckpt_meta, arch, model)

    # ── Collect prompts ────────────────────────────────────────────────
    prompts: List[str] = list(args.prompts)

    # --prompt-file / --batch-file
    batch_file = args.batch_file or args.prompt_file
    if batch_file:
        bf = Path(batch_file)
        if not bf.exists():
            parser.error(f"Prompt file not found: {batch_file}")
        lines = [l.rstrip("\n") for l in bf.read_text(encoding="utf-8").splitlines() if l.strip()]
        prompts.extend(lines)
        print(f"[prompts] loaded {len(lines)} prompts from {batch_file}")

    # ── Dispatch ───────────────────────────────────────────────────────
    if prompts:
        run_batch(prompts, model, tokenizer, device, args)
    else:
        # No prompts supplied → interactive
        run_interactive(model, tokenizer, device, args)


if __name__ == "__main__":
    main()

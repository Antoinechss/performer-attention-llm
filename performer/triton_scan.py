"""
Triton CUDA kernels for FAVOR+ causal attention.

Two kernels:
  1. triton_scan_forward  — prefill: causal scan over N tokens
  2. triton_decode_forward — decode: fused phi + state query for one token
"""

import math
import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ── Prefill scan kernel ───────────────────────────────────────────────────────

if _TRITON_AVAILABLE:
    @triton.jit
    def _favor_scan_kernel(
        phi_q_ptr, phi_k_ptr, v_ptr, out_ptr,
        N,
        stride_qk_b, stride_qk_h, stride_qk_n,
        stride_v_b, stride_v_h, stride_v_n,
        H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """One program per (batch, head). Streams N tokens, state in registers."""
        bh = tl.program_id(0)
        b, h = bh // H, bh % H

        base_qk = b * stride_qk_b + h * stride_qk_h
        base_v  = b * stride_v_b  + h * stride_v_h
        m_idx = tl.arange(0, BLOCK_M)
        d_idx = tl.arange(0, BLOCK_D)

        S = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        z = tl.zeros((BLOCK_M,),         dtype=tl.float32)

        for n in range(N):
            phi_k_n = tl.load(phi_k_ptr + base_qk + n * stride_qk_n + m_idx)
            v_n     = tl.load(v_ptr     + base_v  + n * stride_v_n  + d_idx)

            S = S + phi_k_n[:, None] * v_n[None, :]
            z = z + phi_k_n

            phi_q_n = tl.load(phi_q_ptr + base_qk + n * stride_qk_n + m_idx)
            num   = tl.sum(phi_q_n[:, None] * S, axis=0)
            denom = tl.sum(phi_q_n * z) + 1e-6
            tl.store(out_ptr + base_v + n * stride_v_n + d_idx, num / denom)


def triton_scan_forward(phi_q, phi_k, v):
    """Causal scan on pre-computed phi tensors. [B,H,N,M] + [B,H,N,D] → [B,H,N,D]"""
    assert _TRITON_AVAILABLE and phi_q.device.type == "cuda"
    phi_q, phi_k, v = phi_q.contiguous(), phi_k.contiguous(), v.contiguous()
    B, H, N, M = phi_q.shape
    D = v.shape[-1]
    out = torch.empty(B, H, N, D, dtype=phi_q.dtype, device=phi_q.device)
    _favor_scan_kernel[(B * H,)](
        phi_q, phi_k, v, out, N,
        phi_q.stride(0), phi_q.stride(1), phi_q.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        H=H, BLOCK_M=M, BLOCK_D=D,
    )
    return out


# ── Fused decode kernel ───────────────────────────────────────────────────────

if _TRITON_AVAILABLE:
    @triton.jit
    def _favor_decode_kernel(
        q_ptr, omega_ptr, kv_ptr, k_ptr, out_ptr,
        M,
        stride_q_b, stride_q_h, stride_om_m,
        stride_kv_b, stride_kv_h, stride_kv_m,
        stride_k_b, stride_k_h,
        stride_out_b, stride_out_h,
        H: tl.constexpr, BLOCK_D: tl.constexpr, inv_sqrt_M: tl.constexpr,
    ):
        """One program per (batch, head). Computes phi(q) on-the-fly in registers."""
        bh = tl.program_id(0)
        b, h = bh // H, bh % H
        d_idx = tl.arange(0, BLOCK_D)

        base_q   = b * stride_q_b   + h * stride_q_h
        base_kv  = b * stride_kv_b  + h * stride_kv_h
        base_k   = b * stride_k_b   + h * stride_k_h
        base_out = b * stride_out_b + h * stride_out_h

        q_n    = tl.load(q_ptr + base_q + d_idx).to(tl.float32)
        norm_x = 0.5 * tl.sum(q_n * q_n)

        # Pass 1: find max for stability
        max_log_phi = -1e30
        for m in range(M):
            omega_m = tl.load(omega_ptr + m * stride_om_m + d_idx).to(tl.float32)
            log_phi_m = tl.sum(q_n * omega_m) - norm_x
            max_log_phi = tl.where(log_phi_m > max_log_phi, log_phi_m, max_log_phi)

        # Pass 2: accumulate phi(q) @ kv_state and phi(q) @ k_state
        out   = tl.zeros([BLOCK_D], dtype=tl.float32)
        denom = 0.0
        for m in range(M):
            omega_m = tl.load(omega_ptr + m * stride_om_m + d_idx).to(tl.float32)
            phi_m = tl.exp(tl.sum(q_n * omega_m) - norm_x - max_log_phi) * inv_sqrt_M + 1e-4

            kv_m = tl.load(kv_ptr + base_kv + m * stride_kv_m + d_idx).to(tl.float32)
            out  = out + phi_m * kv_m

            k_m   = tl.load(k_ptr + base_k + m).to(tl.float32)
            denom = denom + phi_m * k_m

        tl.store(out_ptr + base_out + d_idx, out / (denom + 1e-6))


def triton_decode_forward(q, omega, kv_state, k_state):
    """Fused decode: phi(q) computed inside kernel, one pass over kv_state."""
    assert _TRITON_AVAILABLE and q.device.type == "cuda"
    q        = q.float().contiguous()
    omega    = omega.float().contiguous()
    kv_state = kv_state.float().contiguous()
    k_state  = k_state.float().contiguous()

    B, H, _, D = q.shape
    M = omega.shape[0]
    out = torch.empty(B, H, 1, D, dtype=torch.float32, device=q.device)

    _favor_decode_kernel[(B * H,)](
        q, omega, kv_state, k_state, out, M,
        q.stride(0), q.stride(1), omega.stride(0),
        kv_state.stride(0), kv_state.stride(1), kv_state.stride(2),
        k_state.stride(0), k_state.stride(1),
        out.stride(0), out.stride(1),
        H=H, BLOCK_D=D, inv_sqrt_M=1.0 / math.sqrt(M),
    )
    return out

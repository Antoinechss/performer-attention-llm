"""
Triton CUDA kernel for the FAVOR+ causal sequential scan.

Replaces the Python `for i in range(N)` loop in PerformerAttentionCore.forward()
with a single fused GPU kernel launch, keeping the running S[M,D] and z[M] state
in on-chip registers instead of bouncing through global memory on every step.

Why this is faster than the Python loop:
- Python loop: ~512 kernel launches at N=512, each with ~5µs CUDA launch overhead
- Triton kernel: 1 kernel launch, all N steps run inside the GPU without Python involvement

Requirements:
- pip install triton    (already a PyTorch dependency on CUDA builds)
- CUDA GPU             (no native macOS/MPS support)

Usage is automatic: PerformerAttentionCore.forward() dispatches here when
   device.type == "cuda"  AND  triton is installed.
"""

import math
import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def _favor_scan_kernel(
        phi_q_ptr, phi_k_ptr, v_ptr, out_ptr,
        N,
        # Strides for phi_q / phi_k tensors: shape [B, H, N, M]
        stride_qk_b, stride_qk_h, stride_qk_n,
        # Strides for v / out tensors: shape [B, H, N, D]
        stride_v_b, stride_v_h, stride_v_n,
        H:       tl.constexpr,
        BLOCK_M: tl.constexpr,  # num random features M (must be power of 2)
        BLOCK_D: tl.constexpr,  # head_dim D (must be power of 2)
    ):
        """
        One GPU program handles one (batch, head) pair.
        Grid = (B * H,).

        Maintains S[M, D] and z[M] in registers (no global memory per step).
        Streams N tokens sequentially inside the GPU — no Python loop overhead.
        """
        bh = tl.program_id(0)
        b  = bh // H
        h  = bh  % H

        # Base pointers for this (b, h) slice
        base_qk = b * stride_qk_b + h * stride_qk_h
        base_v  = b * stride_v_b  + h * stride_v_h

        m_idx = tl.arange(0, BLOCK_M)  # [M]
        d_idx = tl.arange(0, BLOCK_D)  # [D]

        # Running state — lives in registers, never written to global memory
        S = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)  # kv accumulator
        z = tl.zeros((BLOCK_M,),         dtype=tl.float32)  # k  accumulator

        for n in range(N):
            # Load phi_k[b, h, n, :] and v[b, h, n, :]
            phi_k_n = tl.load(phi_k_ptr + base_qk + n * stride_qk_n + m_idx)  # [M]
            v_n     = tl.load(v_ptr     + base_v  + n * stride_v_n  + d_idx)  # [D]

            # Causal update: include token n in the state before querying it
            S = S + phi_k_n[:, None] * v_n[None, :]  # [M, D]  outer product
            z = z + phi_k_n                           # [M]

            # Load phi_q[b, h, n, :] and compute output
            phi_q_n = tl.load(phi_q_ptr + base_qk + n * stride_qk_n + m_idx)  # [M]
            num   = tl.sum(phi_q_n[:, None] * S, axis=0)   # [D]   phi_q @ S
            denom = tl.sum(phi_q_n * z) + 1e-6              # scalar
            tl.store(out_ptr + base_v + n * stride_v_n + d_idx, num / denom)


def triton_scan_forward(phi_q: torch.Tensor, phi_k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    FAVOR+ causal sequential scan — single fused GPU kernel.

    Args:
        phi_q : [B, H, N, M] float32, contiguous — random-feature queries
        phi_k : [B, H, N, M] float32, contiguous — random-feature keys
        v     : [B, H, N, D] float32, contiguous — values

    Returns:
        out   : [B, H, N, D] float32
    """
    assert _TRITON_AVAILABLE, (
        "triton package is not installed.\n"
        "Install it with:  pip install triton\n"
        "Note: Triton requires a CUDA GPU — not available on macOS."
    )
    assert phi_q.device.type == "cuda", "Triton kernel requires a CUDA device"

    # Ensure contiguous layout for pointer arithmetic
    phi_q = phi_q.contiguous()
    phi_k = phi_k.contiguous()
    v     = v.contiguous()

    B, H, N, M = phi_q.shape
    D = v.shape[-1]

    out = torch.empty(B, H, N, D, dtype=phi_q.dtype, device=phi_q.device)

    grid = (B * H,)
    _favor_scan_kernel[grid](
        phi_q, phi_k, v, out,
        N,
        phi_q.stride(0), phi_q.stride(1), phi_q.stride(2),
        v.stride(0),     v.stride(1),     v.stride(2),
        H=H,
        BLOCK_M=M,
        BLOCK_D=D,
    )
    return out


# ── Fused prefill kernel ───────────────────────────────────────────────────────
#
# The non-fused path does:  phi_q = phi(q)  →  phi_k = phi(k)  →  scan(phi_q, phi_k, v)
# That's 10+ kernel launches (5 per phi call) + 2 large HBM round-trips for phi tensors.
#
# The fused kernel loads omega [M, D] ONCE into registers and computes phi on-the-fly
# for every token inside the sequential scan.  One kernel launch, zero HBM for phi.

if _TRITON_AVAILABLE:
    @triton.jit
    def _fused_favor_scan_kernel(
        q_ptr, k_ptr, v_ptr, omega_ptr, out_ptr,
        N, M,
        stride_b, stride_h, stride_n,  # shared strides for [B, H, N, D]
        stride_om_m,                    # omega [M, D] stride along dim 0
        H:          tl.constexpr,
        BLOCK_M:    tl.constexpr,
        BLOCK_D:    tl.constexpr,
        inv_sqrt_M: tl.constexpr,
        scale:      tl.constexpr,       # D^(-0.25), applied to q and k
    ):
        """
        One GPU program per (batch, head).
        Loads omega [M, D] once, then streams N tokens: for each token
        computes phi_k and phi_q in registers, updates the running S/z state,
        and writes the normalised output.  phi never touches HBM.
        """
        bh = tl.program_id(0)
        b  = bh // H
        h  = bh  % H

        m_idx = tl.arange(0, BLOCK_M)
        d_idx = tl.arange(0, BLOCK_D)

        base = b * stride_b + h * stride_h

        # Load omega [M, D] once — reused for all N tokens (fits in L1/registers)
        omega = tl.load(omega_ptr + m_idx[:, None] * stride_om_m
                        + d_idx[None, :]).to(tl.float32)   # [M, D]

        # Running causal state
        S = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)  # kv accumulator
        z = tl.zeros((BLOCK_M,),         dtype=tl.float32)  # k  accumulator

        for n in range(N):
            off_n = base + n * stride_n

            # Load and scale q, k, v for token n
            q_n = tl.load(q_ptr + off_n + d_idx).to(tl.float32) * scale
            k_n = tl.load(k_ptr + off_n + d_idx).to(tl.float32) * scale
            v_n = tl.load(v_ptr + off_n + d_idx).to(tl.float32)

            # All M projections at once: omega [M, D] · x [D] → [M]
            proj_k = tl.sum(omega * k_n[None, :], axis=1)   # [M]
            proj_q = tl.sum(omega * q_n[None, :], axis=1)   # [M]

            norm_k = 0.5 * tl.sum(k_n * k_n)
            norm_q = 0.5 * tl.sum(q_n * q_n)

            # phi_k: exp(proj - norm - max) / sqrt(M) + eps
            log_k   = proj_k - norm_k
            phi_k_n = tl.exp(log_k - tl.max(log_k, axis=0)) * inv_sqrt_M + 1e-4

            # phi_q: same, per-position max
            log_q   = proj_q - norm_q
            phi_q_n = tl.exp(log_q - tl.max(log_q, axis=0)) * inv_sqrt_M + 1e-4

            # Causal update: include token n before querying
            S = S + phi_k_n[:, None] * v_n[None, :]   # [M, D]
            z = z + phi_k_n                             # [M]

            # Output for token n
            num   = tl.sum(phi_q_n[:, None] * S, axis=0)   # [D]
            denom = tl.sum(phi_q_n * z) + 1e-6              # scalar
            tl.store(out_ptr + off_n + d_idx, num / denom)


def triton_fused_scan_forward(
    q:     torch.Tensor,
    k:     torch.Tensor,
    v:     torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    """
    Fused FAVOR+ causal scan — phi computed inside the kernel.

    Accepts raw q, k (NOT pre-scaled, NOT feature-mapped).
    The D^(-0.25) scaling and phi() are computed on-the-fly per token
    inside a single GPU kernel.  omega [M, D] is loaded once into registers.

    Args:
        q     : [B, H, N, D] any dtype, contiguous — raw queries
        k     : [B, H, N, D] any dtype, contiguous — raw keys
        v     : [B, H, N, D] any dtype, contiguous — values
        omega : [M, D]       float32              — random feature matrix

    Returns:
        out : [B, H, N, D] float32
    """
    assert _TRITON_AVAILABLE, "triton not installed"
    assert q.device.type == "cuda", "Triton fused scan requires CUDA"

    q     = q.contiguous()
    k     = k.contiguous()
    v     = v.contiguous()
    omega = omega.float().contiguous()

    B, H, N, D = q.shape
    M = omega.shape[0]

    out = torch.empty(B, H, N, D, dtype=torch.float32, device=q.device)

    _fused_favor_scan_kernel[(B * H,)](
        q, k, v, omega, out,
        N, M,
        q.stride(0), q.stride(1), q.stride(2),
        omega.stride(0),
        H=H,
        BLOCK_M=M,
        BLOCK_D=D,
        inv_sqrt_M=1.0 / math.sqrt(M),
        scale=D ** -0.25,
    )
    return out


# ── Fused decode kernel ────────────────────────────────────────────────────────
#
# Problem with the naive torch decode step:
#   phi_q  = phi(q)                            → 2 kernels (einsum + exp)
#   out    = einsum("bhnm,bhmd->bhnd", phi_q, kv_state)  → 1 kernel
#   denom  = einsum("bhnm,bhm->bhn",   phi_q, k_state)   → 1 kernel
#   result = out / denom                       → 1 kernel
#   Total: ~6 kernel launches, kv_state read 3× from HBM
#
# Fused approach: one kernel per (batch, head).
#   - Load q[D] into registers once.
#   - Loop over M: compute phi_q[m] in registers, accumulate out[D] and denom.
#   - Write result once.
#   kv_state and omega each read exactly once from HBM.

if _TRITON_AVAILABLE:
    @triton.jit
    def _favor_decode_kernel(
        q_ptr, omega_ptr, kv_ptr, k_ptr, out_ptr,
        M,
        stride_q_b,   stride_q_h,
        stride_om_m,
        stride_kv_b,  stride_kv_h,  stride_kv_m,
        stride_k_b,   stride_k_h,
        stride_out_b, stride_out_h,
        H:          tl.constexpr,
        BLOCK_D:    tl.constexpr,   # head_dim D, must be power of 2
        inv_sqrt_M: tl.constexpr,   # 1 / sqrt(M), baked in at compile time
    ):
        """
        One GPU program per (batch, head).
        Fuses phi(q) computation with both contractions against kv_state / k_state.
        phi_q[m] computed on-the-fly in registers — never written to HBM.
        """
        bh = tl.program_id(0)
        b  = bh // H
        h  = bh  % H

        d_idx = tl.arange(0, BLOCK_D)

        base_q   = b * stride_q_b   + h * stride_q_h
        base_kv  = b * stride_kv_b  + h * stride_kv_h
        base_k   = b * stride_k_b   + h * stride_k_h
        base_out = b * stride_out_b + h * stride_out_h

        # Load pre-scaled q[b, h, 0, :] → [D] stays in registers
        q_n    = tl.load(q_ptr + base_q + d_idx).to(tl.float32)
        norm_x = 0.5 * tl.sum(q_n * q_n)   # scalar: ||q||²/2

        # Pass 1: find max(log_phi) across M for numerical stability
        max_log_phi = -1e30
        for m in range(M):
            omega_m = tl.load(omega_ptr + m * stride_om_m + d_idx).to(tl.float32)
            log_phi_m = tl.sum(q_n * omega_m) - norm_x
            max_log_phi = tl.where(log_phi_m > max_log_phi, log_phi_m, max_log_phi)

        # Pass 2: compute phi = exp(log_phi - max) / sqrt(M) + 1e-4, accumulate
        out   = tl.zeros([BLOCK_D], dtype=tl.float32)
        denom = 0.0

        for m in range(M):
            omega_m = tl.load(omega_ptr + m * stride_om_m + d_idx).to(tl.float32)
            phi_m   = tl.exp(tl.sum(q_n * omega_m) - norm_x - max_log_phi) * inv_sqrt_M + 1e-4

            kv_m = tl.load(kv_ptr + base_kv + m * stride_kv_m + d_idx).to(tl.float32)
            out  = out + phi_m * kv_m

            k_m   = tl.load(k_ptr + base_k + m).to(tl.float32)
            denom = denom + phi_m * k_m

        tl.store(out_ptr + base_out + d_idx, out / (denom + 1e-6))


def triton_decode_forward(
    q:        torch.Tensor,
    omega:    torch.Tensor,
    kv_state: torch.Tensor,
    k_state:  torch.Tensor,
) -> torch.Tensor:
    """
    Fused FAVOR+ decode step — single GPU kernel.

    Computes:  phi(q) @ kv_state / (phi(q) @ k_state + 1e-6)
    with phi(q) computed inside the kernel (never materialised in HBM).

    Args:
        q        : [B, H, 1, D] float32  pre-scaled raw query  (q * scale)
        omega    : [M, D]       float32  random feature matrix
        kv_state : [B, H, M, D] float32  accumulated outer-product state
        k_state  : [B, H, M]    float32  accumulated key state

    Returns:
        out : [B, H, 1, D] float32
    """
    assert _TRITON_AVAILABLE, "triton not installed — pip install triton"
    assert q.device.type == "cuda", "Triton decode kernel requires CUDA"

    q        = q.float().contiguous()
    omega    = omega.float().contiguous()
    kv_state = kv_state.float().contiguous()
    k_state  = k_state.float().contiguous()

    B, H, _, D = q.shape
    M = omega.shape[0]

    out = torch.empty(B, H, 1, D, dtype=torch.float32, device=q.device)

    _favor_decode_kernel[(B * H,)](
        q, omega, kv_state, k_state, out,
        M,
        q.stride(0),         q.stride(1),
        omega.stride(0),
        kv_state.stride(0),  kv_state.stride(1),  kv_state.stride(2),
        k_state.stride(0),   k_state.stride(1),
        out.stride(0),       out.stride(1),
        H=H,
        BLOCK_D=D,
        inv_sqrt_M=1.0 / math.sqrt(M),
    )
    return out

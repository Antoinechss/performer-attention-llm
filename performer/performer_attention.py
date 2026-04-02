import torch
from torch import nn
import math
import importlib.util as _ilu
import os as _os

# Load Triton kernels if available (CUDA only)
try:
    _ts_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'triton_scan.py')
    _ts_spec = _ilu.spec_from_file_location('performer_triton_scan', _ts_path)
    _ts_mod  = _ilu.module_from_spec(_ts_spec)
    _ts_spec.loader.exec_module(_ts_mod)
    _triton_scan = _ts_mod.triton_scan_forward
    _HAS_TRITON  = _ts_mod._TRITON_AVAILABLE
except Exception:
    _HAS_TRITON  = False
    _triton_scan = None


def _sample_orf(head_dim, num_features, device=None):
    """Sample orthogonal random features with chi(d) norm scaling (FAVOR+)."""
    if device is None:
        device = torch.device("cpu")
    blocks = []
    while len(blocks) * head_dim < num_features:
        G = torch.randn(head_dim, head_dim, device=device, dtype=torch.float32)
        Q, _ = torch.linalg.qr(G)
        norms = torch.randn(head_dim, head_dim, device=device).norm(dim=1)
        blocks.append(torch.diag(norms) @ Q.T)
    return torch.cat(blocks, dim=0)[:num_features]


def _phi(x, omega, num_features, is_query=True):
    """FAVOR+ feature map with max-subtraction for numerical stability."""
    omega = omega.to(device=x.device, dtype=x.dtype)
    proj_x = torch.einsum("bhnd,md->bhnm", x, omega)
    norm_x = 0.5 * (x ** 2).sum(dim=-1, keepdim=True)
    log_phi = proj_x - norm_x
    if is_query:
        log_phi = log_phi - log_phi.max(dim=-1, keepdim=True).values
    else:
        log_phi = log_phi - log_phi.max()
    return torch.exp(log_phi) / math.sqrt(num_features) + 1e-4


def _python_scan(phi_q, phi_k, v):
    """Causal sequential scan — CPU/MPS fallback. O(M*D) memory."""
    B, H, N, M = phi_q.shape
    D = v.shape[-1]
    S = torch.zeros(B, H, M, D, dtype=phi_q.dtype, device=phi_q.device)
    z = torch.zeros(B, H, M,    dtype=phi_q.dtype, device=phi_q.device)
    out = torch.empty(B, H, N, D, dtype=phi_q.dtype, device=phi_q.device)
    for i in range(N):
        S = S + torch.einsum("bhm,bhd->bhmd", phi_k[:, :, i], v[:, :, i])
        z = z + phi_k[:, :, i]
        num   = torch.einsum("bhm,bhmd->bhd", phi_q[:, :, i], S)
        denom = (phi_q[:, :, i] * z).sum(-1, keepdim=True) + 1e-6
        out[:, :, i] = num / denom
    return out


class PerformerAttention(nn.Module):
    """Standalone Performer attention with Q/K/V projections (for testing)."""

    def __init__(self, dim, num_heads, head_dim, num_features):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_features = num_features

        self.register_buffer("omega", _sample_orf(head_dim, num_features))

        inner_dim = num_heads * head_dim
        self.q_proj   = nn.Linear(dim, inner_dim, bias=False)
        self.k_proj   = nn.Linear(dim, inner_dim, bias=False)
        self.v_proj   = nn.Linear(dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, dim)

    def phi(self, x, is_query=True):
        return _phi(x, self.omega, self.num_features, is_query)

    def forward(self, x):
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.25
        phi_q = self.phi(q * scale, is_query=True)
        phi_k = self.phi(k * scale, is_query=False)

        kv_cumsum = torch.einsum("bhnm,bhnd->bhnmd", phi_k, v).cumsum(dim=2)
        k_cumsum  = phi_k.cumsum(dim=2)
        out = torch.einsum("bhnm,bhnmd->bhnd", phi_q, kv_cumsum)
        z   = 1 / (torch.einsum("bhnm,bhnm->bhn", phi_q, k_cumsum) + 1e-6)
        out = out * z.unsqueeze(-1)

        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)


class PerformerAttentionCore(nn.Module):
    """Core FAVOR+ attention — no projections, plugs into any architecture."""

    def __init__(self, head_dim, num_features):
        super().__init__()
        self.head_dim = head_dim
        self.num_features = num_features
        self.register_buffer("omega", _sample_orf(head_dim, num_features), persistent=False)

    def phi(self, x, is_query=True):
        return _phi(x, self.omega, self.num_features, is_query)

    def forward(self, q, k, v):
        scale = q.shape[-1] ** -0.25
        phi_q = self.phi(q * scale, is_query=True)
        phi_k = self.phi(k * scale, is_query=False)

        if q.shape[2] == k.shape[2]:
            # Prefill: causal scan
            pq, pk, vf = phi_q.float(), phi_k.float(), v.float()
            if _HAS_TRITON and q.device.type == "cuda":
                out = _triton_scan(pq, pk, vf)
            else:
                out = _python_scan(pq, pk, vf)
            out = out.to(q.dtype)
        else:
            # Decode: single new token against accumulated state
            kv_sum = torch.einsum("bhnm,bhnd->bhmd", phi_k, v)
            k_sum  = phi_k.sum(dim=2)
            num    = torch.einsum("bhnm,bhmd->bhnd", phi_q, kv_sum)
            denom  = torch.einsum("bhnm,bhm->bhn", phi_q, k_sum) + 1e-6
            out    = num / denom.unsqueeze(-1)

        return out

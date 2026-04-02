import torch
from torch import nn
import math
import importlib.util as _ilu
import os as _os

# Load Triton scan kernel if available (CUDA only, requires: pip install triton)
try:
    _ts_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'triton_scan.py')
    _ts_spec = _ilu.spec_from_file_location('performer_triton_scan', _ts_path)
    _ts_mod  = _ilu.module_from_spec(_ts_spec)
    _ts_spec.loader.exec_module(_ts_mod)
    _triton_scan       = _ts_mod.triton_scan_forward
    _triton_fused_scan = getattr(_ts_mod, 'triton_fused_scan_forward', None)
    _HAS_TRITON        = _ts_mod._TRITON_AVAILABLE
except Exception:
    _HAS_TRITON        = False
    _triton_scan       = None
    _triton_fused_scan = None

"""
First full generic implementation of a Performer Attention framework, 
The first version below cannot be directly plugged into open source model, 
to pluggin open source models, use the PerformerAttentionCore class and adapt 
to model architecture
"""

class PerformerAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim, num_features):
        super().__init__()

        self.dim = dim  # Size of token embeddings, dim
        self.num_heads = num_heads  # H
        self.head_dim = head_dim  # Dimension of each query/key/value vector within one head, D
        self.num_features = num_features  # nb of features used for approximation, M
        
        # Sample omega and store it in class variables
        self.sample_features_ORF() # Orthogonal random features FAVOR+ implementation

        # total dimension of all attention heads combined
        inner_dim = num_heads * head_dim

        # projections
        self.q_proj = nn.Linear(dim, inner_dim, bias=False)  # Q = XWq
        self.k_proj = nn.Linear(dim, inner_dim, bias=False)  # K = XWk
        self.v_proj = nn.Linear(dim, inner_dim, bias=False)  # V = XWv
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.dim)  # original dimensions [B, N, H*D = dim]

    def sample_features_ORF(self):
        """
        Orthogonal Random Feature implementation (FAVOR+)
        omega.shape = [num_features, head_dim]
        Builds Omega by stacking orthogonal blocks of size [D, D],
        then scales rows by chi(D) norms for unbiased softmax estimation.
        """
        blocks = []
        while len(blocks) * self.head_dim < self.num_features:
            G = torch.randn(self.head_dim, self.head_dim)
            Q, _ = torch.linalg.qr(G)
            # chi(d) scaling: multiply orthogonal directions by norms of fresh Gaussian rows
            norms = torch.randn(self.head_dim, self.head_dim).norm(dim=1)
            blocks.append(torch.diag(norms) @ Q.T)
        stacked_blocks = torch.cat(blocks, dim=0)
        omega = stacked_blocks[: self.num_features]
        self.register_buffer("omega", omega)

    def phi(self, x, is_query=True):
        proj_x = torch.einsum("bhnd, md -> bhnm", x, self.omega)
        norm_x = 0.5 * (x**2).sum(dim=-1, keepdim=True)
        log_phi = proj_x - norm_x
        # Max-subtraction (log-sum-exp trick) prevents exp overflow
        if is_query:
            log_phi = log_phi - log_phi.max(dim=-1, keepdim=True).values
        else:
            log_phi = log_phi - log_phi.max()
        return torch.exp(log_phi) / math.sqrt(self.num_features) + 1e-4

    def reshape_heads(self, x):
        """Reshapes projected vector from [B, N, H*D] to [B, H, N, D]"""
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x):
        B, N, _ = x.shape

        # Compute key, query and value matrices: shape [B, N, H*D]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to individual heads for multi head attention [B, H, N, D]
        q = self.reshape_heads(q)
        k = self.reshape_heads(k)
        v = self.reshape_heads(v)

        # Apply feature map phi, [B, H, N, M] with M << D
        scale = self.head_dim ** -0.25
        phi_q = self.phi(q * scale, is_query=True)
        phi_k = self.phi(k * scale, is_query=False)

        # Causal masking prefix summing: dimensions dont change as we sum 
        kv = torch.einsum("bhnm,bhnd->bhnmd", phi_k, v) # [B, H, N, M, D]
        kv_cumsum = kv.cumsum(dim=2) # [B, H, N, M, D]
        k_cumsum = phi_k.cumsum(dim=2) # [B, H, N, M]

        # Compute numerator 
        out = torch.einsum("bhnm,bhnmd->bhnd", phi_q, kv_cumsum)

        # Normalization
        z = 1 / (torch.einsum("bhnm,bhnm->bhn", phi_q, k_cumsum) + 1e-6)  # [B, H, N]
        z = z.unsqueeze(-1) # [B, H, N, 1]
        out = out*z
        
        # Merge individual heads outputs  
        out = out.transpose(1, 2).contiguous().view(B, N, -1)

        # output projection
        out = self.out_proj(out)

        return out


def _python_scan(phi_q: torch.Tensor, phi_k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    FAVOR+ causal sequential scan — pure Python/PyTorch fallback.
    Used on CPU / MPS when Triton is not available.
    O(M×D) memory: never materialises the [N, M, D] cumsum tensor.
    """
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


"""
Implements the core mechanisms of performer FAVOR+ attention
This class does not handle the projections, or anything model specific (masking, RoPE etc)
Handles the feature mapping, sampling of features and normalization
"""

class PerformerAttentionCore(nn.Module):
    def __init__(self, head_dim, num_features):
        super().__init__()
        self.head_dim = head_dim
        self.num_features = num_features
        
        self.sample_features_ORF()

    def sample_features_ORF(self):
        blocks = []
        device = torch.device("cpu")

        while len(blocks) * self.head_dim < self.num_features:
            G = torch.randn(self.head_dim, self.head_dim, device=device, dtype=torch.float32)
            Q, _ = torch.linalg.qr(G)
            norms = torch.randn(self.head_dim, self.head_dim, device=device).norm(dim=1)
            blocks.append(torch.diag(norms) @ Q.T)

        omega = torch.cat(blocks, dim=0)[:self.num_features]
        self.register_buffer("omega", omega, persistent=False)

    def phi(self, x, is_query=True):
        omega = self.omega.to(device=x.device, dtype=x.dtype)
        proj_x = torch.einsum("bhnd,md->bhnm", x, omega)
        norm_x = 0.5 * (x**2).sum(dim=-1, keepdim=True)
        log_phi = proj_x - norm_x
        if is_query:
            log_phi = log_phi - log_phi.max(dim=-1, keepdim=True).values
        else:
            log_phi = log_phi - log_phi.max()
        return torch.exp(log_phi) / math.sqrt(self.num_features) + 1e-4

    def forward(self, q, k, v):
        scale = q.shape[-1] ** -0.25
        phi_q = self.phi(q * scale, is_query=True)   # [B, H, N_q, M]
        phi_k = self.phi(k * scale, is_query=False)  # [B, H, N_k, M]

        if q.shape[2] == k.shape[2]:
            # Prefill: causal FAVOR+ via sequential scan.
            pq = phi_q.float()
            pk = phi_k.float()
            vf = v.float()

            if _HAS_TRITON and q.device.type == "cuda":
                out = _triton_scan(pq, pk, vf)
            else:
                out = _python_scan(pq, pk, vf)

            out = out.to(q.dtype)
        else:
            # Decoding: N_q=1, full KV cache already accumulated
            kv_sum = torch.einsum("bhnm,bhnd->bhmd", phi_k, v)            # [B, H, M, D]
            k_sum  = phi_k.sum(dim=2)                                       # [B, H, M]
            num    = torch.einsum("bhnm,bhmd->bhnd", phi_q, kv_sum)        # [B, H, 1, D]
            denom  = torch.einsum("bhnm,bhm->bhn",   phi_q, k_sum) + 1e-6 # [B, H, 1]
            out    = num / denom.unsqueeze(-1)

        return out

# --------------- TEST ---------------------


# Instantiate the module
head_dim = 32
num_features = 64
model = PerformerAttentionCore(head_dim, num_features)

# Create dummy inputs: (batch, heads, seq_len, head_dim)
B, H, N, D = 2, 4, 10, head_dim
q = torch.randn(B, H, N, D)
k = torch.randn(B, H, N, D)
v = torch.randn(B, H, N, D)

# Forward pass
out = model(q, k, v)

# Check output
print("Output shape:", out.shape)
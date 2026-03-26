import torch
from torch import nn
import math

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
        Builds Omega by stacking orthogonal blocks of size [D, D]
        """
        blocks = []
        # Generate blocks until num_features is reached
        while len(blocks) * self.head_dim < self.num_features:
            G = torch.randn(self.head_dim, self.head_dim)  # Sample a gaussian matrix
            Q, _ = torch.linalg.qr(G)  # QR decomposition: provides orthogonal matrix Q
            blocks.append(Q.T)  # Rows of Q.T are orthonormal vectors, each row = one orthonormal feature direction 
        stacked_blocks = torch.cat(blocks, dim=0) # [k * D, D]
        omega = stacked_blocks[: self.num_features] # Trim to desired number of features to get [M, D]
        self.register_buffer("omega", omega)

    def phi(self, x):
        # Project x onto approximation space : compute wi^T * x for i in [1, m]
        # for every batch b, head h and token n
        proj_x = torch.einsum("bhnd, md -> bhnm", x, self.omega)
        # Square every coordinate and sum along d dimension
        # keepdim=True to conserve dimension for substraction against om_x
        norm_x = 0.5 * (x**2).sum(dim=-1, keepdim=True)
        phi = torch.exp(proj_x - norm_x) / math.sqrt(self.num_features)
        return phi

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
        phi_q = self.phi(q)
        phi_k = self.phi(k)

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
            G = torch.randn(self.head_dim, self.head_dim, device=device)
            Q, _ = torch.linalg.qr(G)
            blocks.append(Q.T)

        omega = torch.cat(blocks, dim=0)[:self.num_features]
        self.register_buffer("omega", omega)

    def phi(self, x):
        proj_x = torch.einsum("bhnd,md->bhnm", x, self.omega)
        norm_x = 0.5 * (x**2).sum(dim=-1, keepdim=True)
        return torch.exp(proj_x - norm_x) / math.sqrt(self.num_features)

    def forward(self, q, k, v):
        phi_q = self.phi(q)
        phi_k = self.phi(k)

        kv = torch.einsum("bhnm,bhnd->bhnmd", phi_k, v)

        kv_cumsum = kv.cumsum(dim=2)
        k_cumsum = phi_k.cumsum(dim=2)

        out = torch.einsum("bhnm,bhnmd->bhnd", phi_q, kv_cumsum)

        denom = torch.einsum("bhnm,bhnm->bhn", phi_q, k_cumsum) + 1e-6
        out = out / denom.unsqueeze(-1)

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
import torch 
from torch import nn 
from typing import Optional, Tuple

# Import your custom Performer attention core
from performer_attention import PerformerAttentionCore


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary position embeddings to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class LlamaPerformerAttention(nn.Module):
    """LLaMA attention wrapper using Performer causal kernel attention"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Grouped query attention
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        # Adding Performer Attention core engine  
        self.performer_att = PerformerAttentionCore(
            head_dim=self.head_dim,
            num_features=256
        )
         

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[object] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size, seq_length, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]

        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # Expanding K/V to match Q heads for GQA
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_output = self.performer_att(query_states, key_states, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        attn_weights = None
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    

# -------------- TEST --------------------------
if __name__ == "__main__":
    # Dummy configs
    class TestConfig:
        hidden_size = 512
        num_attention_heads = 8
        num_key_value_heads = 4
        attention_bias = False 
        attention_dropout = 0.0
        performer_num_features = 64

    # Instantiate model 
    config = TestConfig()
    attention = LlamaPerformerAttention(config, layer_idx=0)

    # Fake inputs 
    B = 2
    N = 16
    dim = config.hidden_size

    hidden_states = torch.randn(B, N, dim)

    # Mock Rotary Position Embedding
    head_dim = config.hidden_size // config.num_attention_heads

    # cos/sin should be [B, N, head_dim] before unsqueezing in apply_rotary_pos_emb
    cos = torch.randn(B, N, head_dim)
    sin = torch.randn(B, N, head_dim)

    position_embeddings = (cos, sin)

    # Forward pass
    out, weights = attention(
        hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=None,
        past_key_values=None
    )

    print(out.shape)        # should be [B, N, dim]
    print("Contains NaN:", torch.isnan(out).any())
    print("Contains Inf:", torch.isinf(out).any())


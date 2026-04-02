import torch
from torch import nn
from typing import Optional, Tuple
from performer_attention import PerformerAttentionCore


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class LlamaPerformerAttention(nn.Module):
    """Drop-in replacement for LlamaAttention using FAVOR+ linear attention."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.performer_att = PerformerAttentionCore(head_dim=self.head_dim, num_features=256)

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

        query_states = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        key_states   = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_output = self.performer_att(query_states, key_states, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        return self.o_proj(attn_output), None


if __name__ == "__main__":
    class TestConfig:
        hidden_size = 512
        num_attention_heads = 8
        num_key_value_heads = 4
        attention_bias = False
        attention_dropout = 0.0

    config = TestConfig()
    attn = LlamaPerformerAttention(config, layer_idx=0)

    B, N, dim = 2, 16, config.hidden_size
    hidden = torch.randn(B, N, dim)
    head_dim = dim // config.num_attention_heads
    pos_emb = (torch.randn(B, N, head_dim), torch.randn(B, N, head_dim))

    out, _ = attn(hidden, position_embeddings=pos_emb)
    print(f"Output: {out.shape}, NaN: {torch.isnan(out).any()}, Inf: {torch.isinf(out).any()}")

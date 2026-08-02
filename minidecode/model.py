import torch
import torch.nn.functional as F
from torch import nn

from .config import MiniDecodeConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        origin_type = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(origin_type)


class MLP(nn.Module):
    def __init__(self, config: MiniDecodeConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = F.silu(gate) * up
        return self.down_proj(hidden)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, config: MiniDecodeConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        indices = torch.arange(0, self.head_dim, 2)
        inv_freq = 1.0 / self.rope_theta ** (indices / self.head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, idxs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        origin_type = x.dtype
        freqs = idxs.unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)
        freqs = torch.cat([freqs, freqs], dim=-1).to(torch.float32)
        cos = torch.cos(freqs).to(origin_type)
        sin = torch.sin(freqs).to(origin_type)
        return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rotated = q * cos + rotate_half(q) * sin
    k_rotated = k * cos + rotate_half(k) * sin
    return q_rotated, k_rotated


def repeat_kv(
    hidden_states: torch.Tensor,
    num_key_value_groups: int,
) -> torch.Tensor:
    if num_key_value_groups == 1:
        return hidden_states
    B, S, D = hidden_states.shape[0], hidden_states.shape[2], hidden_states.shape[3]
    hidden_states = (
        hidden_states.unsqueeze(2)
        .tile([1, 1, num_key_value_groups, 1, 1])
        .reshape(B, -1, S, D)
    )
    return hidden_states


def make_causal_mask(x: torch.Tensor) -> torch.Tensor:
    S = x.shape[1]
    mask = torch.full(
        (1, 1, S, S), torch.finfo(x.dtype).min, dtype=x.dtype, device=x.device
    )
    mask = torch.triu(mask, diagonal=1)
    return mask


class Attention(nn.Module):
    def __init__(self, config: MiniDecodeConfig):
        super().__init__()
        self.q_proj = nn.Linear(
            config.hidden_size, config.query_projection_size, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.kv_projection_size, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.kv_projection_size, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.query_projection_size, config.hidden_size, bias=config.attention_bias
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, _ = hidden_states.shape
        Q = (
            self.q_proj(hidden_states)  # B, S, 2048
            .reshape(B, S, self.num_attention_heads, self.head_dim)  # B, S, 16, 128
            .transpose(1, 2)  # B, 16, S, 128
            .contiguous()  # B, 16, S, 128
        )
        K = (
            self.k_proj(hidden_states)  # B, S, 1024
            .reshape(B, S, self.num_key_value_heads, self.head_dim)  # B, S, 8, 128
            .transpose(1, 2)  # B, 8, S, 128
            .contiguous()  # B, 8, S, 128
        )
        V = (
            self.v_proj(hidden_states)  # B, S, 1024
            .reshape(B, S, self.num_key_value_heads, self.head_dim)  # B, S, 8, 128
            .transpose(1, 2)  # B, 8, S, 128
            .contiguous()  # B, 8, S, 128
        )

        Q = self.q_norm(Q)  # B, 16, S, 128
        K = self.k_norm(K)  # B, 8, S, 128
        Q, K = apply_rotary_pos_emb(
            Q, K, position_embeddings[0], position_embeddings[1]
        )
        K = repeat_kv(K, self.num_key_value_groups)  # B, 16, S, 128
        V = repeat_kv(V, self.num_key_value_groups)  # B, 16, S, 128

        scores = Q @ K.transpose(-1, -2)
        scores = scores * self.scaling

        scores = scores + attention_mask
        origin_type = hidden_states.dtype
        weights = torch.softmax(scores.to(torch.float32), dim=-1).to(
            origin_type
        )  # B, 16, S, S
        context = weights @ V  # B, 16, S, 128
        context = context.transpose(1, 2).contiguous().view(B, S, -1)  # B, S, 2048
        context = self.o_proj(context)  # B, S, 1024

        return context, weights


class DecoderLayer(nn.Module):
    def __init__(self, config: MiniDecodeConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states, position_embeddings, attention_mask
        )
        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

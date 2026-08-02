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

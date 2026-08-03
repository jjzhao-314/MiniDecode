import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MiniDecodeConfig
from .kv_cache import ContiguousKVCache
from .paged_kv_cache import PagedKVCache


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float):
        super().__init__()
        # 不用config是因为RMSNorm有时候对hidden_state做
        # 有时候对head做，所以需要一个单独的参数指定
        self.weight = nn.Parameter(torch.ones(hidden_size))  # bf16
        self.eps = eps

    def forward(
        self,
        hidden_states: torch.Tensor,  # B, S, hidden_size
    ):
        origin_type = hidden_states.dtype
        rms = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(rms + self.eps)
        return self.weight * hidden_states.to(origin_type)


class MLP(nn.Module):
    def __init__(self, config: MiniDecodeConfig):
        super().__init__()
        # Qwen3 中不存在带bias的MLP
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # B, S, hidden_size
    ):
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        result = F.silu(gate) * up
        down = self.down_proj(result)
        return down


def rotate_half(
    x: torch.Tensor,  # B, num_head, S, head_dim
):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, config: MiniDecodeConfig):
        super().__init__()
        self.theta = config.rope_theta
        self.head_dim = config.head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,  # B, num_head, S, head_dim
        positions: torch.Tensor,  # B, S
    ):
        origin_type = hidden_states.dtype
        idxs = torch.arange(
            0, self.head_dim, 2, dtype=torch.float32, device=hidden_states.device
        )  # head_dim / 2
        inv_freqs = 1.0 / self.theta ** (idxs / self.head_dim)  # head_dim / 2
        angle = (
            positions.to(torch.float32)[:, :, None] * inv_freqs[None, None, :]
        )  # B, S, head_dim / 2
        angle = torch.cat([angle, angle], dim=-1)  # B, S, head_dim
        cos = torch.cos(angle).to(origin_type)  # B, S, head_dim
        sin = torch.sin(angle).to(origin_type)  # B, S, head_dim
        return cos, sin


def apply_rotary_pos_emb(
    q,  # B, num_head, S, head_dim
    k,  # B, num_head, S, head_dim
    cos,  # B, S, head_dim
    sin,  # B, S, head_dim
):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]

    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


def repeat_kv(
    x: torch.Tensor,  # B, S, kv_head_num, head_dim
    num_key_value_groups,
):
    shape = x.shape
    return (
        x.unsqueeze(2)
        .tile(1, 1, num_key_value_groups, 1, 1)
        .view(shape[0], -1, shape[2], shape[3])
    )


class Attention(nn.Module):
    def __init__(self, config: MiniDecodeConfig, layer_idx: int):
        super().__init__()
        self.q_proj = nn.Linear(
            config.hidden_size, config.query_projection_size, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.kv_projection_size, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.kv_projection_size, bias=False
        )
        self.o_proj = nn.Linear(
            config.query_projection_size, config.hidden_size, bias=False
        )
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.scaling = config.head_dim**-0.5
        self.layer_idx = layer_idx

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups

    def forward(
        self,
        hidden_states: torch.Tensor,  # B, S, hidden_states
        PE: tuple[torch.Tensor, torch.Tensor],  # B, S, head_dim
        causal_mask: torch.Tensor,  # 1, 1, S, T
        kv_cache: ContiguousKVCache | PagedKVCache,
        block_table=None,
        slot_mapping=None,
    ):
        B, S, _ = hidden_states.shape
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        Q = (
            Q.unsqueeze(2)  # B, S, 1, hidden_states
            .reshape(B, S, self.num_attention_heads, -1)  # B, S, q_head_num, head_dim
            .transpose(1, 2)  # B, q_head_num, S, head_dim
            .contiguous()
        )
        K = (
            K.unsqueeze(2)  # B, S, 1, hidden_states
            .reshape(B, S, self.num_key_value_heads, -1)  # B, S, kv_head_num, head_dim
            .transpose(1, 2)  # B, kv_head_num, S, head_dim
            .contiguous()
        )
        V = (
            V.unsqueeze(2)  # B, S, 1, hidden_states
            .reshape(B, S, self.num_key_value_heads, -1)  # B, S, kv_head_num, head_dim
            .transpose(1, 2)  # B, kv_head_num, S, head_dim
            .contiguous()
        )

        Q = self.q_norm(Q)  # B, q_head_num, S, head_dim
        K = self.k_norm(K)  # B, kv_head_num, S, head_dim

        Q, K = apply_rotary_pos_emb(
            Q, K, PE[0], PE[1]
        )  # B, q_head_num, S, head_dim, # B, kv_head_num, S, head_dim

        # save kv cache
        if isinstance(kv_cache, ContiguousKVCache):
            k_cache, v_cache = kv_cache.update(self.layer_idx, K, V)
        elif isinstance(kv_cache, PagedKVCache):
            if block_table is None or slot_mapping is None:
                raise ValueError(
                    "block_table and slot_mapping are required for PagedKVCache"
                )
            kv_cache.write(self.layer_idx, K, V, slot_mapping)
            k_cache, v_cache = kv_cache.read(
                self.layer_idx, block_table.block_ids, block_table.num_tokens
            )
        else:
            raise TypeError("unsupported KV cache type")
        K = repeat_kv(k_cache, self.num_key_value_groups)  # B, q_head_num, S, head_dim
        V = repeat_kv(v_cache, self.num_key_value_groups)  # B, q_head_num, S, head_dim

        scores = Q @ K.transpose(-1, -2)  # B, q_head_num, S, S
        scores = scores * self.scaling

        scores = scores + causal_mask  # B, q_head_num, S, S
        origin_type = scores.dtype
        weights = torch.softmax(scores.to(torch.float32), dim=-1).to(
            origin_type
        )  # B, q_head_num, S, S

        context = weights @ V  # B, q_head_num, S, head_dim
        context = (
            context.transpose(1, 2).contiguous().view(B, S, -1)
        )  # B, S, query_projection_size

        result = self.o_proj(context)  # B, S, hidden_size
        return result


class DecoderLayer(nn.Module):
    def __init__(self, config: MiniDecodeConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,  # B, S, hidden_size
        PE: tuple[torch.Tensor, torch.Tensor],  # B, S, head_dim
        causal_mask: torch.Tensor,  # 1, 1, S, T
        kv_cache: ContiguousKVCache | PagedKVCache,
        block_table=None,
        slot_mapping=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, PE, causal_mask, kv_cache, block_table, slot_mapping
        )
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


def make_causal_mask(hidden_states: torch.Tensor):
    S = hidden_states.shape[1]
    minimal = torch.full(
        (S, S),
        torch.finfo(hidden_states.dtype).min,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )[None, None, :, :]
    minimal = torch.triu(minimal, diagonal=1)
    return minimal


class MiniDecodeModel(nn.Module):
    def __init__(self, config: MiniDecodeConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: ContiguousKVCache | PagedKVCache,
        position_ids: torch.Tensor | None = None,
        block_table=None,
    ):
        hidden_states = self.embed_tokens(input_ids)
        token_len = input_ids.shape[1]

        if isinstance(kv_caches, ContiguousKVCache):
            start = kv_caches.cur_len
            slot_mapping = None
        elif isinstance(kv_caches, PagedKVCache):
            if block_table is None:
                raise ValueError("block_table is required for PagedKVCache")
            if input_ids.shape[0] != 1:
                raise ValueError("PagedKVCache currently requires batch size 1")
            if block_table.block_size != kv_caches.block_size:
                raise ValueError("block table and cache block sizes must match")
            start = block_table.num_tokens
            slot_mapping = block_table.append_tokens(token_len)
        else:
            raise TypeError("unsupported KV cache type")

        if position_ids is None:
            end = start + token_len
            position_ids = torch.arange(start, end, device=input_ids.device).unsqueeze(
                0
            )
        causal_mask = make_causal_mask(hidden_states)
        PE = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, PE, causal_mask, kv_caches, block_table, slot_mapping
            )
        if isinstance(kv_caches, ContiguousKVCache):
            kv_caches.advance(token_len)
        return self.norm(hidden_states)


class MiniDecodeForCausalLM(nn.Module):
    def __init__(self, config: MiniDecodeConfig):
        super().__init__()
        self.config = config
        self.model = MiniDecodeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: ContiguousKVCache | PagedKVCache,
        position_ids: torch.Tensor | None = None,
        last_token_only: bool = False,
        block_table=None,
    ):
        hidden_states = self.model(
            input_ids, kv_caches, position_ids=position_ids, block_table=block_table
        )
        if last_token_only:
            logits = self.lm_head(hidden_states[:, -1:, :])
        else:
            logits = self.lm_head(hidden_states)
        return logits

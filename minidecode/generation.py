import torch

from ._C import BlockManager
from .block_table import SequenceBlockTable
from .kv_cache import ContiguousKVCache
from .model import MiniDecodeForCausalLM
from .paged_kv_cache import PagedKVCache


def greedy_generate(
    model: MiniDecodeForCausalLM,
    input_ids: torch.Tensor,  # 1, S
    max_new_tokens: int,
    eos_token_id: int | None = None,
    cache_mode="contiguous",
    block_size=16,
) -> torch.Tensor:
    B, S = input_ids.shape
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if cache_mode not in ("contiguous", "paged"):
        raise ValueError("cache_mode must be 'contiguous' or 'paged'")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if cache_mode == "paged" and B != 1:
        raise ValueError("paged generation currently requires batch size 1")

    with torch.no_grad():
        generated = input_ids
        model_input = generated
        block_table = None
        if cache_mode == "contiguous":
            kv_caches = ContiguousKVCache(
                model.config.num_hidden_layers,
                B,
                model.config.num_key_value_heads,
                S + max_new_tokens,
                model.config.head_dim,
                model.lm_head.weight.dtype,
                input_ids.device,
            )
        else:
            capacity = S + max_new_tokens
            num_blocks = (capacity + block_size - 1) // block_size
            manager = BlockManager(num_blocks)

            block_table = SequenceBlockTable(
                manager,
                block_size=block_size,
            )
            kv_caches = PagedKVCache(
                num_layers=model.config.num_hidden_layers,
                num_blocks=num_blocks,
                num_kv_heads=model.config.num_key_value_heads,
                block_size=block_size,
                head_dim=model.config.head_dim,
                dtype=model.lm_head.weight.dtype,
                device=input_ids.device,
            )

        try:
            for _ in range(max_new_tokens):
                if block_table is None:
                    output = model(model_input, kv_caches, last_token_only=True)
                else:
                    output = model(
                        model_input,
                        kv_caches,
                        last_token_only=True,
                        block_table=block_table,
                    )
                next_logits = output[:, -1, :]
                next_id = torch.argmax(next_logits, dim=-1)
                model_input = next_id[:, None]
                generated = torch.cat([generated, model_input], dim=-1)
                if next_id == eos_token_id:
                    break
            return generated
        finally:
            if block_table is not None:
                block_table.release()

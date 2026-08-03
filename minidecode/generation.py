import torch

from .kv_cache import ContiguousKVCache
from .model import MiniDecodeForCausalLM


def greedy_generate(
    model: MiniDecodeForCausalLM,
    input_ids: torch.Tensor,  # 1, S
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    B, S = input_ids.shape
    with torch.no_grad():
        generated = input_ids
        model_input = generated
        kv_caches = ContiguousKVCache(
            model.config.num_hidden_layers,
            B,
            model.config.num_key_value_heads,
            S + max_new_tokens,
            model.config.head_dim,
            model.lm_head.weight.dtype,
            input_ids.device,
        )
        for _ in range(max_new_tokens):
            output = model(model_input, kv_caches, last_token_only=True)
            next_logits = output[:, -1, :]
            next_id = torch.argmax(next_logits, dim=-1)
            model_input = next_id[:, None]
            generated = torch.cat([generated, model_input], dim=-1)
            if next_id == eos_token_id:
                break
        return generated

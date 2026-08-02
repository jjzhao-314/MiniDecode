import torch

from .model import MiniDecodeForCausalLM


def greedy_generate(
    model: MiniDecodeForCausalLM,
    input_ids: torch.Tensor,  # 1, S
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    with torch.no_grad():
        generated = input_ids
        for _ in range(max_new_tokens):
            output = model(generated)
            next_logits = output[:, -1, :]
            next_id = torch.argmax(next_logits, dim=-1)
            generated = torch.cat([generated, next_id[None, :]], dim=-1)
            if next_id == eos_token_id:
                break
        return generated

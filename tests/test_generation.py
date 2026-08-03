import torch
from torch import nn

from minidecode.config import MiniDecodeConfig
from minidecode.generation import greedy_generate
from minidecode.model import MiniDecodeForCausalLM


class PredictSequenceModel(nn.Module):
    def __init__(self, prompt_length: int, predicted_ids: list[int], vocab_size: int):
        super().__init__()
        self.prompt_length = prompt_length
        self.predicted_ids = predicted_ids
        self.vocab_size = vocab_size
        self.grad_enabled_during_forward: list[bool] = []
        self.input_lengths: list[int] = []
        self.received_position_ids: list[torch.Tensor | None] = []
        self.received_kv_caches: list[list[torch.Tensor] | None] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        kv_caches: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        self.grad_enabled_during_forward.append(torch.is_grad_enabled())
        self.input_lengths.append(input_ids.shape[1])
        self.received_position_ids.append(position_ids)
        self.received_kv_caches.append(kv_caches)
        step = len(self.input_lengths) - 1
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.vocab_size,
        )
        logits[:, -1, self.predicted_ids[step]] = 1.0
        return logits, [torch.tensor(step)]


def test_greedy_generate_appends_argmax_tokens() -> None:
    input_ids = torch.tensor([[1, 2, 3]])
    model = PredictSequenceModel(
        prompt_length=3,
        predicted_ids=[4, 5, 6],
        vocab_size=8,
    )

    actual = greedy_generate(model, input_ids, max_new_tokens=3)

    expected = torch.tensor([[1, 2, 3, 4, 5, 6]])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(input_ids, torch.tensor([[1, 2, 3]]))
    assert model.grad_enabled_during_forward == [False, False, False]
    assert model.input_lengths == [3, 1, 1]
    assert model.received_position_ids == [None, None, None]
    assert model.received_kv_caches[0] is None
    assert model.received_kv_caches[1] is not None
    assert model.received_kv_caches[2] is not None


def test_greedy_generate_stops_after_eos() -> None:
    input_ids = torch.tensor([[1, 2]])
    model = PredictSequenceModel(
        prompt_length=2,
        predicted_ids=[3, 7, 4, 5],
        vocab_size=8,
    )

    actual = greedy_generate(
        model,
        input_ids,
        max_new_tokens=4,
        eos_token_id=7,
    )

    expected = torch.tensor([[1, 2, 3, 7]])
    torch.testing.assert_close(actual, expected)
    assert model.grad_enabled_during_forward == [False, False]
    assert model.input_lengths == [2, 1]
    assert model.received_position_ids == [None, None]
    assert model.received_kv_caches[0] is None
    assert model.received_kv_caches[1] is not None


def test_greedy_generate_handles_zero_new_tokens() -> None:
    input_ids = torch.tensor([[1, 2, 3]])
    model = PredictSequenceModel(
        prompt_length=3,
        predicted_ids=[],
        vocab_size=8,
    )

    actual = greedy_generate(model, input_ids, max_new_tokens=0)

    torch.testing.assert_close(actual, input_ids)
    assert model.grad_enabled_during_forward == []
    assert model.input_lengths == []


def test_cached_generation_matches_full_sequence_recomputation() -> None:
    config = MiniDecodeConfig(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        hidden_act="silu",
        attention_bias=False,
        tie_word_embeddings=True,
    )
    model = MiniDecodeForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    expected = input_ids
    with torch.no_grad():
        for _ in range(4):
            logits, _ = model(expected)
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            expected = torch.cat([expected, next_id], dim=-1)

    actual = greedy_generate(model, input_ids, max_new_tokens=4)

    torch.testing.assert_close(actual, expected)

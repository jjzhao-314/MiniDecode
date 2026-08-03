import torch
from torch import nn

from minidecode.config import MiniDecodeConfig
from minidecode.generation import greedy_generate
from minidecode.kv_cache import ContiguousKVCache
from minidecode.model import MiniDecodeForCausalLM


class PredictSequenceModel(nn.Module):
    def __init__(self, prompt_length: int, predicted_ids: list[int], vocab_size: int):
        super().__init__()
        self.prompt_length = prompt_length
        self.predicted_ids = predicted_ids
        self.vocab_size = vocab_size
        self.config = MiniDecodeConfig(
            vocab_size=vocab_size,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
            rms_norm_eps=1e-6,
            rope_theta=1_000_000.0,
            hidden_act="silu",
            attention_bias=False,
            tie_word_embeddings=True,
        )
        self.lm_head = nn.Linear(1, 1, bias=False)
        self.grad_enabled_during_forward: list[bool] = []
        self.input_lengths: list[int] = []
        self.cache_lengths_before_forward: list[int] = []
        self.cache_object_ids: list[int] = []
        self.last_token_only_values: list[bool] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: ContiguousKVCache,
        position_ids: torch.Tensor | None = None,
        last_token_only: bool = False,
    ) -> torch.Tensor:
        self.grad_enabled_during_forward.append(torch.is_grad_enabled())
        self.input_lengths.append(input_ids.shape[1])
        self.cache_lengths_before_forward.append(kv_caches.cur_len)
        self.cache_object_ids.append(id(kv_caches))
        self.last_token_only_values.append(last_token_only)
        step = len(self.input_lengths) - 1
        output_length = 1 if last_token_only else input_ids.shape[1]
        logits = torch.zeros(
            input_ids.shape[0],
            output_length,
            self.vocab_size,
        )
        logits[:, -1, self.predicted_ids[step]] = 1.0
        kv_caches.advance(input_ids.shape[1])
        return logits


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
    assert model.cache_lengths_before_forward == [0, 3, 4]
    assert len(set(model.cache_object_ids)) == 1
    assert model.last_token_only_values == [True, True, True]


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
    assert model.cache_lengths_before_forward == [0, 2]
    assert len(set(model.cache_object_ids)) == 1
    assert model.last_token_only_values == [True, True]


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
    recompute_cache = ContiguousKVCache(
        num_layers=config.num_hidden_layers,
        batch_size=1,
        num_kv_heads=config.num_key_value_heads,
        max_seq_len=input_ids.shape[1] + 4,
        head_dim=config.head_dim,
        dtype=model.lm_head.weight.dtype,
        device=input_ids.device,
    )
    with torch.no_grad():
        for _ in range(4):
            recompute_cache.reset()
            logits = model(expected, recompute_cache)
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            expected = torch.cat([expected, next_id], dim=-1)

    actual = greedy_generate(model, input_ids, max_new_tokens=4)

    torch.testing.assert_close(actual, expected)

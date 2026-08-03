import pytest
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
        self.block_tables = []
        self.block_counts_after_forward: list[int] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: ContiguousKVCache,
        position_ids: torch.Tensor | None = None,
        last_token_only: bool = False,
        block_table=None,
    ) -> torch.Tensor:
        self.grad_enabled_during_forward.append(torch.is_grad_enabled())
        self.input_lengths.append(input_ids.shape[1])
        if block_table is None:
            cache_length = kv_caches.cur_len
        else:
            cache_length = block_table.num_tokens
            self.block_tables.append(block_table)
        self.cache_lengths_before_forward.append(cache_length)
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
        if block_table is None:
            kv_caches.advance(input_ids.shape[1])
        else:
            block_table.append_tokens(input_ids.shape[1])
            self.block_counts_after_forward.append(len(block_table.block_ids))
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


def test_paged_generation_matches_contiguous_generation() -> None:
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

    contiguous = greedy_generate(
        model,
        input_ids,
        max_new_tokens=4,
        cache_mode="contiguous",
    )
    paged = greedy_generate(
        model,
        input_ids,
        max_new_tokens=4,
        cache_mode="paged",
        block_size=2,
    )

    torch.testing.assert_close(paged, contiguous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_paged_generation_matches_contiguous_generation() -> None:
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
    model = MiniDecodeForCausalLM(config).cuda().to(torch.bfloat16).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")

    contiguous = greedy_generate(
        model,
        input_ids,
        max_new_tokens=4,
        cache_mode="contiguous",
    )
    paged = greedy_generate(
        model,
        input_ids,
        max_new_tokens=4,
        cache_mode="paged",
        block_size=2,
    )

    torch.testing.assert_close(paged, contiguous)


def test_paged_generation_stops_after_eos_and_releases_blocks() -> None:
    input_ids = torch.tensor([[1, 2, 3]])
    model = PredictSequenceModel(
        prompt_length=3,
        predicted_ids=[4, 7, 5],
        vocab_size=8,
    )

    actual = greedy_generate(
        model,
        input_ids,
        max_new_tokens=3,
        eos_token_id=7,
        cache_mode="paged",
        block_size=2,
    )

    torch.testing.assert_close(actual, torch.tensor([[1, 2, 3, 4, 7]]))
    assert model.input_lengths == [3, 1]
    assert model.block_counts_after_forward == [2, 2]
    assert len(set(map(id, model.block_tables))) == 1
    released_table = model.block_tables[0]
    assert released_table.block_ids == []
    assert released_table.num_tokens == 0
    assert (
        released_table.manager.num_free_blocks()
        == released_table.manager.num_total_blocks()
    )


def test_paged_generation_releases_blocks_when_model_raises() -> None:
    class FailingModel(PredictSequenceModel):
        def forward(self, *args, **kwargs) -> torch.Tensor:
            super().forward(*args, **kwargs)
            raise RuntimeError("model failure")

    model = FailingModel(prompt_length=3, predicted_ids=[4], vocab_size=8)

    with pytest.raises(RuntimeError, match="model failure"):
        greedy_generate(
            model,
            torch.tensor([[1, 2, 3]]),
            max_new_tokens=1,
            cache_mode="paged",
            block_size=2,
        )

    released_table = model.block_tables[0]
    assert released_table.block_ids == []
    assert released_table.num_tokens == 0
    assert (
        released_table.manager.num_free_blocks()
        == released_table.manager.num_total_blocks()
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_new_tokens": -1}, "max_new_tokens"),
        ({"max_new_tokens": 1, "cache_mode": "unknown"}, "cache_mode"),
        ({"max_new_tokens": 1, "block_size": 0}, "block_size"),
    ],
)
def test_greedy_generate_rejects_invalid_options(kwargs, message: str) -> None:
    model = PredictSequenceModel(prompt_length=1, predicted_ids=[2], vocab_size=4)

    with pytest.raises(ValueError, match=message):
        greedy_generate(model, torch.tensor([[1]]), **kwargs)


def test_paged_generation_rejects_batch_greater_than_one() -> None:
    model = PredictSequenceModel(prompt_length=1, predicted_ids=[2], vocab_size=4)

    with pytest.raises(ValueError, match="batch size 1"):
        greedy_generate(
            model,
            torch.tensor([[1], [2]]),
            max_new_tokens=1,
            cache_mode="paged",
        )

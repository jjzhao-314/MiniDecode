from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from minidecode.config import MiniDecodeConfig


def make_hf_config(**overrides: object) -> SimpleNamespace:
    values = {
        "model_type": "qwen3",
        "vocab_size": 151936,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 40960,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 1_000_000.0},
        "hidden_act": "silu",
        "attention_bias": False,
        "tie_word_embeddings": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_from_hf_config_reads_qwen3_values() -> None:
    config = MiniDecodeConfig.from_hf_config(make_hf_config())

    assert config.hidden_size == 1024
    assert config.head_dim == 128
    assert config.rope_theta == 1_000_000.0
    assert config.query_projection_size == 2048
    assert config.kv_projection_size == 1024
    assert config.num_key_value_groups == 2


def test_config_is_immutable() -> None:
    config = MiniDecodeConfig.from_hf_config(make_hf_config())

    with pytest.raises(FrozenInstanceError):
        config.num_attention_heads = 32


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("hidden_size", 0),
        ("intermediate_size", -1),
        ("num_hidden_layers", 0),
        ("head_dim", -1),
        ("max_position_embeddings", 0),
        ("rms_norm_eps", 0.0),
        ("rope_parameters", {"rope_theta": 0.0}),
    ],
)
def test_config_rejects_non_positive_values(
    field_name: str, invalid_value: object
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        MiniDecodeConfig.from_hf_config(
            make_hf_config(**{field_name: invalid_value})
        )


def test_config_rejects_non_divisible_head_counts() -> None:
    with pytest.raises(
        ValueError,
        match="num_attention_heads must be divisible by num_key_value_heads",
    ):
        MiniDecodeConfig.from_hf_config(make_hf_config(num_attention_heads=15))


def test_config_rejects_other_model_types() -> None:
    with pytest.raises(ValueError, match="expected a qwen3 config"):
        MiniDecodeConfig.from_hf_config(make_hf_config(model_type="llama"))

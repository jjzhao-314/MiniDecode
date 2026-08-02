import pytest
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm

from minidecode.config import MiniDecodeConfig
from minidecode.model import MLP, RMSNorm


def make_test_config() -> MiniDecodeConfig:
    return MiniDecodeConfig(
        vocab_size=256,
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


@pytest.mark.parametrize("shape", [(2, 3, 1024), (2, 8, 5, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rms_norm_matches_hugging_face(
    shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    eps = 1e-6
    hidden_size = shape[-1]
    actual_norm = RMSNorm(hidden_size, eps=eps).to(dtype=dtype)
    expected_norm = Qwen3RMSNorm(hidden_size, eps=eps).to(dtype=dtype)

    weight = torch.randn(hidden_size, dtype=dtype)
    with torch.no_grad():
        actual_norm.weight.copy_(weight)
        expected_norm.weight.copy_(weight)

    input_tensor = torch.randn(shape, dtype=dtype)

    actual = actual_norm(input_tensor)
    expected = expected_norm(input_tensor)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == input_tensor.shape
    assert actual.dtype == input_tensor.dtype


@pytest.mark.parametrize("hidden_size", [128, 1024])
def test_rms_norm_handles_zero_input(hidden_size: int) -> None:
    norm = RMSNorm(hidden_size)
    input_tensor = torch.zeros(2, 3, hidden_size)

    output = norm(input_tensor)

    torch.testing.assert_close(output, torch.zeros_like(input_tensor))
    assert torch.isfinite(output).all()


def test_rms_norm_initializes_weight_to_one() -> None:
    norm = RMSNorm(128)

    torch.testing.assert_close(norm.weight, torch.ones(128))


def test_mlp_projection_structure() -> None:
    config = make_test_config()
    mlp = MLP(config)

    assert isinstance(mlp.gate_proj, nn.Linear)
    assert isinstance(mlp.up_proj, nn.Linear)
    assert isinstance(mlp.down_proj, nn.Linear)
    assert mlp.gate_proj.weight.shape == (
        config.intermediate_size,
        config.hidden_size,
    )
    assert mlp.up_proj.weight.shape == (
        config.intermediate_size,
        config.hidden_size,
    )
    assert mlp.down_proj.weight.shape == (
        config.hidden_size,
        config.intermediate_size,
    )
    assert mlp.gate_proj.bias is None
    assert mlp.up_proj.bias is None
    assert mlp.down_proj.bias is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mlp_matches_hugging_face(dtype: torch.dtype) -> None:
    config = make_test_config()
    actual_mlp = MLP(config).to(dtype=dtype)
    expected_mlp = Qwen3MLP(config).to(dtype=dtype)
    expected_mlp.load_state_dict(actual_mlp.state_dict())
    input_tensor = torch.randn(2, 5, config.hidden_size, dtype=dtype)

    actual = actual_mlp(input_tensor)
    expected = expected_mlp(input_tensor)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == input_tensor.shape
    assert actual.dtype == input_tensor.dtype

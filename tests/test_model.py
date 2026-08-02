import pytest
import torch
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
)

from minidecode.config import MiniDecodeConfig
from minidecode.model import (
    MLP,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)


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


def make_hf_test_config(config: MiniDecodeConfig) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": config.rope_theta,
        },
        hidden_act=config.hidden_act,
        attention_bias=config.attention_bias,
        tie_word_embeddings=config.tie_word_embeddings,
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


def test_rotate_half_uses_split_half_layout() -> None:
    input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    actual = rotate_half(input_tensor)

    expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
    torch.testing.assert_close(actual, expected)


def test_rotary_embedding_registers_non_persistent_buffer() -> None:
    rope = RotaryEmbedding(make_test_config())

    assert rope.inv_freq.shape == (8,)
    assert rope.inv_freq.dtype == torch.float32
    assert "inv_freq" in dict(rope.named_buffers())
    assert "inv_freq" not in rope.state_dict()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotary_embedding_matches_hugging_face(dtype: torch.dtype) -> None:
    config = make_test_config()
    actual_rope = RotaryEmbedding(config)
    expected_rope = Qwen3RotaryEmbedding(make_hf_test_config(config))
    input_tensor = torch.randn(2, 5, config.hidden_size, dtype=dtype)
    position_ids = torch.tensor(
        [[0, 1, 2, 3, 4], [3, 4, 5, 6, 7]], dtype=torch.long
    )

    actual_cos, actual_sin = actual_rope(input_tensor, position_ids)
    expected_cos, expected_sin = expected_rope(input_tensor, position_ids)

    torch.testing.assert_close(actual_cos, expected_cos)
    torch.testing.assert_close(actual_sin, expected_sin)
    assert actual_cos.shape == (2, 5, config.head_dim)
    assert actual_cos.dtype == dtype
    assert actual_sin.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_apply_rotary_pos_emb_matches_hugging_face(dtype: torch.dtype) -> None:
    config = make_test_config()
    actual_rope = RotaryEmbedding(config)
    expected_rope = Qwen3RotaryEmbedding(make_hf_test_config(config))
    input_tensor = torch.randn(2, 5, config.hidden_size, dtype=dtype)
    position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
    actual_cos, actual_sin = actual_rope(input_tensor, position_ids)
    expected_cos, expected_sin = expected_rope(input_tensor, position_ids)
    query = torch.randn(
        2, config.num_attention_heads, 5, config.head_dim, dtype=dtype
    )
    key = torch.randn(
        2, config.num_key_value_heads, 5, config.head_dim, dtype=dtype
    )

    actual_query, actual_key = apply_rotary_pos_emb(
        query, key, actual_cos, actual_sin
    )
    expected_query, expected_key = hf_apply_rotary_pos_emb(
        query, key, expected_cos, expected_sin
    )

    torch.testing.assert_close(actual_query, expected_query)
    torch.testing.assert_close(actual_key, expected_key)
    torch.testing.assert_close(actual_query[:, :, 0], query[:, :, 0])
    torch.testing.assert_close(actual_key[:, :, 0], key[:, :, 0])
    assert actual_query.shape == query.shape
    assert actual_key.shape == key.shape

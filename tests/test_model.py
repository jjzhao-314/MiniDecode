import pytest
import torch
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
)

from minidecode.config import MiniDecodeConfig
from minidecode.model import (
    Attention,
    DecoderLayer,
    MLP,
    MiniDecodeForCausalLM,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    make_causal_mask,
    repeat_kv,
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotary_embedding_matches_hugging_face(dtype: torch.dtype) -> None:
    config = make_test_config()
    actual_rope = RotaryEmbedding(config).to(dtype=dtype)
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


def test_repeat_kv_preserves_grouped_head_order() -> None:
    hidden_states = torch.tensor([[[[10.0]], [[20.0]]]])

    actual = repeat_kv(hidden_states, num_key_value_groups=2)

    expected = torch.tensor([[[[10.0]], [[10.0]], [[20.0]], [[20.0]]]])
    torch.testing.assert_close(actual, expected)


def test_repeat_kv_returns_input_for_single_group() -> None:
    hidden_states = torch.randn(2, 4, 3, 8)

    actual = repeat_kv(hidden_states, num_key_value_groups=1)

    assert actual is hidden_states


def test_make_causal_mask_blocks_future_positions() -> None:
    input_tensor = torch.randn(2, 3, 64)

    actual = make_causal_mask(input_tensor)

    minimum = torch.finfo(input_tensor.dtype).min
    expected = torch.tensor(
        [[[[0.0, minimum, minimum], [0.0, 0.0, minimum], [0.0, 0.0, 0.0]]]]
    )
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (1, 1, 3, 3)
    assert actual.dtype == input_tensor.dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_matches_hugging_face(dtype: torch.dtype) -> None:
    config = make_test_config()
    hf_config = make_hf_test_config(config)
    hf_config._attn_implementation = "eager"
    actual_attention = Attention(config).to(dtype=dtype)
    expected_attention = Qwen3Attention(hf_config, layer_idx=0).to(dtype=dtype)
    expected_attention.load_state_dict(actual_attention.state_dict())
    hidden_states = torch.randn(2, 5, config.hidden_size, dtype=dtype)
    position_ids = torch.tensor([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6]])
    actual_position_embeddings = RotaryEmbedding(config)(
        hidden_states, position_ids
    )
    expected_position_embeddings = Qwen3RotaryEmbedding(hf_config)(
        hidden_states, position_ids
    )
    attention_mask = make_causal_mask(hidden_states)

    actual_output, actual_weights = actual_attention(
        hidden_states, actual_position_embeddings, attention_mask
    )
    expected_output, expected_weights = expected_attention(
        hidden_states, expected_position_embeddings, attention_mask
    )

    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_weights, expected_weights)
    torch.testing.assert_close(
        actual_weights.sum(dim=-1), torch.ones_like(actual_weights[..., 0])
    )
    assert (actual_weights.triu(diagonal=1) == 0).all()
    assert actual_output.shape == hidden_states.shape
    assert actual_weights.shape == (
        2,
        config.num_attention_heads,
        5,
        5,
    )
    assert actual_output.dtype == dtype
    assert actual_weights.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decoder_layer_matches_hugging_face(dtype: torch.dtype) -> None:
    config = make_test_config()
    hf_config = make_hf_test_config(config)
    hf_config._attn_implementation = "eager"
    actual_layer = DecoderLayer(config).to(dtype=dtype)
    expected_layer = Qwen3DecoderLayer(hf_config, layer_idx=0).to(dtype=dtype)
    expected_layer.load_state_dict(actual_layer.state_dict())
    hidden_states = torch.randn(2, 5, config.hidden_size, dtype=dtype)
    position_ids = torch.tensor([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6]])
    actual_position_embeddings = RotaryEmbedding(config)(
        hidden_states, position_ids
    )
    expected_position_embeddings = Qwen3RotaryEmbedding(hf_config)(
        hidden_states, position_ids
    )
    attention_mask = make_causal_mask(hidden_states)

    actual = actual_layer(
        hidden_states, actual_position_embeddings, attention_mask
    )
    expected = expected_layer(
        hidden_states,
        attention_mask=attention_mask,
        position_embeddings=expected_position_embeddings,
    )

    torch.testing.assert_close(actual, expected)
    assert actual.shape == hidden_states.shape
    assert actual.dtype == dtype


def test_causal_lm_ties_embedding_and_lm_head_weights() -> None:
    model = MiniDecodeForCausalLM(make_test_config())

    assert model.lm_head.weight is model.model.embed_tokens.weight


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_causal_lm_matches_hugging_face(dtype: torch.dtype) -> None:
    config = make_test_config()
    hf_config = make_hf_test_config(config)
    hf_config._attn_implementation = "eager"
    actual_model = MiniDecodeForCausalLM(config).to(dtype=dtype).eval()
    expected_model = Qwen3ForCausalLM(hf_config).to(dtype=dtype).eval()
    expected_model.model.rotary_emb = Qwen3RotaryEmbedding(hf_config)
    expected_model.load_state_dict(actual_model.state_dict())
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [8, 7, 6, 5, 4]])
    position_ids = torch.tensor([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6]])

    with torch.no_grad():
        actual = actual_model(input_ids, position_ids)
        expected = expected_model(
            input_ids=input_ids,
            position_ids=position_ids,
        ).logits

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 5, config.vocab_size)
    assert actual.dtype == dtype

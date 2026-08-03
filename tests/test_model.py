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

from minidecode import _C
from minidecode.block_table import SequenceBlockTable
from minidecode.config import MiniDecodeConfig
from minidecode.kv_cache import ContiguousKVCache
from minidecode.model import (
    Attention,
    DecoderLayer,
    MLP,
    MiniDecodeForCausalLM,
    MiniDecodeModel,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    make_causal_mask,
    repeat_kv,
    rotate_half,
)
from minidecode.paged_kv_cache import PagedKVCache


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


def make_test_cache(
    config: MiniDecodeConfig,
    *,
    batch_size: int = 2,
    max_seq_len: int = 5,
    dtype: torch.dtype = torch.float32,
) -> ContiguousKVCache:
    return ContiguousKVCache(
        num_layers=config.num_hidden_layers,
        batch_size=batch_size,
        num_kv_heads=config.num_key_value_heads,
        max_seq_len=max_seq_len,
        head_dim=config.head_dim,
        dtype=dtype,
        device=torch.device("cpu"),
    )


def make_test_paged_cache(
    config: MiniDecodeConfig,
    *,
    num_blocks: int = 4,
    block_size: int = 4,
    dtype: torch.dtype = torch.float32,
) -> PagedKVCache:
    return PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=num_blocks,
        num_kv_heads=config.num_key_value_heads,
        block_size=block_size,
        head_dim=config.head_dim,
        dtype=dtype,
        device=torch.device("cpu"),
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
    norm = RMSNorm(hidden_size, eps=1e-6)
    input_tensor = torch.zeros(2, 3, hidden_size)

    output = norm(input_tensor)

    torch.testing.assert_close(output, torch.zeros_like(input_tensor))
    assert torch.isfinite(output).all()


def test_rms_norm_initializes_weight_to_one() -> None:
    norm = RMSNorm(128, eps=1e-6)

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

    torch.testing.assert_close(actual, hidden_states)


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
    actual_attention = Attention(config, layer_idx=0).to(dtype=dtype)
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
    kv_cache = make_test_cache(config, dtype=dtype)

    actual_output = actual_attention(
        hidden_states, actual_position_embeddings, attention_mask, kv_cache
    )
    expected_output, expected_weights = expected_attention(
        hidden_states, expected_position_embeddings, attention_mask
    )

    torch.testing.assert_close(actual_output, expected_output)
    assert actual_output.shape == hidden_states.shape
    assert expected_weights.shape == (
        2,
        config.num_attention_heads,
        5,
        5,
    )
    assert actual_output.dtype == dtype
    assert expected_weights.dtype == dtype
    assert kv_cache.k_cache[0, :, :, :5].shape == (
        2,
        config.num_key_value_heads,
        5,
        config.head_dim,
    )
    assert kv_cache.v_cache[0, :, :, :5].shape == kv_cache.k_cache[
        0, :, :, :5
    ].shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_paged_prefill_matches_contiguous(dtype: torch.dtype) -> None:
    config = make_test_config()
    attention = Attention(config, layer_idx=0).to(dtype=dtype).eval()
    rotary_embedding = RotaryEmbedding(config)
    hidden_states = torch.randn(1, 6, config.hidden_size, dtype=dtype)
    position_ids = torch.arange(6).unsqueeze(0)
    position_embeddings = rotary_embedding(hidden_states, position_ids)
    attention_mask = make_causal_mask(hidden_states)

    contiguous_cache = make_test_cache(
        config, batch_size=1, max_seq_len=6, dtype=dtype
    )
    paged_cache = make_test_paged_cache(config, block_size=4, dtype=dtype)
    block_table = SequenceBlockTable(_C.BlockManager(4), block_size=4)
    slot_mapping = block_table.append_tokens(6)

    with torch.no_grad():
        contiguous_output = attention(
            hidden_states,
            position_embeddings,
            attention_mask,
            contiguous_cache,
        )
        paged_output = attention(
            hidden_states,
            position_embeddings,
            attention_mask,
            paged_cache,
            block_table,
            slot_mapping,
        )

    paged_k, paged_v = paged_cache.read(
        layer_idx=0,
        block_ids=block_table.block_ids,
        num_tokens=block_table.num_tokens,
    )
    torch.testing.assert_close(paged_output, contiguous_output)
    torch.testing.assert_close(paged_k, contiguous_cache.k_cache[0, :, :, :6])
    torch.testing.assert_close(paged_v, contiguous_cache.v_cache[0, :, :, :6])
    assert len(block_table.block_ids) == 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_paged_decode_matches_contiguous(dtype: torch.dtype) -> None:
    config = make_test_config()
    attention = Attention(config, layer_idx=0).to(dtype=dtype).eval()
    rotary_embedding = RotaryEmbedding(config)
    hidden_states = torch.randn(1, 6, config.hidden_size, dtype=dtype)

    contiguous_cache = make_test_cache(
        config, batch_size=1, max_seq_len=6, dtype=dtype
    )
    paged_cache = make_test_paged_cache(config, block_size=4, dtype=dtype)
    block_table = SequenceBlockTable(_C.BlockManager(4), block_size=4)

    prefill_hidden_states = hidden_states[:, :5]
    prefill_positions = torch.arange(5).unsqueeze(0)
    prefill_position_embeddings = rotary_embedding(
        prefill_hidden_states, prefill_positions
    )
    prefill_mask = make_causal_mask(prefill_hidden_states)
    prefill_slots = block_table.append_tokens(5)

    with torch.no_grad():
        attention(
            prefill_hidden_states,
            prefill_position_embeddings,
            prefill_mask,
            contiguous_cache,
        )
        attention(
            prefill_hidden_states,
            prefill_position_embeddings,
            prefill_mask,
            paged_cache,
            block_table,
            prefill_slots,
        )
        contiguous_cache.advance(5)

        decode_hidden_states = hidden_states[:, 5:]
        decode_positions = torch.tensor([[5]])
        decode_position_embeddings = rotary_embedding(
            decode_hidden_states, decode_positions
        )
        decode_mask = torch.zeros(1, 1, 1, 6, dtype=dtype)
        decode_slots = block_table.append_tokens(1)

        contiguous_output = attention(
            decode_hidden_states,
            decode_position_embeddings,
            decode_mask,
            contiguous_cache,
        )
        paged_output = attention(
            decode_hidden_states,
            decode_position_embeddings,
            decode_mask,
            paged_cache,
            block_table,
            decode_slots,
        )

    paged_k, paged_v = paged_cache.read(
        layer_idx=0,
        block_ids=block_table.block_ids,
        num_tokens=block_table.num_tokens,
    )
    torch.testing.assert_close(paged_output, contiguous_output)
    torch.testing.assert_close(paged_k, contiguous_cache.k_cache[0, :, :, :6])
    torch.testing.assert_close(paged_v, contiguous_cache.v_cache[0, :, :, :6])
    assert decode_slots[0] % block_table.block_size == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decoder_layer_matches_hugging_face(dtype: torch.dtype) -> None:
    config = make_test_config()
    hf_config = make_hf_test_config(config)
    hf_config._attn_implementation = "eager"
    actual_layer = DecoderLayer(config, layer_idx=0).to(dtype=dtype)
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
    kv_cache = make_test_cache(config, dtype=dtype)

    actual = actual_layer(
        hidden_states, actual_position_embeddings, attention_mask, kv_cache
    )
    expected = expected_layer(
        hidden_states,
        attention_mask=attention_mask,
        position_embeddings=expected_position_embeddings,
    )

    torch.testing.assert_close(actual, expected)
    assert actual.shape == hidden_states.shape
    assert actual.dtype == dtype
    assert kv_cache.k_cache[0, :, :, :5].shape == (
        2,
        config.num_key_value_heads,
        5,
        config.head_dim,
    )
    assert kv_cache.v_cache[0, :, :, :5].shape == kv_cache.k_cache[
        0, :, :, :5
    ].shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decoder_layer_paged_prefill_matches_contiguous(
    dtype: torch.dtype,
) -> None:
    config = make_test_config()
    layer = DecoderLayer(config, layer_idx=0).to(dtype=dtype).eval()
    rotary_embedding = RotaryEmbedding(config)
    hidden_states = torch.randn(1, 6, config.hidden_size, dtype=dtype)
    position_ids = torch.arange(6).unsqueeze(0)
    position_embeddings = rotary_embedding(hidden_states, position_ids)
    attention_mask = make_causal_mask(hidden_states)

    contiguous_cache = make_test_cache(
        config, batch_size=1, max_seq_len=6, dtype=dtype
    )
    paged_cache = make_test_paged_cache(config, block_size=4, dtype=dtype)
    block_table = SequenceBlockTable(_C.BlockManager(4), block_size=4)
    slot_mapping = block_table.append_tokens(6)

    with torch.no_grad():
        contiguous_output = layer(
            hidden_states,
            position_embeddings,
            attention_mask,
            contiguous_cache,
        )
        paged_output = layer(
            hidden_states,
            position_embeddings,
            attention_mask,
            paged_cache,
            block_table,
            slot_mapping,
        )

    paged_k, paged_v = paged_cache.read(
        layer_idx=0,
        block_ids=block_table.block_ids,
        num_tokens=block_table.num_tokens,
    )
    torch.testing.assert_close(paged_output, contiguous_output)
    torch.testing.assert_close(paged_k, contiguous_cache.k_cache[0, :, :, :6])
    torch.testing.assert_close(paged_v, contiguous_cache.v_cache[0, :, :, :6])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decoder_layer_paged_decode_matches_contiguous(dtype: torch.dtype) -> None:
    config = make_test_config()
    layer = DecoderLayer(config, layer_idx=0).to(dtype=dtype).eval()
    rotary_embedding = RotaryEmbedding(config)
    hidden_states = torch.randn(1, 6, config.hidden_size, dtype=dtype)

    contiguous_cache = make_test_cache(
        config, batch_size=1, max_seq_len=6, dtype=dtype
    )
    paged_cache = make_test_paged_cache(config, block_size=4, dtype=dtype)
    block_table = SequenceBlockTable(_C.BlockManager(4), block_size=4)

    prefill_hidden_states = hidden_states[:, :5]
    prefill_position_embeddings = rotary_embedding(
        prefill_hidden_states, torch.arange(5).unsqueeze(0)
    )
    prefill_mask = make_causal_mask(prefill_hidden_states)
    prefill_slots = block_table.append_tokens(5)

    with torch.no_grad():
        layer(
            prefill_hidden_states,
            prefill_position_embeddings,
            prefill_mask,
            contiguous_cache,
        )
        layer(
            prefill_hidden_states,
            prefill_position_embeddings,
            prefill_mask,
            paged_cache,
            block_table,
            prefill_slots,
        )
        contiguous_cache.advance(5)

        decode_hidden_states = hidden_states[:, 5:]
        decode_position_embeddings = rotary_embedding(
            decode_hidden_states, torch.tensor([[5]])
        )
        decode_mask = torch.zeros(1, 1, 1, 6, dtype=dtype)
        decode_slots = block_table.append_tokens(1)

        contiguous_output = layer(
            decode_hidden_states,
            decode_position_embeddings,
            decode_mask,
            contiguous_cache,
        )
        paged_output = layer(
            decode_hidden_states,
            decode_position_embeddings,
            decode_mask,
            paged_cache,
            block_table,
            decode_slots,
        )

    paged_k, paged_v = paged_cache.read(
        layer_idx=0,
        block_ids=block_table.block_ids,
        num_tokens=block_table.num_tokens,
    )
    torch.testing.assert_close(paged_output, contiguous_output)
    torch.testing.assert_close(paged_k, contiguous_cache.k_cache[0, :, :, :6])
    torch.testing.assert_close(paged_v, contiguous_cache.v_cache[0, :, :, :6])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decoder_layer_cached_decode_matches_full_sequence(
    dtype: torch.dtype,
) -> None:
    config = make_test_config()
    layer = DecoderLayer(config, layer_idx=0).to(dtype=dtype).eval()
    rotary_embedding = RotaryEmbedding(config)
    hidden_states = torch.randn(2, 5, config.hidden_size, dtype=dtype)

    full_position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
    full_position_embeddings = rotary_embedding(hidden_states, full_position_ids)
    full_mask = make_causal_mask(hidden_states)

    prefill_hidden_states = hidden_states[:, :4]
    prefill_position_ids = full_position_ids[:, :4]
    prefill_position_embeddings = rotary_embedding(
        prefill_hidden_states, prefill_position_ids
    )
    prefill_mask = make_causal_mask(prefill_hidden_states)

    decode_hidden_states = hidden_states[:, 4:]
    decode_position_ids = full_position_ids[:, 4:]
    decode_position_embeddings = rotary_embedding(
        decode_hidden_states, decode_position_ids
    )
    decode_mask = torch.zeros(1, 1, 1, 5, dtype=dtype)
    full_cache = make_test_cache(config, max_seq_len=5, dtype=dtype)
    incremental_cache = make_test_cache(config, max_seq_len=5, dtype=dtype)

    with torch.no_grad():
        full_output = layer(
            hidden_states, full_position_embeddings, full_mask, full_cache
        )
        full_cache.advance(5)
        layer(
            prefill_hidden_states,
            prefill_position_embeddings,
            prefill_mask,
            incremental_cache,
        )
        incremental_cache.advance(4)
        decode_output = layer(
            decode_hidden_states,
            decode_position_embeddings,
            decode_mask,
            incremental_cache,
        )
        incremental_cache.advance(1)

    torch.testing.assert_close(decode_output, full_output[:, 4:])
    torch.testing.assert_close(
        incremental_cache.k_cache[0], full_cache.k_cache[0]
    )
    torch.testing.assert_close(
        incremental_cache.v_cache[0], full_cache.v_cache[0]
    )
    assert incremental_cache.cur_len == 5


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_model_cached_decode_matches_full_sequence(dtype: torch.dtype) -> None:
    config = make_test_config()
    model = MiniDecodeModel(config).to(dtype=dtype).eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [8, 7, 6, 5, 4]])
    full_cache = make_test_cache(config, max_seq_len=5, dtype=dtype)
    incremental_cache = make_test_cache(config, max_seq_len=5, dtype=dtype)
    key_pointer = incremental_cache.k_cache.data_ptr()
    value_pointer = incremental_cache.v_cache.data_ptr()

    with torch.no_grad():
        full_output = model(input_ids, full_cache)
        model(input_ids[:, :4], incremental_cache)
        decode_output = model(input_ids[:, 4:], incremental_cache)

    torch.testing.assert_close(decode_output, full_output[:, 4:])
    torch.testing.assert_close(incremental_cache.k_cache, full_cache.k_cache)
    torch.testing.assert_close(incremental_cache.v_cache, full_cache.v_cache)
    assert incremental_cache.cur_len == 5
    assert incremental_cache.k_cache.data_ptr() == key_pointer
    assert incremental_cache.v_cache.data_ptr() == value_pointer


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_model_paged_prefill_and_decode_match_contiguous(
    dtype: torch.dtype,
) -> None:
    config = make_test_config()
    model = MiniDecodeModel(config).to(dtype=dtype).eval()
    prompt_ids = torch.tensor([[1, 2, 3, 4, 5]])
    decode_ids = torch.tensor([[6]])

    contiguous_cache = make_test_cache(
        config, batch_size=1, max_seq_len=6, dtype=dtype
    )
    paged_cache = make_test_paged_cache(config, block_size=4, dtype=dtype)
    block_table = SequenceBlockTable(_C.BlockManager(4), block_size=4)

    with torch.no_grad():
        contiguous_prefill = model(prompt_ids, contiguous_cache)
        paged_prefill = model(prompt_ids, paged_cache, block_table=block_table)
        contiguous_decode = model(decode_ids, contiguous_cache)
        paged_decode = model(decode_ids, paged_cache, block_table=block_table)

    torch.testing.assert_close(paged_prefill, contiguous_prefill)
    torch.testing.assert_close(paged_decode, contiguous_decode)
    assert contiguous_cache.cur_len == 6
    assert block_table.num_tokens == 6
    assert len(block_table.block_ids) == 2

    for layer_idx in range(config.num_hidden_layers):
        paged_k, paged_v = paged_cache.read(
            layer_idx=layer_idx,
            block_ids=block_table.block_ids,
            num_tokens=block_table.num_tokens,
        )
        torch.testing.assert_close(
            paged_k, contiguous_cache.k_cache[layer_idx, :, :, :6]
        )
        torch.testing.assert_close(
            paged_v, contiguous_cache.v_cache[layer_idx, :, :, :6]
        )


def test_model_paged_supports_explicit_position_ids() -> None:
    config = make_test_config()
    model = MiniDecodeModel(config).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    position_ids = torch.tensor([[7, 8, 9]])
    contiguous_cache = make_test_cache(config, batch_size=1, max_seq_len=3)
    paged_cache = make_test_paged_cache(config, block_size=4)
    block_table = SequenceBlockTable(_C.BlockManager(2), block_size=4)

    with torch.no_grad():
        contiguous_output = model(
            input_ids, contiguous_cache, position_ids=position_ids
        )
        paged_output = model(
            input_ids,
            paged_cache,
            position_ids=position_ids,
            block_table=block_table,
        )

    torch.testing.assert_close(paged_output, contiguous_output)
    assert block_table.num_tokens == 3


def test_model_paged_requires_block_table() -> None:
    config = make_test_config()
    model = MiniDecodeModel(config).eval()
    paged_cache = make_test_paged_cache(config)

    with pytest.raises(ValueError, match="block_table is required"):
        model(torch.tensor([[1]]), paged_cache)


def test_model_paged_rejects_batch_before_allocating() -> None:
    config = make_test_config()
    model = MiniDecodeModel(config).eval()
    paged_cache = make_test_paged_cache(config)
    block_table = SequenceBlockTable(_C.BlockManager(2), block_size=4)

    with pytest.raises(ValueError, match="batch size 1"):
        model(torch.tensor([[1], [2]]), paged_cache, block_table=block_table)

    assert block_table.num_tokens == 0


def test_model_paged_rejects_mismatched_block_size_before_allocating() -> None:
    config = make_test_config()
    model = MiniDecodeModel(config).eval()
    paged_cache = make_test_paged_cache(config, block_size=4)
    block_table = SequenceBlockTable(_C.BlockManager(2), block_size=8)

    with pytest.raises(ValueError, match="block sizes must match"):
        model(torch.tensor([[1]]), paged_cache, block_table=block_table)

    assert block_table.num_tokens == 0


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
    kv_cache = make_test_cache(config, max_seq_len=5, dtype=dtype)

    with torch.no_grad():
        actual = actual_model(input_ids, kv_cache, position_ids=position_ids)
        expected = expected_model(
            input_ids=input_ids,
            position_ids=position_ids,
        ).logits

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 5, config.vocab_size)
    assert actual.dtype == dtype
    assert kv_cache.cur_len == 5


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_causal_lm_cached_decode_matches_full_sequence(dtype: torch.dtype) -> None:
    config = make_test_config()
    model = MiniDecodeForCausalLM(config).to(dtype=dtype).eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [8, 7, 6, 5, 4]])
    full_cache = make_test_cache(config, max_seq_len=5, dtype=dtype)
    incremental_cache = make_test_cache(config, max_seq_len=5, dtype=dtype)

    with torch.no_grad():
        full_logits = model(input_ids, full_cache)
        model(input_ids[:, :4], incremental_cache)
        decode_logits = model(input_ids[:, 4:], incremental_cache)

    torch.testing.assert_close(decode_logits, full_logits[:, 4:])
    torch.testing.assert_close(incremental_cache.k_cache, full_cache.k_cache)
    torch.testing.assert_close(incremental_cache.v_cache, full_cache.v_cache)
    assert incremental_cache.cur_len == 5


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_causal_lm_paged_prefill_and_decode_match_contiguous(
    dtype: torch.dtype,
) -> None:
    config = make_test_config()
    model = MiniDecodeForCausalLM(config).to(dtype=dtype).eval()
    prompt_ids = torch.tensor([[1, 2, 3, 4, 5]])
    decode_ids = torch.tensor([[6]])

    contiguous_cache = make_test_cache(
        config, batch_size=1, max_seq_len=6, dtype=dtype
    )
    paged_cache = make_test_paged_cache(config, block_size=4, dtype=dtype)
    block_table = SequenceBlockTable(_C.BlockManager(4), block_size=4)

    with torch.no_grad():
        contiguous_prefill = model(prompt_ids, contiguous_cache)
        paged_prefill = model(prompt_ids, paged_cache, block_table=block_table)
        contiguous_decode = model(decode_ids, contiguous_cache)
        paged_decode = model(decode_ids, paged_cache, block_table=block_table)

    torch.testing.assert_close(paged_prefill, contiguous_prefill)
    torch.testing.assert_close(paged_decode, contiguous_decode)
    assert paged_prefill.shape == (1, 5, config.vocab_size)
    assert paged_decode.shape == (1, 1, config.vocab_size)
    assert block_table.num_tokens == 6


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_causal_lm_paged_last_token_logits_match_full_logits(
    dtype: torch.dtype,
) -> None:
    config = make_test_config()
    model = MiniDecodeForCausalLM(config).to(dtype=dtype).eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    full_cache = make_test_paged_cache(config, block_size=4, dtype=dtype)
    full_table = SequenceBlockTable(_C.BlockManager(4), block_size=4)
    last_cache = make_test_paged_cache(config, block_size=4, dtype=dtype)
    last_table = SequenceBlockTable(_C.BlockManager(4), block_size=4)

    with torch.no_grad():
        full_logits = model(input_ids, full_cache, block_table=full_table)
        last_logits = model(
            input_ids,
            last_cache,
            last_token_only=True,
            block_table=last_table,
        )

    torch.testing.assert_close(last_logits, full_logits[:, -1:])
    assert full_logits.shape == (1, 5, config.vocab_size)
    assert last_logits.shape == (1, 1, config.vocab_size)
    assert full_table.num_tokens == last_table.num_tokens == 5


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_causal_lm_last_token_logits_match_full_logits(dtype: torch.dtype) -> None:
    config = make_test_config()
    model = MiniDecodeForCausalLM(config).to(dtype=dtype).eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [8, 7, 6, 5, 4]])
    full_cache = make_test_cache(config, max_seq_len=5, dtype=dtype)
    last_token_cache = make_test_cache(config, max_seq_len=5, dtype=dtype)

    with torch.no_grad():
        full_logits = model(input_ids, full_cache)
        last_token_logits = model(
            input_ids,
            last_token_cache,
            last_token_only=True,
        )

    assert full_logits.shape == (2, 5, config.vocab_size)
    assert last_token_logits.shape == (2, 1, config.vocab_size)
    torch.testing.assert_close(last_token_logits, full_logits[:, -1:])

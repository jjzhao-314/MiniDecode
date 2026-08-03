import pytest
import torch

from minidecode.kv_cache import ContiguousKVCache


def make_cache(max_seq_len: int = 6) -> ContiguousKVCache:
    return ContiguousKVCache(
        num_layers=2,
        batch_size=2,
        num_kv_heads=3,
        max_seq_len=max_seq_len,
        head_dim=4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )


def make_kv(num_tokens: int, value: float) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (2, 3, num_tokens, 4)
    key = torch.full(shape, value, dtype=torch.bfloat16)
    value_tensor = torch.full(shape, value + 10, dtype=torch.bfloat16)
    return key, value_tensor


def test_contiguous_kv_cache_allocates_expected_storage() -> None:
    cache = make_cache()

    assert cache.k_cache.shape == (2, 2, 3, 6, 4)
    assert cache.v_cache.shape == (2, 2, 3, 6, 4)
    assert cache.k_cache.dtype == torch.bfloat16
    assert cache.v_cache.dtype == torch.bfloat16
    assert cache.k_cache.device.type == "cpu"
    assert cache.cur_len == 0
    assert cache.cap == 6


def test_update_writes_each_layer_without_advancing_length() -> None:
    cache = make_cache()
    layer_0_key, layer_0_value = make_kv(num_tokens=3, value=1)
    layer_1_key, layer_1_value = make_kv(num_tokens=3, value=2)

    layer_0_present = cache.update(0, layer_0_key, layer_0_value)
    layer_1_present = cache.update(1, layer_1_key, layer_1_value)

    assert cache.cur_len == 0
    assert layer_0_present[0].shape == (2, 3, 3, 4)
    assert layer_0_present[1].shape == (2, 3, 3, 4)
    torch.testing.assert_close(layer_0_present[0], layer_0_key)
    torch.testing.assert_close(layer_0_present[1], layer_0_value)
    torch.testing.assert_close(layer_1_present[0], layer_1_key)
    torch.testing.assert_close(layer_1_present[1], layer_1_value)


def test_decode_appends_without_reallocating_storage() -> None:
    cache = make_cache()
    prefill_key, prefill_value = make_kv(num_tokens=3, value=1)
    for layer_index in range(2):
        cache.update(layer_index, prefill_key, prefill_value)
    cache.advance(3)
    key_pointer = cache.k_cache.data_ptr()
    value_pointer = cache.v_cache.data_ptr()

    decode_key, decode_value = make_kv(num_tokens=1, value=5)
    present_key, present_value = cache.update(0, decode_key, decode_value)
    cache.advance(1)

    expected_key = torch.cat([prefill_key, decode_key], dim=2)
    expected_value = torch.cat([prefill_value, decode_value], dim=2)
    torch.testing.assert_close(present_key, expected_key)
    torch.testing.assert_close(present_value, expected_value)
    assert cache.cur_len == 4
    assert cache.k_cache.data_ptr() == key_pointer
    assert cache.v_cache.data_ptr() == value_pointer


def test_update_and_advance_reject_capacity_overflow() -> None:
    cache = make_cache(max_seq_len=3)
    key, value = make_kv(num_tokens=4, value=1)

    with pytest.raises(ValueError):
        cache.update(0, key, value)

    with pytest.raises(ValueError):
        cache.advance(4)


def test_reset_reuses_storage() -> None:
    cache = make_cache()
    key, value = make_kv(num_tokens=3, value=1)
    cache.update(0, key, value)
    cache.advance(3)
    key_pointer = cache.k_cache.data_ptr()
    value_pointer = cache.v_cache.data_ptr()

    cache.reset()

    assert cache.cur_len == 0
    assert cache.k_cache.data_ptr() == key_pointer
    assert cache.v_cache.data_ptr() == value_pointer

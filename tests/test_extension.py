import pytest
import torch

from minidecode import _C


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


def test_add_one_matches_torch() -> None:
    input_tensor = torch.randn(1025, device="cuda", dtype=torch.float32)

    actual = _C.add_one(input_tensor)

    torch.testing.assert_close(actual, input_tensor + 1.0)


def test_add_one_handles_empty_tensor() -> None:
    input_tensor = torch.empty(0, device="cuda", dtype=torch.float32)

    actual = _C.add_one(input_tensor)

    assert actual.shape == input_tensor.shape
    assert actual.dtype == input_tensor.dtype
    assert actual.device == input_tensor.device
    assert actual.numel() == 0


def test_add_one_rejects_non_contiguous_tensor() -> None:
    input_tensor = torch.randn(3, 4, device="cuda", dtype=torch.float32).T
    assert not input_tensor.is_contiguous()

    with pytest.raises(RuntimeError, match="input must be contiguous"):
        _C.add_one(input_tensor)


def test_add_one_rejects_cpu_tensor() -> None:
    input_tensor = torch.randn(8, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="input must be a CUDA tensor"):
        _C.add_one(input_tensor)


def test_add_one_rejects_unsupported_dtype() -> None:
    input_tensor = torch.randn(8, device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="input must have dtype float32"):
        _C.add_one(input_tensor)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16]
)
def test_write_kv_cache_matches_reference(dtype: torch.dtype) -> None:
    num_blocks = 4
    num_kv_heads = 2
    block_size = 4
    head_dim = 8
    num_tokens = 3
    key = torch.randn(
        1,
        num_kv_heads,
        num_tokens,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    value = torch.randn_like(key)
    key_cache = torch.zeros(
        num_blocks,
        num_kv_heads,
        block_size,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    value_cache = torch.zeros_like(key_cache)
    expected_key_cache = key_cache.clone()
    expected_value_cache = value_cache.clone()
    slots = [7, 8, 13]
    slot_mapping = torch.tensor(slots, device="cuda", dtype=torch.int64)

    for token, slot in enumerate(slots):
        physical_block = slot // block_size
        block_offset = slot % block_size
        expected_key_cache[physical_block, :, block_offset, :].copy_(
            key[0, :, token, :]
        )
        expected_value_cache[physical_block, :, block_offset, :].copy_(
            value[0, :, token, :]
        )

    _C.write_kv_cache(key, value, key_cache, value_cache, slot_mapping)

    torch.testing.assert_close(key_cache, expected_key_cache)
    torch.testing.assert_close(value_cache, expected_value_cache)

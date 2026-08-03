import pytest
import torch

from minidecode.paged_kv_cache import PagedKVCache


def make_cache() -> PagedKVCache:
    return PagedKVCache(
        num_layers=2,
        num_blocks=4,
        num_kv_heads=3,
        block_size=4,
        head_dim=5,
        dtype=torch.bfloat16,
        device="cpu",
    )


def make_kv(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (1, 3, num_tokens, 5)
    K = torch.arange(torch.tensor(shape).prod()).reshape(shape).to(torch.bfloat16)
    V = K + 1000
    return K, V


def test_paged_kv_cache_allocates_expected_storage() -> None:
    cache = make_cache()

    assert cache.K.shape == (2, 4, 3, 4, 5)
    assert cache.V.shape == cache.K.shape
    assert cache.K.dtype == torch.bfloat16
    assert cache.V.dtype == torch.bfloat16
    assert cache.K.device.type == "cpu"


def test_write_maps_tokens_within_and_across_blocks() -> None:
    cache = make_cache()
    K, V = make_kv(num_tokens=3)
    key_pointer = cache.K.data_ptr()
    value_pointer = cache.V.data_ptr()

    cache.write(layer_idx=0, K=K, V=V, slot_mapping=[7, 8, 13])

    torch.testing.assert_close(cache.K[0, 1, :, 3, :], K[0, :, 0, :])
    torch.testing.assert_close(cache.V[0, 1, :, 3, :], V[0, :, 0, :])
    torch.testing.assert_close(cache.K[0, 2, :, 0, :], K[0, :, 1, :])
    torch.testing.assert_close(cache.V[0, 2, :, 0, :], V[0, :, 1, :])
    torch.testing.assert_close(cache.K[0, 3, :, 1, :], K[0, :, 2, :])
    torch.testing.assert_close(cache.V[0, 3, :, 1, :], V[0, :, 2, :])
    assert cache.K.data_ptr() == key_pointer
    assert cache.V.data_ptr() == value_pointer


def test_write_keeps_layers_independent() -> None:
    cache = make_cache()
    cache.K.fill_(-1)
    cache.V.fill_(-1)
    K, V = make_kv(num_tokens=2)

    cache.write(layer_idx=1, K=K, V=V, slot_mapping=[0, 1])

    torch.testing.assert_close(cache.K[0], torch.full_like(cache.K[0], -1))
    torch.testing.assert_close(cache.V[0], torch.full_like(cache.V[0], -1))
    torch.testing.assert_close(cache.K[1, 0, :, 0, :], K[0, :, 0, :])
    torch.testing.assert_close(cache.K[1, 0, :, 1, :], K[0, :, 1, :])


@pytest.mark.parametrize("block_size", [0, -1])
def test_paged_kv_cache_rejects_non_positive_block_size(block_size: int) -> None:
    with pytest.raises(ValueError, match="block_size must be positive"):
        PagedKVCache(2, 4, 3, block_size, 5, torch.bfloat16, "cpu")


@pytest.mark.parametrize("layer_idx", [-1, 2])
def test_write_rejects_invalid_layer(layer_idx: int) -> None:
    cache = make_cache()
    K, V = make_kv(num_tokens=1)

    with pytest.raises(IndexError, match="invalid layer index"):
        cache.write(layer_idx, K, V, [0])


@pytest.mark.parametrize("slot_mapping", [[-1], [16]])
def test_write_rejects_invalid_slot_before_writing(slot_mapping: list[int]) -> None:
    cache = make_cache()
    cache.K.fill_(-1)
    cache.V.fill_(-1)
    K, V = make_kv(num_tokens=1)

    with pytest.raises(IndexError, match="invalid physical slot"):
        cache.write(0, K, V, slot_mapping)

    torch.testing.assert_close(cache.K, torch.full_like(cache.K, -1))
    torch.testing.assert_close(cache.V, torch.full_like(cache.V, -1))


def test_write_rejects_slot_count_mismatch() -> None:
    cache = make_cache()
    K, V = make_kv(num_tokens=2)

    with pytest.raises(ValueError, match="slot_mapping length"):
        cache.write(0, K, V, [0])


def test_write_rejects_incompatible_kv_shape() -> None:
    cache = make_cache()
    K, V = make_kv(num_tokens=2)

    with pytest.raises(ValueError, match="same shape"):
        cache.write(0, K, V[:, :, :1], [0, 1])

    with pytest.raises(ValueError, match="batch size 1"):
        cache.write(0, K.expand(2, -1, -1, -1), V.expand(2, -1, -1, -1), [0, 1])

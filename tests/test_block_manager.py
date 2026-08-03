import pytest
import torch  # noqa: F401 - load PyTorch shared libraries before minidecode._C

from minidecode import _C


def test_block_manager_allocates_all_blocks() -> None:
    manager = _C.BlockManager(4)

    allocated = [manager.allocate() for _ in range(4)]

    assert len(set(allocated)) == 4
    assert set(allocated) == {0, 1, 2, 3}
    assert manager.num_total_blocks() == 4
    assert manager.num_free_blocks() == 0
    assert all(manager.is_allocated(block_id) for block_id in allocated)


def test_block_manager_rejects_exhaustion() -> None:
    manager = _C.BlockManager(1)
    manager.allocate()

    with pytest.raises(RuntimeError, match="no block left"):
        manager.allocate()

    assert manager.num_free_blocks() == 0


def test_block_manager_frees_and_reuses_block() -> None:
    manager = _C.BlockManager(3)
    block_id = manager.allocate()
    manager.free(block_id)

    assert not manager.is_allocated(block_id)
    assert manager.num_free_blocks() == 3

    reused_id = manager.allocate()

    assert reused_id == block_id
    assert manager.is_allocated(reused_id)
    assert manager.num_free_blocks() == 2


@pytest.mark.parametrize("num_blocks", [0, -1])
def test_block_manager_rejects_non_positive_capacity(num_blocks: int) -> None:
    with pytest.raises(ValueError, match="num_blocks must be positive"):
        _C.BlockManager(num_blocks)


@pytest.mark.parametrize("block_id", [-1, 3])
def test_block_manager_rejects_invalid_free(block_id: int) -> None:
    manager = _C.BlockManager(3)

    with pytest.raises(IndexError, match="invalid block id"):
        manager.free(block_id)

    assert manager.num_free_blocks() == 3


@pytest.mark.parametrize("block_id", [-1, 3])
def test_block_manager_rejects_invalid_query(block_id: int) -> None:
    manager = _C.BlockManager(3)

    with pytest.raises(IndexError, match="invalid block id"):
        manager.is_allocated(block_id)


def test_block_manager_rejects_double_free() -> None:
    manager = _C.BlockManager(2)
    block_id = manager.allocate()
    manager.free(block_id)

    with pytest.raises(RuntimeError, match="internal error"):
        manager.free(block_id)

    assert manager.num_free_blocks() == 2

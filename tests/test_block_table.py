import pytest
import torch  # noqa: F401 - load PyTorch shared libraries before minidecode._C

from minidecode import _C
from minidecode.block_table import SequenceBlockTable


def test_append_tokens_uses_remaining_slots_before_allocating() -> None:
    manager = _C.BlockManager(4)
    table = SequenceBlockTable(manager, block_size=4)

    first_slots = table.append_tokens(3)
    first_block = table.block_ids[0]

    assert first_slots == [first_block * 4 + offset for offset in range(3)]
    assert table.num_tokens == 3
    assert len(table.block_ids) == 1
    assert manager.num_free_blocks() == 3

    final_slot = table.append_tokens(1)

    assert final_slot == [first_block * 4 + 3]
    assert table.num_tokens == 4
    assert len(table.block_ids) == 1
    assert manager.num_free_blocks() == 3


def test_append_tokens_crosses_physical_block_boundary() -> None:
    manager = _C.BlockManager(4)
    table = SequenceBlockTable(manager, block_size=4)
    table.append_tokens(3)
    first_block = table.block_ids[0]

    slots = table.append_tokens(3)
    second_block = table.block_ids[1]

    assert slots == [
        first_block * 4 + 3,
        second_block * 4,
        second_block * 4 + 1,
    ]
    assert table.num_tokens == 6
    assert len(table.block_ids) == 2
    assert first_block != second_block
    assert manager.num_free_blocks() == 2


def test_append_zero_tokens_is_no_op() -> None:
    manager = _C.BlockManager(2)
    table = SequenceBlockTable(manager, block_size=4)

    slots = table.append_tokens(0)

    assert slots == []
    assert table.block_ids == []
    assert table.num_tokens == 0
    assert manager.num_free_blocks() == 2


@pytest.mark.parametrize("block_size", [0, -1])
def test_block_table_rejects_non_positive_block_size(block_size: int) -> None:
    manager = _C.BlockManager(2)

    with pytest.raises(ValueError, match="block_size must be positive"):
        SequenceBlockTable(manager, block_size)


def test_append_tokens_rejects_negative_count() -> None:
    manager = _C.BlockManager(2)
    table = SequenceBlockTable(manager, block_size=4)

    with pytest.raises(ValueError, match="num_tokens must be non-negative"):
        table.append_tokens(-1)

    assert table.block_ids == []
    assert table.num_tokens == 0
    assert manager.num_free_blocks() == 2


def test_release_returns_all_blocks_and_is_idempotent() -> None:
    manager = _C.BlockManager(3)
    table = SequenceBlockTable(manager, block_size=4)
    table.append_tokens(9)
    allocated_blocks = table.block_ids.copy()

    table.release()

    assert table.block_ids == []
    assert table.num_tokens == 0
    assert manager.num_free_blocks() == 3
    assert all(not manager.is_allocated(block_id) for block_id in allocated_blocks)

    table.release()
    assert manager.num_free_blocks() == 3


def test_allocation_failure_rolls_back_new_blocks() -> None:
    manager = _C.BlockManager(2)
    table = SequenceBlockTable(manager, block_size=4)
    table.append_tokens(4)
    original_blocks = table.block_ids.copy()
    original_free_count = manager.num_free_blocks()

    with pytest.raises(RuntimeError, match="no block left"):
        table.append_tokens(5)

    assert table.block_ids == original_blocks
    assert table.num_tokens == 4
    assert manager.num_free_blocks() == original_free_count
    assert manager.is_allocated(original_blocks[0])


def test_multiple_requests_allocate_and_release_independently() -> None:
    manager = _C.BlockManager(4)
    first = SequenceBlockTable(manager, block_size=2)
    second = SequenceBlockTable(manager, block_size=2)

    first.append_tokens(3)
    second.append_tokens(2)
    first_blocks = set(first.block_ids)
    second_blocks = set(second.block_ids)

    assert first_blocks.isdisjoint(second_blocks)
    assert manager.num_free_blocks() == 1

    first.release()

    assert manager.num_free_blocks() == 3
    assert second.num_tokens == 2
    assert set(second.block_ids) == second_blocks
    assert all(manager.is_allocated(block_id) for block_id in second_blocks)

    replacement = SequenceBlockTable(manager, block_size=2)
    replacement.append_tokens(3)

    assert set(replacement.block_ids).issubset(first_blocks)
    assert set(replacement.block_ids).isdisjoint(second_blocks)

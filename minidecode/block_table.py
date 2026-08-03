import torch

from . import _C


class SequenceBlockTable:
    def __init__(self, manager: _C.BlockManager, block_size):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.manager = manager
        self.block_ids = []
        self.num_tokens = 0
        self.block_size = block_size

    def append_tokens(self, num_tokens: int) -> list[int]:
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")

        if num_tokens == 0:
            return []
        total = self.num_tokens + num_tokens
        required_blocks = (total + self.block_size - 1) // self.block_size
        blocks_to_allocate = required_blocks - len(self.block_ids)
        new_block_ids = []
        try:
            for _ in range(blocks_to_allocate):
                block_id = self.manager.allocate()
                new_block_ids.append(block_id)
        except Exception:
            for block_id in new_block_ids:
                self.manager.free(block_id)
            raise
        self.block_ids.extend(new_block_ids)
        slots = []
        for pos in range(self.num_tokens, total):
            logical = pos // self.block_size
            offset = pos % self.block_size
            physical = self.block_ids[logical]
            slots.append(physical * self.block_size + offset)
        self.num_tokens = total
        return slots

    def release(self):
        for block_id in self.block_ids:
            self.manager.free(block_id)
        self.block_ids = []
        self.num_tokens = 0

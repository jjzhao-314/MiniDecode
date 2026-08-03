import torch

from . import _C


class PagedKVCache:
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        num_kv_heads: int,
        block_size: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.K = torch.empty(
            (num_layers, num_blocks, num_kv_heads, block_size, head_dim),
            dtype=dtype,
            device=device,
        )
        self.V = torch.empty(
            (num_layers, num_blocks, num_kv_heads, block_size, head_dim),
            dtype=dtype,
            device=device,
        )
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def write(
        self,
        layer_idx: int,
        K: torch.Tensor,  # 1, Hkv, S, D
        V: torch.Tensor,  # 1, Hkv, S, D
        slot_mapping: list[int] | torch.Tensor,  # S
    ) -> None:
        if layer_idx < 0 or layer_idx >= self.K.shape[0]:
            raise IndexError("invalid layer index")
        if K.ndim != 4 or V.ndim != 4:
            raise ValueError("K and V must have shape [1, Hkv, S, D]")
        if K.shape != V.shape:
            raise ValueError("K and V must have the same shape")
        if K.shape[0] != 1:
            raise ValueError("PagedKVCache currently requires batch size 1")

        S = K.shape[2]
        if len(slot_mapping) != S:
            raise ValueError("slot_mapping length must match the token count")

        if K.is_cuda:
            if not isinstance(slot_mapping, torch.Tensor):
                slot_mapping = torch.tensor(
                    slot_mapping,
                    dtype=torch.int64,
                    device=K.device,
                )
            _C.write_kv_cache(
                K,
                V,
                self.K[layer_idx],
                self.V[layer_idx],
                slot_mapping,
            )
            return

        if isinstance(slot_mapping, torch.Tensor):
            slots = slot_mapping.tolist()
        else:
            slots = slot_mapping

        num_slots = self.K.shape[1] * self.block_size
        for i, slot in enumerate(slots):
            if slot < 0 or slot >= num_slots:
                raise IndexError("invalid physical slot")
            physical = slot // self.block_size
            block_offset = slot % self.block_size
            self.K[layer_idx, physical, :, block_offset, :].copy_(K[0, :, i, :])
            self.V[layer_idx, physical, :, block_offset, :].copy_(V[0, :, i, :])

    def read(
        self,
        layer_idx: int,
        block_ids: list[int],
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < 0 or layer_idx >= self.K.shape[0]:
            raise IndexError("invalid layer index")
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")

        required_blocks = (num_tokens + self.block_size - 1) // self.block_size
        if len(block_ids) != required_blocks:
            raise ValueError("block_ids does not match num_tokens")
        for block_id in block_ids:
            if block_id < 0 or block_id >= self.K.shape[1]:
                raise IndexError("invalid physical block")

        sequence_capacity = len(block_ids) * self.block_size
        K = (
            self.K[layer_idx, block_ids]
            .transpose(0, 1)
            .contiguous()
            .reshape(self.num_kv_heads, sequence_capacity, self.head_dim)[
                :, :num_tokens, :
            ]
            .unsqueeze(0)
        )

        V = (
            self.V[layer_idx, block_ids]
            .transpose(0, 1)
            .contiguous()
            .reshape(self.num_kv_heads, sequence_capacity, self.head_dim)[
                :, :num_tokens, :
            ]
            .unsqueeze(0)
        )
        return K, V

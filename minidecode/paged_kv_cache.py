import torch


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

    def write(
        self,
        layer_idx: int,
        K: torch.Tensor,  # 1, Hkv, S, D
        V: torch.Tensor,  # 1, Hkv, S, D
        slot_mapping: list[int],  # S
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

        num_slots = self.K.shape[1] * self.block_size
        for i, slot in enumerate(slot_mapping):
            if slot < 0 or slot >= num_slots:
                raise IndexError("invalid physical slot")
            physical = slot // self.block_size
            block_offset = slot % self.block_size
            self.K[layer_idx, physical, :, block_offset, :].copy_(K[0, :, i, :])
            self.V[layer_idx, physical, :, block_offset, :].copy_(V[0, :, i, :])

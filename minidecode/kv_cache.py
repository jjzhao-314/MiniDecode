import torch


class ContiguousKVCache:
    def __init__(
        self, num_layers, batch_size, num_kv_heads, max_seq_len, head_dim, dtype, device
    ):
        self.k_cache = torch.empty(
            (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.empty(
            (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device,
        )
        self.cur_len = 0
        self.cap = max_seq_len

    def update(
        self,
        idx,
        K: torch.Tensor,  # B, Hkv, S, D
        V: torch.Tensor,  # B, Hkv, S, D
    ):
        S = K.shape[2]
        end = self.cur_len + S
        if end > self.cap:
            raise ValueError("KV Cache capacity exceeded")
        self.k_cache[idx, :, :, self.cur_len : end, :] = K
        self.v_cache[idx, :, :, self.cur_len : end, :] = V
        return self.k_cache[idx, :, :, :end, :], self.v_cache[idx, :, :, :end, :]

    def advance(self, num_tokens):
        end = self.cur_len + num_tokens
        if end > self.cap:
            raise ValueError("KV Cache capacity exceeded")
        self.cur_len = end

    def reset(self):
        self.cur_len = 0

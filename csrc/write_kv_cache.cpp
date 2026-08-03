#include "kernels/write_kv_cache.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace minidecode {

void write_kv_cache(torch::Tensor key, torch::Tensor value,
                    torch::Tensor key_cache, torch::Tensor value_cache,
                    torch::Tensor slot_mapping) {
    TORCH_CHECK(key.is_cuda() && value.is_cuda() && key_cache.is_cuda() &&
                    value_cache.is_cuda() && slot_mapping.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(key.is_contiguous() && value.is_contiguous() &&
                    key_cache.is_contiguous() && value_cache.is_contiguous() &&
                    slot_mapping.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(key.dim() == 4 && value.dim() == 4,
                "key and value must have shape [1, Hkv, S, D]");
    TORCH_CHECK(key.sizes() == value.sizes(),
                "key and value must have the same shape");
    TORCH_CHECK(key.size(0) == 1,
                "write_kv_cache currently requires batch size 1");
    TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
                "key_cache and value_cache must have shape [P, Hkv, B, D]");
    TORCH_CHECK(key_cache.sizes() == value_cache.sizes(),
                "key_cache and value_cache must have the same shape");
    TORCH_CHECK(
        key.size(1) == key_cache.size(1) && key.size(3) == key_cache.size(3),
        "cache head dimensions must match key and value");
    TORCH_CHECK(slot_mapping.dim() == 1 && slot_mapping.size(0) == key.size(2),
                "slot_mapping must have shape [S]");
    TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt64,
                "slot_mapping must have dtype int64");
    TORCH_CHECK(key.scalar_type() == value.scalar_type() &&
                    key.scalar_type() == key_cache.scalar_type() &&
                    key.scalar_type() == value_cache.scalar_type(),
                "key, value, and caches must have the same dtype");
    TORCH_CHECK(key.device() == value.device() &&
                    key.device() == key_cache.device() &&
                    key.device() == value_cache.device() &&
                    key.device() == slot_mapping.device(),
                "all tensors must be on the same device");

    if (key.size(2) == 0) {
        return;
    }

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_write_kv_cache_cuda(
        key.const_data_ptr(), value.const_data_ptr(),
        key_cache.mutable_data_ptr(), value_cache.mutable_data_ptr(),
        static_cast<const int64_t*>(slot_mapping.const_data_ptr()),
        key.scalar_type(), key.numel(), key.size(2), key.size(1), key.size(3),
        key_cache.size(2), key_cache.size(0), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace minidecode

#pragma once

#include <c10/core/ScalarType.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace minidecode {

void launch_write_kv_cache_cuda(
    const void* key, const void* value, void* key_cache, void* value_cache,
    const int64_t* slot_mapping, c10::ScalarType scalar_type,
    int64_t num_elements, int64_t num_tokens, int64_t num_kv_heads,
    int64_t head_dim, int64_t block_size, int64_t num_blocks,
    cudaStream_t stream);

}  // namespace minidecode

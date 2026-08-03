#include <ATen/Dispatch.h>
#include <c10/util/Exception.h>

#include <cstdint>

#include "write_kv_cache.h"

namespace minidecode {
namespace {

template <typename scalar_t>
__global__ void write_kv_cache_kernel(
    const scalar_t* key, const scalar_t* value, scalar_t* key_cache,
    scalar_t* value_cache, const int64_t* slot_mapping, int64_t num_elements,
    int64_t num_tokens, int64_t num_kv_heads, int64_t head_dim,
    int64_t block_size, int64_t num_blocks) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return;
    int64_t d = tid % head_dim;
    int64_t temp = tid / head_dim;

    int64_t token = temp % num_tokens;
    int64_t head = temp / num_tokens;

    int64_t slot = slot_mapping[token];
    if (slot < 0 || slot >= num_blocks * block_size) {
        return;
    }
    int64_t physical_block = slot / block_size;
    int64_t block_offset = slot % block_size;
    int64_t destination_index =
        ((physical_block * num_kv_heads + head) * block_size + block_offset) *
            head_dim +
        d;

    key_cache[destination_index] = key[tid];
    value_cache[destination_index] = value[tid];
}

}  // namespace

void launch_write_kv_cache_cuda(const void* key, const void* value,
                                void* key_cache, void* value_cache,
                                const int64_t* slot_mapping,
                                c10::ScalarType scalar_type,
                                int64_t num_elements, int64_t num_tokens,
                                int64_t num_kv_heads, int64_t head_dim,
                                int64_t block_size, int64_t num_blocks,
                                cudaStream_t stream) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((num_elements + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::ScalarType::Half, c10::ScalarType::BFloat16, scalar_type,
        "write_kv_cache_cuda", [&] {
            write_kv_cache_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                static_cast<const scalar_t*>(key),
                static_cast<const scalar_t*>(value),
                static_cast<scalar_t*>(key_cache),
                static_cast<scalar_t*>(value_cache), slot_mapping, num_elements,
                num_tokens, num_kv_heads, head_dim, block_size, num_blocks);
        });
}

}  // namespace minidecode

#include <cuda_runtime.h>

#include <cstdint>

namespace minidecode {
namespace {

__global__ void add_one_kernel(const float* input, float* output,
                               int64_t num_elements) {
    const int64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_elements) {
        output[tid] = input[tid] + 1.0F;
    }
}

}  // namespace

void launch_add_one_cuda(const float* input, float* output,
                         int64_t num_elements, cudaStream_t stream) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((num_elements + threads - 1) / threads);

    {
        add_one_kernel<<<blocks, threads, 0, stream>>>(input, output,
                                                       num_elements);
    }
}

}  // namespace minidecode

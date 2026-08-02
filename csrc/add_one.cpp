#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>

namespace minidecode {

void launch_add_one_cuda(const float* input, float* output,
                         int64_t num_elements, cudaStream_t stream);

torch::Tensor add_one(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "input must have dtype float32");

    torch::Tensor output = torch::empty_like(input);
    const int64_t num_elements = input.numel();

    if (num_elements == 0) {
        return output;
    }

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_add_one_cuda(input.data_ptr<float>(), output.data_ptr<float>(),
                        num_elements, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

}  // namespace minidecode

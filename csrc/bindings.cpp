#include <torch/extension.h>

namespace minidecode {

torch::Tensor add_one(torch::Tensor input);

}  // namespace minidecode

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("add_one", &minidecode::add_one,
               "Add one to each tensor element (CUDA)");
}

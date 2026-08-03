#include <torch/extension.h>

#include "runtime/block_manager.h"

namespace minidecode {

torch::Tensor add_one(torch::Tensor input);

}  // namespace minidecode

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("add_one", &minidecode::add_one,
               "Add one to each tensor element (CUDA)");

    pybind11::class_<minidecode::BlockManager>(module, "BlockManager")
        .def(pybind11::init<int>(), pybind11::arg("num_blocks"))
        .def("allocate", &minidecode::BlockManager::allocate)
        .def("free", &minidecode::BlockManager::free,
             pybind11::arg("block_id"))
        .def("num_free_blocks", &minidecode::BlockManager::num_free_blocks)
        .def("num_total_blocks", &minidecode::BlockManager::num_total_blocks)
        .def("is_allocated", &minidecode::BlockManager::is_allocated,
             pybind11::arg("block_id"));
}

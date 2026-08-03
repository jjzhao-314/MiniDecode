from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    ext_modules=[
        CUDAExtension(
            name="minidecode._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/add_one.cpp",
                "csrc/write_kv_cache.cpp",
                "csrc/runtime/block_manager.cpp",
                "csrc/kernels/add_one.cu",
                "csrc/kernels/write_kv_cache.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

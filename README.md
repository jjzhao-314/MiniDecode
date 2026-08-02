# MiniDecode

MiniDecode 是一个从零实现的教学型单 GPU LLM 推理引擎。项目首先服务于对
Transformer 推理、CUDA kernel 和 inference runtime 的深入理解，同时形成一个可运行、
可测试、可分析的完整项目。

## 项目目标

第一版固定支持：

- 模型：Qwen3-0.6B；
- 精度：BF16；
- 硬件：单张 NVIDIA GeForce RTX 5070；
- 模型组织、调度、测试和 Benchmark 使用 Python；
- Block Manager 和资源管理使用 C++；
- RoPE、KV Cache Write 和 Paged Attention 等核心算子使用 CUDA；
- GEMM 使用 PyTorch/cuBLAS，不重复实现通用矩阵乘法。

项目遵循“先正确、再优化”的原则：每个模块都需要先与 PyTorch 或 Hugging Face
reference 对齐，再进行性能测试和 CUDA 优化。

## 技术路线

1. 实现不依赖 Hugging Face 完整 `forward()` 的 Qwen3 PyTorch reference model；
2. 分离 Prefill 和 Decode，并实现 Contiguous KV Cache；
3. 使用 C++ Block Manager 管理 Paged KV Cache；
4. 实现 KV Cache Write、RoPE 和 Paged Attention CUDA kernel；
5. 实现最小可用的 Continuous Batching；
6. 建立正确性测试、kernel benchmark 和端到端 benchmark；
7. 使用 NCU/NSYS 分析性能，并记录 TTFT、TPOT、吞吐量和显存占用。

暂不考虑多 GPU、量化、完整服务接口、训练和反向传播、Tensor Parallel，以及完整
FlashAttention。

## 计划结构

```text
MiniDecode/
├── minidecode/       # 模型、KV Cache、调度和采样
├── csrc/
│   ├── runtime/      # C++ Block Manager
│   └── kernels/      # CUDA kernels
├── tests/            # 正确性与回归测试
├── benchmarks/       # kernel 与端到端性能测试
└── README.md
```

## 开发环境

当前开发环境：

- GPU：NVIDIA GeForce RTX 5070，Compute Capability 12.0（`sm_120`）；
- Python：3.14.6；
- PyTorch：2.13.0+cu132；
- CUDA Toolkit：13.3；
- Ninja：1.13.2。

环境和依赖会随着项目搭建进一步固化，最终提供可复现的构建、测试和 Benchmark
命令。

---
layout: default
title: AI Infra
description: AI 基础设施方向：推理优化、训练系统、C++ 底层与部署工程
eyebrow: Module 09
---

# AI Infra

面向 AI 基础设施岗位（推理优化、训练框架、部署工程）的系统知识，覆盖 KV Cache、量化、编译优化、显存计算、C++ 系统编程等高频面试考点。适合有深度学习基础、准备转向或兼顾 Infra 方向的同学。

## 这部分会解决什么

- 推理加速全链路：KV Cache、量化、TVM、MoE、Attention 变体
- 推理服务：FlashAttention、PagedAttention、Chunk Prefill、Continuous Batching、vLLM/SGLang
- 分布式：TP/PP/EP/SP 并行、ZeRO、通信算子与优化
- 训练系统：显存拆解、混合精度、并行策略
- C++ 底层：内存管理、智能指针、OOP、移动语义
- CUDA / GPU：硬件架构、内存模型、GEMM/Reduce 优化、Triton、编译流程
- 模型架构：Transformer 结构、BN/LN/RMSNorm、RoPE、参数量计算、DeepSeek V3

## 建议阅读顺序

1. [模型架构基础](./09-model-architecture/index.html)
2. [推理加速全景：从 KV Cache 到部署](./01-inference-acceleration/index.html)
3. [量化方法与精度保障](./02-quantization/index.html)
4. [训练显存计算与优化](./03-training-memory/index.html)
5. [C++ 系统编程高频题](./04-cpp-systems/index.html)
6. [PyTorch 性能优化机制](./05-pytorch-internals/index.html)
7. [CUDA 编程与 GPU 架构](./06-cuda-gpu-programming/index.html)
8. [推理服务框架与调度](./07-inference-serving/index.html)
9. [分布式训练与推理](./08-distributed-parallel/index.html)

## 当前文章

| 文章 | 作用 |
| --- | --- |
| [模型架构基础](./09-model-architecture/index.html) | Transformer 结构、BN/LN/RMSNorm、RoPE、参数计算、DeepSeek V3、Decoder-only |
| [推理加速全景：从 KV Cache 到部署](./01-inference-acceleration/index.html) | KV Cache、推理加速、TVM、MoE、Attention 变体、MLA 权重吸收、PD 分离 |
| [量化方法与精度保障](./02-quantization/index.html) | INT8/INT4/FP8 量化、SmoothQuant/AWQ/GPTQ、剪枝稀疏化、精度保障 |
| [训练显存计算与优化](./03-training-memory/index.html) | 参数/梯度/优化器/激活值显存拆解、混合精度、重计算 |
| [C++ 系统编程高频题](./04-cpp-systems/index.html) | 内存管理、智能指针、OOP/虚函数、STL、移动语义、进程线程、GIL、虚拟内存 |
| [PyTorch 性能优化机制](./05-pytorch-internals/index.html) | Python→C++ 调度、torch.compile、算子注册、tensor API、算子融合 |
| [CUDA 编程与 GPU 架构](./06-cuda-gpu-programming/index.html) | GPU/SM 架构、GEMM/Reduce、Bank Conflict、Warp Shuffle、Online Softmax、Stream、DMA/RDMA |
| [推理服务框架与调度](./07-inference-serving/index.html) | FlashAttention、PagedAttention、Chunk Prefill、Continuous Batching、Speculative Decoding、AF分离 |
| [分布式训练与推理](./08-distributed-parallel/index.html) | TP/PP/EP/SP、ZeRO、DualPipe、RL训练优化、容错、通信优化 |

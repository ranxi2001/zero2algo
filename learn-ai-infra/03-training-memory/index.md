---
layout: default
title: 训练显存计算与优化
description: 参数/梯度/优化器状态/激活值显存拆解、混合精度与重计算
parent: AI Infra
---

# 训练显存计算与优化

训练大模型时，显存是最关键的约束之一。本文拆解训练显存的四大组成部分，并介绍常见的优化手段。

---

## Q：训练显存计算

> 来源：AI Infra / 抖音搜推架构一面（牛客网）

**普通回答**：训练显存包括模型参数、梯度和优化器状态。

**更好的回答**：

训练显存由四部分组成，以 Adam + FP16 混合精度训练为例：

**1. 模型参数（Model Parameters）**
- FP16 参数：2 bytes × P（P = 参数量）
- FP32 master copy（混合精度需要）：4 bytes × P
- 合计：6P bytes

**2. 梯度（Gradients）**
- FP16 梯度：2 bytes × P
- 合计：2P bytes

**3. 优化器状态（Optimizer States）**
- Adam 需要一阶矩（momentum）和二阶矩（variance），各 FP32
- 4 bytes × P × 2 = 8P bytes
- SGD 只需 momentum：4P bytes

**4. 激活值（Activations）**— 见下题详解

**总结（不含激活值）**：

| 组件 | FP32 训练 | 混合精度（Adam） |
|------|-----------|-----------------|
| 参数 | 4P | 2P + 4P = 6P |
| 梯度 | 4P | 2P |
| 优化器 | 8P | 8P |
| **合计** | **16P** | **16P** |

举例：7B 模型（P = 7×10⁹）
- 不含激活值：16 × 7B = 112 GB
- 这就是为什么 7B 模型训练至少需要 2 张 A100 80GB

**优化手段**：
- **ZeRO（DeepSpeed）**：将参数/梯度/优化器状态切分到多卡
  - ZeRO-1：切优化器状态 → 省 ~4×
  - ZeRO-2：切优化器 + 梯度
  - ZeRO-3：全切（参数也分片）
- **Offload**：将优化器状态 offload 到 CPU 内存
- **梯度累积**：减小单步 batch → 减少激活值显存

**考察点**：能否精确拆解每一项的来源和大小，以及对 ZeRO 等优化策略的理解。

---

## Q：激活值会占多少显存

> 来源：AI Infra / 抖音搜推架构一面（牛客网）

**普通回答**：激活值显存和 batch size、序列长度有关，通常比参数大。

**更好的回答**：

激活值是前向传播中每层的中间输出，需要保留到反向传播时计算梯度。

**Transformer 单层激活值显存**（FP16，不含 attention score）：

```
每层激活 ≈ seq_len × batch_size × hidden_dim × 约 10~14 × 2 bytes
```

其中 "10~14" 来自：
- LayerNorm 输入输出（2 份）
- QKV 投影输出（3 × hidden_dim）
- Attention score matrix（seq_len × seq_len × num_heads）← 这一项是平方级
- Attention output
- FFN 中间层（通常 4 × hidden_dim）
- FFN 输出

**总激活值显存** = 单层 × num_layers

举例：LLaMA-7B（32 层，hidden=4096，seq_len=2048，batch=1，FP16）
- Attention score：2048 × 2048 × 32 heads × 2 bytes × 32 layers ≈ 8 GB
- 其余激活：约 5-10 GB
- batch=8 → 激活值可超 100 GB

**关键特征**：
- 激活值随 batch_size 和 seq_len **线性甚至平方**增长
- 参数/梯度/优化器与 batch_size 无关，但激活值与 batch_size 成正比

**优化手段**：
- **Activation Checkpointing（梯度重计算）**：只保存部分层的激活，反向时重新计算中间层。以约 33% 额外计算换取大幅显存节省
- **FlashAttention**：不显式存储 seq_len × seq_len 的 attention matrix，O(n) 显存
- **Sequence Parallelism**：将序列维度切分到多卡

**考察点**：激活值是训练显存中最容易被忽视但占比最大的部分，面试官想看你是否理解这个"隐形大户"。

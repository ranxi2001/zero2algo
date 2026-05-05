---
layout: default
title: 模型架构基础
description: Transformer 结构、BN/LN/RMSNorm、位置编码/RoPE、参数量与计算量、Decoder-only 设计
parent: AI Infra
---

# 模型架构基础

AI Infra 面试不只考工程，也考对模型架构的理解。本文覆盖 Transformer 结构、归一化方法、位置编码和参数/计算量估算等高频基础题。

---

## Q：Transformer 相比 RNN/MLP 的优势

> 来源：美团北斗 / AI Infra · 拼多多 / AI Infra · 蔚来 / AI Infra

**普通回答**：Transformer 可以并行计算，RNN 不行。

**更好的回答**：

| 维度 | RNN | Transformer |
|------|-----|------------|
| 并行性 | 串行（t 依赖 t-1） | 并行（所有位置同时计算） |
| 长距离依赖 | 梯度消失/爆炸（LSTM 缓解但不根治） | 直接 attention 到任意距离 |
| 训练效率 | 无法并行，GPU 利用率低 | 矩阵乘为主，GPU 友好 |
| 推理 | 每步 O(1) 额外计算 | 每步 O(n) attention（需 KV Cache 优化） |

**Transformer 相比 MLP 的优势**：
- MLP 是 position-independent（第 i 个位置的输出只取决于第 i 个输入）
- Transformer 通过 Attention 实现 token 间交互 → 上下文理解能力
- MLP 处理变长序列需要 padding → 浪费；Transformer 原生支持

**Attention 解决的核心问题**：
- 让任意两个位置的信息可以在一步内交互（O(1) 跳数 vs RNN 的 O(n)）
- 可解释性：attention weight 显示模型"看哪里"

**Attention 的缺点**：
- 计算复杂度 O(n²)：长序列代价高
- 没有归纳偏置：需要更多数据才能学好（vs CNN 的局部性、RNN 的时序性）

**考察点**：从并行性、长距离依赖、训练效率三个维度对比。

---

## Q：Transformer 参数分布在哪里 / 参数量最大和计算量最大的分别是哪部分

> 来源：美团北斗 / AI Infra 校招 · 快手 / AI Infra 实习 · 阿里云 / AI Infra

**普通回答**：参数主要在线性层。

**更好的回答**：

**单层 Transformer 参数分布**（hidden_dim = d, FFN intermediate = 4d）：

| 模块 | 参数量 | 占比 |
|------|--------|------|
| QKV projection | 3 × d × d = 3d² | ~25% |
| Output projection | d × d = d² | ~8% |
| FFN (gate+up+down) | 3 × d × 4d = 12d² | ~67% |
| LayerNorm | 4d（两层各 2d） | ~0% |

**结论：FFN 参数量远大于 Attention（约 3:1）**

**计算量（FLOPs）分析**（seq_len = s）：

| 模块 | FLOPs | 说明 |
|------|-------|------|
| QKV projection | 3 × 2 × s × d² = 6sd² | 矩阵乘 |
| Attention score | 2 × s × s × d = 2s²d | Q×K^T |
| Attention output | 2 × s × s × d = 2s²d | Score×V |
| Output projection | 2sd² | 矩阵乘 |
| FFN | 3 × 2 × s × d × 4d = 24sd² | 三个线性层 |

- 当 s < 12d 时（如 s=4096, d=4096）：FFN 计算量 > Attention
- 当 s > 12d 时（超长序列）：Attention 的 O(s²) 占主导

**模型总参数量公式**：
```
P ≈ L × (12d² + 13d) + V × d
  ≈ 12Ld²  (忽略小项)
```
- L = 层数，d = hidden_dim，V = vocab_size
- LLaMA-7B：L=32, d=4096 → 12×32×4096² ≈ 6.4B ✓

**6Nd 公式（单次训练 FLOPs）**：
```
FLOPs_per_token ≈ 6N (N=参数量)
  = 2N (forward) + 4N (backward, 约2×forward)
总 FLOPs = 6 × N × D (D=总训练 token 数)
```

**考察点**：能快速估算参数量和计算量，知道 FFN 是参数/计算的主要贡献者。

---

## Q：BN / LN / RMSNorm 的区别

> 来源：百度 / AI Infra 一面 · 飞腾 / AI Infra 实习

**普通回答**：BN 在 batch 维度归一化，LN 在 hidden 维度归一化。

**更好的回答**：

**归一化维度对比**（输入 shape: [B, S, H]）：

| 方法 | 归一化维度 | 统计量计算范围 | 训练/推理一致性 |
|------|-----------|--------------|----------------|
| BN | H 维度 | 跨 batch 统计 (B×S 个样本) | 推理用 running mean → 有差异 |
| LN | H 维度 | 单样本内统计 (H 维) | 完全一致 |
| RMSNorm | H 维度 | 单样本 RMS | 完全一致 |

**公式**：
```
LN:      y = (x - mean) / sqrt(var + ε) × γ + β
RMSNorm: y = x / sqrt(mean(x²) + ε) × γ        (无 mean 偏移，无 β)
```

**为什么 LLM 用 RMSNorm 而非 LN**：
1. 省去 mean 计算 → 少一次 reduce → 快 ~10-15%
2. 省去 β 偏置 → 参数略少
3. 实践证明效果与 LN 相当

**为什么不用 BN**：
- BN 需要跨 batch 统计 → batch size 小时统计不稳定
- NLP 中 batch 维度和 seq 维度交互复杂
- 推理时用 running statistics → 与训练不一致

**RMSNorm 的 CUDA 优化**（飞腾追问）：
- 计算：一次 reduce（求 mean(x²)），一次 element-wise（归一化）
- memory-bound（reduce 是瓶颈）
- 优化手段：
  - 向量化加载（float4）
  - Warp-level reduction（shfl_down）减少 shared memory 使用
  - 与前一个算子融合（residual add + RMSNorm = 一个 kernel）
  - Double buffer：计算当前块的同时预取下一块

**考察点**：区分三者的归一化维度，解释 RMSNorm 为什么成为 LLM 标配。

---

## Q：位置编码 / RoPE 原理

> 来源：百度 / AI Infra 一面 · 飞腾 / AI Infra 实习

**普通回答**：RoPE 用旋转矩阵编码位置信息。

**更好的回答**：

**位置编码的必要性**：
- Transformer 的 self-attention 是 permutation-equivariant（交换 token 顺序，输出也交换）
- 没有位置编码 → 模型不知道 token 的先后顺序

**三代位置编码**：

| 代 | 方法 | 特点 |
|----|------|------|
| 1 | Sinusoidal (原版) | 固定函数，加到 embedding 上 |
| 2 | Learned positional embedding | 可学习向量，加到 embedding |
| 3 | RoPE (Rotary) | 旋转 Q,K 向量，不加到 embedding |

**RoPE 核心思想**：

将位置信息编码为 Q 和 K 向量的旋转角度：
```
Q_pos = R(θ_pos) × Q    # R 是旋转矩阵
K_pos = R(θ_pos) × K

Attention score = Q_pos · K_pos = Q × R(θ_m - θ_n) × K
```
- score 只依赖相对位置 (m-n)，天然支持相对位置编码
- 旋转不改变向量长度 → 不影响 attention 的分布

**具体实现**（2D 旋转，扩展到 d 维）：
```python
# 将 d 维向量两两配对，每对做 2D 旋转
# 第 k 对的旋转角度 = pos × θ_k, θ_k = 10000^(-2k/d)
def apply_rope(x, pos):
    # x: (..., d), 将 d 维分为 d/2 对
    cos = cos(pos * theta)  # (d/2,)
    sin = sin(pos * theta)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return concat(x1*cos - x2*sin, x1*sin + x2*cos)
```

**RoPE 的优势**：
- 天然相对位置编码（score 只看 m-n）
- 可外推到更长序列（配合 NTK-aware scaling / YaRN）
- 不增加参数量（纯计算）
- 计算高效（逐元素乘+加）

**RoPE 嵌入时机**：
- 在 Q 和 K 投影之后、attention score 计算之前施加
- V 不做 RoPE（V 承载内容信息，不需要位置）

**考察点**：理解 RoPE 通过旋转实现相对位置编码的数学原理，以及它在 MLA 中的特殊处理。

---

## Q：QKV 为什么要有 K，直接 QV 不行吗

> 来源：百度 / AI Infra 一面

**普通回答**：Q 和 K 做匹配，V 是被检索的值。

**更好的回答**：

**信息检索的类比**：
- Q = 查询（"我想找什么"）
- K = 键（"我能提供什么"）
- V = 值（"我的实际内容"）
- Attention = 用 Q 和 K 计算相关性 → 加权聚合 V

**如果去掉 K，直接 Q×V^T**：
```
score = Q × V^T   (没有 K)
output = softmax(score) × V
```
- 此时 V 同时承担"匹配"和"内容"两个角色
- "匹配分数"和"输出内容"被耦合 → 表达力受限

**K 和 V 分离的好处**：
1. **解耦匹配与内容**：K 负责"能否被找到"，V 负责"找到后提供什么"
2. **维度灵活性**：K 和 V 可以有不同维度（GQA/MQA 就是减少 KV heads）
3. **更丰富的学习信号**：W_K 和 W_V 可以学习不同的特征空间
4. **数学证明**：去掉 K 等价于约束 W_K = W_V → 模型自由度减少

**实验验证**：去掉 K 的模型性能明显下降（尤其在需要精确匹配的任务上）。

**考察点**：从表达力和解耦的角度解释 K 的必要性。

---

## Q：为什么 Decoder-only 比 Encoder-Decoder 更主流

> 来源：百度 / AI Infra 一面 · 飞腾 / AI Infra 实习

**普通回答**：Decoder-only 更简单，效果也好。

**更好的回答**：

**架构对比**：
- Encoder-Decoder（T5, BART）：encoder 做双向 attention，decoder 做 causal attention + cross-attention
- Decoder-only（GPT, LLaMA）：只有 causal attention，一个统一的架构

**Decoder-only 胜出的原因**：

1. **Scaling 更友好**：
   - 所有参数都用于同一个任务（next token prediction）
   - Encoder-Decoder 的 encoder 和 decoder 参数分配需要调优
   - 同参数量下 Decoder-only 在生成任务上更强

2. **训练效率**：
   - Decoder-only 每个 token 都有监督信号（predict next token）
   - Encoder-Decoder 中 encoder 端的 token 没有直接损失（只有 decoder 端有）
   - 等效训练数据利用率更高

3. **统一的推理模式**：
   - 所有任务都用自回归生成（问答、翻译、总结...）
   - Encoder-Decoder 的 encoder 需要先跑完再 decode → 两阶段复杂度

4. **In-context learning**：
   - Decoder-only 天然支持 few-shot（prefix 就是 context）
   - Encoder-Decoder 的 "context" 和 "output" 被架构强制分离

5. **KV Cache 简单**：
   - 只有一个 causal attention 的 cache
   - Encoder-Decoder 需要维护 cross-attention 的 KV + self-attention 的 KV

**Encoder-Decoder 仍有优势的场景**：
- 有明确输入-输出结构的任务（翻译、ASR）
- Encoder 可以用更高效的双向 attention
- Whisper、mBART 等仍用 Encoder-Decoder

**考察点**：从 scaling law、训练效率和工程简洁性三个角度解释。

---

## Q：模型参数量和计算量怎么算

> 来源：快手 / AI Infra 实习 · 面经总结

**普通回答**：参数量 = 层数 × 每层参数。

**更好的回答**：

**参数量公式**：

```
Transformer 每层参数:
  Attention: W_Q + W_K + W_V + W_O = 4d²  (标准 MHA)
  FFN: W_gate + W_up + W_down = 3 × d × d_ff
       (d_ff 通常 = 4d 或 8/3×d for SwiGLU)
  LayerNorm: 2 × 2d = 4d (可忽略)
  
  MHA + FFN (SwiGLU, d_ff=8/3×d): 4d² + 3×d×(8d/3) = 4d² + 8d² = 12d²
  
总参数: L × 12d² + V×d (embedding)
```

**常见模型验证**：
| 模型 | L | d | 参数量公式 | 实际 |
|------|---|---|-----------|------|
| LLaMA-7B | 32 | 4096 | 12×32×4096² ≈ 6.4B | 6.7B |
| LLaMA-13B | 40 | 5120 | 12×40×5120² ≈ 12.6B | 13B |
| LLaMA-70B | 80 | 8192 | 12×80×8192² ≈ 64.4B | 70B |

**单次 Forward 计算量（FLOPs）**：
```
矩阵乘 FLOPs = 2 × M × N × K (乘加各算一次)

每层每 token:
  Attention projections: 2 × 4d² = 8d²
  Attention score: 2 × s × d = 2sd (s=seq_len)
  Attention × V: 2sd
  FFN: 2 × 3 × d × d_ff ≈ 16d² (SwiGLU)
  
  总计 ≈ 24d² + 4sd per token per layer

Forward total: L × (24d² + 4sd) × s  tokens
             ≈ 2N × D  (N=参数量, D=token数, 忽略 attention 的 s² 项)
```

**训练 FLOPs = 6ND**（forward 2N + backward 4N）

**推理 FLOPs**：
- Prefill: ≈ 2N × prompt_length
- Decode: ≈ 2N per token（+ KV Cache 读取带宽）

**考察点**：能快速心算模型参数量，理解 6Nd 公式的来源。

---

## Q：DeepSeek V3 的完整结构

> 来源：字节 / AI Infra 实习二面 · 阿里控股 / AI Infra · 小厂 / AI Infra

**普通回答**：MoE + MLA + MTP。

**更好的回答**：

**DeepSeek V3 核心配置**：
- 总参数量：671B，激活参数：37B/token
- 61 层 Transformer，hidden_dim=7168
- Attention：MLA（Multi-head Latent Attention）
- FFN：MoE（256 Expert + 1 Shared Expert，top-8 routing）
- 位置编码：RoPE
- 归一化：RMSNorm（Pre-Norm）

**三大架构创新**：

**1. MLA（已详细覆盖）**：
- KV Cache 压缩到 512 维 latent
- 权重吸收优化 decode
- RoPE 部分单独存（不可吸收）

**2. DeepSeekMoE**：
- 256 routed experts + 1 shared expert（处理通用知识）
- Top-8 routing（每 token 激活 8 个 expert）
- Load balancing：无辅助 loss，用 bias term 动态调整路由

**3. MTP（Multi-Token Prediction）**：
- 训练时同时预测未来 2 个 token（辅助 loss）
- 推理时可做 speculative decoding 的 draft head
- 额外参数很少（共享主模型的大部分权重）

**训练创新**：
- **FP8 训练**：权重和激活用 FP8，累加器 FP32
- **DualPipe**：流水线并行 + All-to-all 通信 overlap
- **DeepEP**：高效 Expert Parallelism 通信库

**推理特性**：
- 每 token 只激活 37B 参数 → 推理成本接近 37B 稠密模型
- 但知识容量 = 671B → 效果超过同规模稠密模型
- KV Cache 极小（MLA）→ 支持超长序列和大 batch

**参数量手算**：
```
每层:
  MLA: ~2d² (压缩投影) + QKV 投影 ≈ 数亿
  MoE FFN: 256 × 3 × d × d_ff_expert ≈ 数百亿
  共享 Expert: 3 × d × d_ff ≈ 数亿
  
总计 ≈ 671B (多数在 MoE Expert 中)
```

**考察点**：能完整描述 MLA+MoE+MTP 三件套，理解各组件的设计动机和推理影响。

---

## Q：常见模型架构差异（DeepSeek / Qwen / LLaVA）

> 来源：美团实习 / AI Infra

**普通回答**：都是 Transformer 但细节不同。

**更好的回答**：

| 维度 | Qwen-2.5 | DeepSeek V3 | LLaVA |
|------|-----------|-------------|-------|
| 架构 | Dense Transformer | MoE Transformer | Vision Encoder + LLM |
| Attention | GQA | MLA | GQA (LLM部分) |
| FFN | SwiGLU | MoE (256 expert) | SwiGLU |
| Norm | RMSNorm | RMSNorm | RMSNorm |
| 位置编码 | RoPE + YaRN | RoPE | RoPE |
| 特殊设计 | — | MTP, FP8 训练 | Vision-Language Connector |

**Qwen 系列特点**：
- 标准 Dense 架构，靠数据和训练 recipe 取胜
- GQA 减少 KV Cache
- YaRN 支持长上下文扩展

**DeepSeek V3 特点**：
- MoE 实现大参数小计算（671B 参数，37B 激活）
- MLA 极致压缩 KV Cache
- 训练创新（FP8, DualPipe）

**LLaVA（多模态）特点**：
- Vision Encoder（CLIP ViT）提取图像特征
- Projection layer（MLP/Resampler）将视觉 token 映射到 LLM 空间
- LLM（LLaMA/Vicuna）处理文本+视觉 token
- 训练：先 pretrain connector，再全量 finetune

**对 Infra 的不同挑战**：
- Dense（Qwen）：标准 TP/PP，推理最简单
- MoE（DeepSeek）：EP 通信、load balancing、显存管理复杂
- 多模态（LLaVA）：ViT prefill + LLM decode 的异构调度

**考察点**：理解不同架构对部署和优化提出的不同工程挑战。

---

## Q：训练与推理在资源消耗上的区别

> 来源：快手 / AI Infra 实习 · 网易 / AI Infra 校招

**普通回答**：训练要更多显存，推理要低延迟。

**更好的回答**：

| 维度 | 训练 | 推理 |
|------|------|------|
| 显存 | 参数+梯度+优化器+激活值 (16N+) | 参数+KV Cache (2N+) |
| 计算模式 | Forward + Backward (6N per token) | Forward only (2N per token) |
| batch size | 大（数百万 token/step） | 动态（几十到几百并发） |
| 精度 | BF16/FP8（需要梯度精度） | INT8/INT4/FP8（可激进量化） |
| 延迟要求 | 不敏感（跑完一个 epoch 即可） | 敏感（TTFT < 1s, TPOT < 50ms） |
| 吞吐目标 | MFU 最大化 | tokens/s/$ 最大化 |
| 显存瓶颈 | 激活值（随 batch×seq 增长） | KV Cache（随 batch×seq 增长） |
| 通信 | 梯度 All-reduce | KV Cache 传输（PD分离） |

**推理特有的优化**：
- KV Cache 管理（PagedAttention）
- Continuous Batching（动态调度）
- Speculative Decoding（加速单请求）
- 量化到 INT4（训练不能这么激进）
- CUDAGraph（减少 launch overhead）

**训练特有的优化**：
- ZeRO 分片（切分优化器状态）
- Activation Checkpointing（重计算换显存）
- 混合精度训练（FP8/BF16）
- 大 batch + 梯度累积
- 流水线并行（PP）

**考察点**：清晰区分训练和推理的资源特征，能解释为什么同一个模型训练和推理的最优策略不同。
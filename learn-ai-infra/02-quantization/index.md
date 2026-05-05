---
layout: default
title: 量化方法与精度保障
description: INT8/INT4 量化方法分类、PTQ vs QAT、精度保障策略
parent: AI Infra
---

# 量化方法与精度保障

量化是推理加速最常用的手段之一——将 FP16/FP32 权重和激活压缩到 INT8/INT4，减少显存占用和计算量。本文覆盖量化方法分类和精度保障的核心思路。

---

## Q：量化的方法

> 来源：抖音 / AI Infra 一面
**普通回答**：有 INT8 量化和 INT4 量化，分为训练后量化和量化感知训练。

**更好的回答**：

**按时机分类**：

| 类型 | 全称 | 特点 |
|------|------|------|
| PTQ | Post-Training Quantization | 训练后直接量化，无需重训；简单快速 |
| QAT | Quantization-Aware Training | 训练时模拟量化误差，精度更高；成本大 |

**按粒度分类**：
- **Per-tensor**：整个 tensor 共享一个 scale/zero_point，最粗
- **Per-channel**：每个输出通道独立量化参数，权重常用
- **Per-group/Per-block**：每 128 个元素一组，LLM 常用（如 GPTQ、AWQ）

**按目标分类**：
- **Weight-only**：只量化权重（INT4），激活保持 FP16（如 GPTQ、AWQ、GGUF）
- **Weight + Activation**：权重和激活都量化（INT8 W8A8），需要校准数据确定激活范围

**主流方法**：
- **GPTQ**：基于 Hessian 的逐层量化，误差最小化，Weight-only INT4
- **AWQ**（Activation-aware Weight Quantization）：根据激活分布保护重要权重通道
- **SmoothQuant**：将激活的量化难度转移到权重上（数学等价变换），实现 W8A8
- **GGML/GGUF**：CPU 推理生态的量化格式，支持多种混合精度（Q4_K_M 等）

**考察点**：不只知道 INT8/INT4 的概念，还要理解不同方法的适用场景和 trade-off。

---

## Q：哪些方法可以保证量化的准确性

> 来源：抖音 / AI Infra 一面
**普通回答**：用 QAT 或者校准数据可以保证精度。

**更好的回答**：

量化精度保障是一个系统性问题，核心思路是"找到并保护敏感部分"：

**1. 校准（Calibration）**
- 用代表性数据（几百条）跑 forward，统计激活值的分布范围
- 选择合适的 scale：MinMax、Percentile（去掉 outlier）、MSE 最小化
- 校准质量直接决定 PTQ 精度

**2. 混合精度量化（Mixed Precision）**
- 不是所有层对量化同样敏感：第一层、最后一层、attention 的 QKV 投影通常更敏感
- 敏感层保持 FP16，其余层 INT8/INT4
- 可用 sensitivity analysis（逐层量化 + 测 loss）自动选择

**3. Outlier 处理**
- LLM 激活中存在少数极大值（outlier），直接量化会压缩正常值的精度
- SmoothQuant：通过数学变换将 outlier 从激活转移到权重
- LLM.int8()：将 outlier 通道单独用 FP16 计算

**4. 量化感知训练（QAT）**
- 前向传播模拟量化噪声（Straight-Through Estimator）
- 模型学会适应量化误差，精度最优但成本最高
- 大模型 QAT 成本极高，通常只在小模型或关键场景使用

**5. 评估验证**
- Perplexity 对比（WikiText-2、C4）
- 下游任务 benchmark（MMLU、HumanEval）
- 关注 worst-case 而非 average-case（某些任务可能退化明显）

**考察点**：量化不是"一键 INT8"就完事，面试官想看你是否理解精度-效率的 trade-off 以及如何系统性保障质量。

---

## Q：量化策略的选择依据——为何选 INT8，A100 与 H100 对不同精度的支持

> 来源：遂原科技 / AI Infra 实习一面

**普通回答**：INT8 是精度和速度的平衡，A100 支持 INT8。

**更好的回答**：

量化精度的选择取决于硬件支持、模型容忍度和部署目标：

**硬件 Tensor Core 支持矩阵**：

| 精度 | A100 | H100 | 说明 |
|------|------|------|------|
| FP32 | 19.5 TFLOPS | 67 TFLOPS | 训练用 |
| TF32 | 156 TFLOPS | 989 TFLOPS | 默认训练精度 |
| FP16/BF16 | 312 TFLOPS | 989 TFLOPS | 混合精度训练/推理 |
| INT8 | 624 TOPS | 1979 TOPS | 推理量化主力 |
| FP8 (E4M3/E5M2) | 不支持 | 1979 TFLOPS | H100 新增，训练+推理 |
| INT4 | 不原生支持 | 不原生支持 | Weight-only，需特殊 kernel |

**为什么选 INT8（W8A8）**：
- A100/H100 原生 Tensor Core 支持 → 真正的 2× 计算加速（vs FP16）
- 精度损失可控：配合 SmoothQuant / 校准，大多数模型 perplexity 退化 <1%
- 成熟工具链：TensorRT-LLM、vLLM 都有完善支持

**INT4 的定位**：
- Weight-only INT4（GPTQ/AWQ）：减少权重读取量，decode 带宽受限时有效
- 但 A100/H100 没有 INT4 Tensor Core → 实际计算用 FP16 dequant 后做，计算没加速
- 主要收益：显存节省 + bandwidth 减少

**H100 的 FP8 优势**：
- 兼具 INT8 的速度和接近 FP16 的精度
- E4M3（推理用）：3 bit 尾数，精度比 INT8 好
- 可用于训练（E5M2 格式做梯度）

**考察点**：量化不是"越低越好"，要结合硬件实际支持和精度影响来选择。

---

## Q：量化对象是模型权重还是 KV Cache，scale 参数如何确定

> 来源：遂原科技 / AI Infra 实习一面

**普通回答**：量化权重，scale 通过校准确定。

**更好的回答**：

**量化对象分类**：

| 对象 | 场景 | 主流方法 |
|------|------|---------|
| 权重 (Weight) | 减少模型体积和 decode 带宽 | GPTQ、AWQ、INT8 per-channel |
| 激活 (Activation) | 加速矩阵乘（需要 W+A 都量化才能用 INT8 Tensor Core） | SmoothQuant、W8A8 |
| KV Cache | 减少 decode 阶段 KV 读取量 + 节省显存 | Per-token / per-channel INT8/INT4 |

**KV Cache 量化的特殊性**：
- KV Cache 是动态生成的（每 decode step 新增），不像权重是静态的
- 需要 online quantization：生成 K/V 后立即量化存储，读取时 dequant
- 通常用 per-token 或 per-head 的 scale
- INT8 KV Cache 可将显存减半 → batch size 翻倍 → 吞吐大幅提升

**Scale 确定方法**：

1. **权重 scale**：
   - 直接用 weight 的 min/max（per-channel）
   - 或通过 GPTQ/AWQ 等方法联合优化 scale 和 zero-point

2. **激活 scale**：
   - 需要校准数据（calibration set）跑 forward
   - 统计每层激活的动态范围
   - 方法：MinMax、Percentile（如 99.9%，去掉 outlier）、MSE 最优

3. **KV Cache scale**：
   - Per-token dynamic：每个新 token 的 K/V 独立计算 absmax → scale
   - Per-channel static：用校准数据预先确定每个 head 的 scale
   - 动态 scale 更准但有额外计算开销

**考察点**：区分不同量化对象的目的和约束，理解 KV Cache 量化与权重量化的本质差异（动态 vs 静态）。

---

## Q：量化后是否进行过精度损失的评测

> 来源：遂原科技 / AI Infra 实习一面

**普通回答**：看 perplexity 有没有涨太多。

**更好的回答**：

量化评测需要多维度验证，不能只看单一指标：

**1. 通用语言能力**：
- **Perplexity**（WikiText-2、C4）：最直接的退化指标
  - INT8 W8A8：通常退化 < 0.5
  - INT4 Weight-only：退化 0.5-2.0（取决于方法）
  - 退化 > 3 通常不可接受

**2. 下游任务 Benchmark**：
- MMLU（知识）、HumanEval（代码）、GSM8K（数学）
- 不同任务对量化的敏感度不同——数学推理通常最敏感
- 报告 average 的同时关注 worst-case task

**3. 端到端业务指标**：
- 实际业务场景的 A/B 测试
- 如：对话质量评分、RAG 答案准确率、代码通过率
- 这是上线决策的最终依据

**4. 逐层敏感度分析**：
- 逐层量化 + 测 perplexity 退化
- 找出最敏感的层（通常是第一层、最后一层、attention QKV）
- 指导混合精度策略

**5. 边界 Case 检查**：
- 长序列（>4K tokens）下精度是否退化更大
- 特定 domain（代码、数学公式）是否有明显退化
- Outlier 输入是否导致异常输出

**评测流程**：
1. 先跑 Perplexity 快速筛选量化配置
2. 通过的配置跑 Benchmark 对比
3. 最终候选上业务数据做细粒度评测

**考察点**：量化评测不是跑一个数就完事，面试官想看系统性的质量保障思路。

---

## Q：FP8 的两种格式有什么区别 / 分别怎么计算

> 来源：讯飞飞星 / AI Infra 校招 · 混元 / AI Infra

**普通回答**：E4M3 和 E5M2，指数和尾数位数不同。

**更好的回答**：

**FP8 两种格式**：

| 格式 | 符号 | 指数 | 尾数 | 动态范围 | 精度 | 用途 |
|------|------|------|------|---------|------|------|
| E4M3 | 1 bit | 4 bits | 3 bits | ±448 | 较高（3 bit 尾数） | 前向推理（权重、激活） |
| E5M2 | 1 bit | 5 bits | 2 bits | ±57344 | 较低 | 反向传播（梯度） |

**计算方式**（同 IEEE 754 但有特殊处理）：
```
value = (-1)^sign × 2^(exponent - bias) × (1 + mantissa/2^M)

E4M3: bias = 7,  正常范围: 2^(-6) ~ 448
E5M2: bias = 15, 正常范围: 2^(-14) ~ 57344
```

**E4M3 特殊规则**（NVIDIA 定义）：
- 没有 inf（所有 exponent=1111 + mantissa!=0 的值都是有效数）
- 最大值 = 448（而非 inf）
- 有 NaN（exponent=1111, mantissa=111）

**为什么推理用 E4M3，梯度用 E5M2**：
- 推理的权重/激活分布较集中，需要精度（更多尾数位）
- 梯度动态范围大（可能很大或很小），需要更大范围（更多指数位）

**NV 硬件对 FP8 的优化（H100+）**：
- H100 Tensor Core 原生支持 FP8 矩阵乘：`mma.sync` 可接受 E4M3/E5M2 输入
- 计算速度 = INT8 的 2×（989 TFLOPS FP8 vs 同 FP16）
- 输出仍为 FP16/FP32 累加器（不会精度损失在累加上）
- **Per-tensor scale**：每个 tensor 一个 FP32 scale factor，硬件自动 dequant

**FP8 KV Cache 大小计算**：
```
FP8 KV Cache = 2 × num_layers × seq_len × num_kv_heads × head_dim × 1 byte
```
- 对比 FP16 KV Cache 直接减半
- 例：LLaMA-70B, seq=4096, GQA 8 heads, head_dim=128
  - FP16: 2×80×4096×8×128×2 = 1.28 GB/请求
  - FP8: 2×80×4096×8×128×1 = 0.64 GB/请求

**NV FP4（Blackwell 架构 B200+）**：
- 4 bit 浮点，E2M1 格式
- 需要 block-wise scaling：每 32 个元素共享一个 FP8 scale
- 存储 = 4 bit value + amortized scale ≈ 4.5 bit/element

**考察点**：区分 E4M3/E5M2 的设计目的，理解 H100 的 FP8 Tensor Core 工作方式。

---

## Q：数据分布极不均匀时应使用什么量化方案

> 来源：摩尔线程 / AI Infra 实习二面

**普通回答**：用 percentile 校准去掉 outlier。

**更好的回答**：

**场景**：最小值 -10000，其余集中在 [-1, 1]。如果用 MinMax：
- scale = (10000 - (-10000)) / 255 ≈ 78.4（INT8）
- [-1, 1] 范围只映射到约 0.025 个量化阶 → 正常值全部量化为同一个整数 → 精度灾难

**解决方案（从简到复杂）**：

1. **Percentile 校准**：
   - 取 99.9% percentile 而非 min/max 作为量化范围
   - outlier -10000 被 clip → 信息丢失但多数值精度保全

2. **非对称量化 + Outlier Clipping**：
   - 只覆盖 [-1, 1] 范围，outlier clip 或单独处理
   - scale = 2/255 ≈ 0.0078，精度好

3. **Per-channel / Per-group 量化**：
   - 如果 outlier 集中在某些 channel，按 channel 各自定 scale
   - 正常 channel scale 小（精度高），outlier channel scale 大

4. **混合精度（LLM.int8() 方式）**：
   - 检测 outlier channel（绝对值 > 6 的）
   - Outlier channel 保持 FP16 计算
   - 其余 INT8 量化
   - 最适合这种极端分布

5. **SmoothQuant 变换**：
   - 如果 outlier 在激活中，通过数学变换将 outlier 分摊到权重
   - 变换后激活分布变平滑 → 量化友好

**考察点**：不是背算法名字，而是能根据具体数据分布特征选择合适的策略。

---

## Q：剪枝和稀疏化的种类

> 来源：美团实习 / AI Infra

**普通回答**：有结构化剪枝和非结构化剪枝。

**更好的回答**：

**按粒度分类**：

| 类型 | 粒度 | 硬件友好性 | 压缩率 | 精度影响 |
|------|------|-----------|--------|---------|
| 非结构化 | 单个权重 | 差（需要稀疏矩阵格式） | 高(90%+) | 低 |
| 半结构化 (2:4) | 每4个权重取2个 | 好（A100 Tensor Core 原生支持） | 固定50% | 中 |
| 结构化 | 整行/列/头/层 | 最好（直接减小矩阵维度） | 中 | 高 |

**非结构化剪枝**：
- Magnitude pruning：按绝对值大小裁掉最小的权重
- 需要 CSR/CSC 等稀疏格式存储
- 稀疏度 <80% 时通常比 dense 更慢（indirect indexing 开销）
- 代表：SparseGPT（LLM 一次性剪枝到 50-60%）

**半结构化 (N:M Sparsity)**：
- NVIDIA 的 2:4 稀疏：每 4 个连续权重保留 2 个
- A100/H100 Sparse Tensor Core 原生加速（2× FLOPS vs dense）
- 实际收益 ~1.5×（有额外 metadata 开销）
- 适合 inference 加速

**结构化剪枝**：
- 剪整个 attention head / FFN neuron / 整层
- 结果：模型直接变小（如 12 层 → 10 层）
- 不需要特殊硬件/格式支持
- 但精度影响大，通常需要 fine-tuning 恢复

**稀疏化 vs 量化**：
- 可以组合使用（如 2:4 稀疏 + INT8 量化 = 4× 加速理论上限）
- 实践中量化更成熟，稀疏化的工具链/框架支持相对不足

**考察点**：区分三种粒度的特点，知道 2:4 稀疏是目前硬件最友好的方案。

---

## Q：SmoothQuant / AWQ / GPTQ 的作用流程对比

> 来源：智谱 / AI Infra 一面 · 美团实习 / AI Infra

**普通回答**：都是量化方法，减少模型大小。

**更好的回答**：

| 方法 | 量化目标 | 核心思想 | 粒度 | 适用场景 |
|------|---------|---------|------|---------|
| SmoothQuant | W8A8 | 转移量化难度从激活到权重 | Per-channel | 需要 W+A 都量化（Tensor Core INT8） |
| AWQ | W4（weight-only） | 保护重要权重通道 | Per-group | 显存/带宽受限 |
| GPTQ | W4（weight-only） | 逐列量化+误差补偿 | Per-group | 显存/带宽受限 |

**SmoothQuant 流程**：
```
问题：激活中有 outlier → 量化激活很难
解法：Y = (X / s) × (s × W)  (s = per-channel scale)
      → X' = X/s (激活变平滑)
      → W' = s×W (权重吸收难度)
      → 现在 X' 和 W' 都好量化了！
```
- s 的选择：`s = max(|X|)^α / max(|W|)^(1-α)`，α 控制平衡点

**AWQ 流程**：
```
观察：1% 的 salient weight channel 对精度影响巨大
流程：
1. 用校准数据跑 forward，统计每个 channel 的激活值大小
2. 激活值大的 channel = 重要 channel → 对应权重需要更大 scale
3. 对重要 channel 乘以 s > 1（等价于"保护"它在量化后的精度）
4. 对不重要 channel 除以 s（牺牲它的精度）
5. 然后做标准 per-group INT4 量化
```

**GPTQ 流程**：
```
基于 OBQ（Optimal Brain Quantization）:
1. 逐层处理（不需要全模型 forward）
2. 每层的权重矩阵按列量化：
   a. 量化一列 → 产生误差
   b. 用 Hessian 逆（H^{-1}）将误差最优分配到未量化列
   c. 重复直到所有列量化完
3. 关键：Hessian 的 Cholesky 分解可以复用 → 高效
```

**选型指南**：
- 需要 Tensor Core INT8 加速 → SmoothQuant (W8A8)
- 只想减显存/带宽（decode 加速） → AWQ 或 GPTQ (W4A16)
- AWQ vs GPTQ：AWQ 更快（不需要 Hessian 计算），GPTQ 精度略好

**考察点**：能说清每种方法的核心 insight 和适用场景，而不是混为一谈。

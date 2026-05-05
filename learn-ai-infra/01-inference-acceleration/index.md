---
layout: default
title: 推理加速全景：从 KV Cache 到部署
description: KV Cache 原理与显存计算、推理加速总体流程、TVM 编译框架
parent: AI Infra
---

# 推理加速全景：从 KV Cache 到部署

大模型推理的核心瓶颈是显存带宽和计算量。本文覆盖 KV Cache 的原理与大小计算、推理加速的总体思路、以及 TVM 等编译优化框架。

---

## Q：KV Cache 的作用以及为什么有 KV Cache

> 来源：抖音 / AI Infra 一面 · 百度 / AI Infra 暑期实习一面
**普通回答**：KV Cache 缓存了之前 token 的 Key 和 Value，避免重复计算。

**更好的回答**：

Transformer 的自回归解码每生成一个 token 都需要对所有已生成 token 做 attention。如果不缓存，每步都要重新计算前面所有 token 的 K、V 投影，计算量随序列长度平方增长。

KV Cache 的作用：
1. **避免重复计算**：将历史 token 的 K、V 向量缓存下来，新 token 只需计算自身的 Q、K、V，然后与缓存拼接做 attention
2. **将 O(n²) 的重复计算降为 O(n)**：每步只做一行 attention 而非整个矩阵
3. **将推理从 compute-bound 转为 memory-bound**：瓶颈从计算变成读取缓存的带宽

代价是显存占用随序列长度线性增长，这也是为什么长序列推理需要 PagedAttention、GQA/MQA 等技术来压缩 KV Cache。

**考察点**：理解 autoregressive decoding 的计算模式，以及空间换时间的工程 trade-off。

---

## Q：KV Cache 大小计算

> 来源：抖音 / AI Infra 一面 · 百度 / AI Infra 暑期实习一面
**普通回答**：KV Cache 大小 = 层数 × 2 × 序列长度 × 隐藏维度 × 数据精度。

**更好的回答**：

公式拆解（以单个请求为例）：

```
KV Cache 显存 = 2 × num_layers × seq_len × num_kv_heads × head_dim × bytes_per_element
```

- **2**：K 和 V 各一份
- **num_layers**：每层都有独立的 KV Cache
- **seq_len**：当前已生成的序列长度（动态增长）
- **num_kv_heads**：GQA/MQA 下 KV head 数可能小于 Q head 数
- **head_dim**：通常 128
- **bytes_per_element**：FP16 = 2 bytes，INT8 = 1 byte

举例：LLaMA-2 70B（80 层，8 KV heads/GQA，head_dim=128，FP16）：
- 单 token 的 KV Cache = 2 × 80 × 1 × 8 × 128 × 2 = 327,680 bytes ≈ 320 KB
- 4096 序列长度 = 320 KB × 4096 ≈ 1.28 GB（单请求）
- batch_size=32 → 约 41 GB，几乎占满一张 A100 80GB 的一半

这就是为什么 vLLM 的 PagedAttention 要做显存分页管理——KV Cache 是推理时最大的显存消耗。

**考察点**：能否快速估算具体模型的 KV Cache 开销，判断 batch 上限和显存瓶颈。

---

## Q：推理加速思路总体流程

> 来源：抖音 / AI Infra 一面 · 百度 / AI Infra 暑期实习一面
**普通回答**：量化、剪枝、蒸馏、算子优化。

**更好的回答**：

推理加速是一个多层次的系统工程，从上到下可以分为：

**1. 算法层**
- 模型压缩：量化（INT8/INT4）、剪枝（结构化/非结构化）、蒸馏
- 架构优化：GQA/MQA 减少 KV head、Sliding Window Attention、稀疏注意力

**2. 计算图层**
- 算子融合：将多个小算子合并减少 kernel launch 和显存读写（如 FlashAttention 融合 QKV + softmax + output）
- 编译优化：TVM、TensorRT、torch.compile 做图优化和代码生成
- 内存优化：PagedAttention、KV Cache 量化、Activation Recomputation

**3. 系统层**
- Batching 策略：Continuous Batching（vLLM）、Dynamic Batching
- 并行策略：Tensor Parallelism、Pipeline Parallelism、Sequence Parallelism
- 调度：Prefill/Decode 分离、Speculative Decoding（小模型草稿 + 大模型验证）

**4. 硬件层**
- GPU kernel 优化：CUDA kernel hand-tuning、Triton
- 异构部署：CPU offload、多卡通信优化（NCCL）

实际工作中，这些手段是组合使用的，比如 vLLM = PagedAttention + Continuous Batching + Tensor Parallel。

**考察点**：是否有全局视角理解推理优化的层次结构，而不是只知道单点技术。

---

## Q：TVM 框架

> 来源：抖音 / AI Infra 一面
**普通回答**：TVM 是一个深度学习编译器，可以把模型编译到不同硬件上。

**更好的回答**：

TVM（Tensor Virtual Machine）是一个端到端的深度学习编译栈，核心思路是将模型从框架表示（PyTorch/TF）编译为特定硬件的高效代码：

**编译流程**：
1. **前端导入**：将 PyTorch/ONNX 模型转为 TVM 的高层 IR（Relay/Relax）
2. **图优化**：算子融合、常量折叠、layout 变换（NCHW→NHWC）
3. **算子调度**：对每个算子生成 schedule（循环变换、tiling、向量化）
4. **代码生成**：生成 CUDA/OpenCL/LLVM 等目标代码

**核心优势**：
- **AutoTVM / MetaSchedule**：自动搜索最优 schedule，避免手写 kernel
- **跨硬件**：同一模型可编译到 GPU、CPU、ARM、FPGA、NPU
- **性能接近手写**：通过搜索空间探索，生成的代码可媲美 cuDNN/TensorRT

**与 TensorRT 对比**：
- TensorRT 是 NVIDIA 闭源的，只支持 NVIDIA GPU，但对 N 卡优化极深
- TVM 开源、跨硬件，但需要调优时间，新算子支持可能滞后

**实际应用场景**：边缘部署（手机/IoT）、非 NVIDIA 硬件、需要自定义算子的场景。

**考察点**：理解编译优化的分层思路，以及 TVM 与 TensorRT 的定位差异。

---

## Q：Decode 阶段输出 token 的选择方案

> 来源：百度 / AI Infra 暑期实习一面

**普通回答**：用 argmax 选概率最大的 token，或者用采样。

**更好的回答**：

Decode 阶段每步生成一个 token，选择策略直接影响生成质量和多样性：

**确定性方案**：
- **Greedy Decoding**：每步取 argmax，速度最快但容易重复、缺乏多样性
- **Beam Search**：维护 top-k 个候选序列，选全局最优。适合翻译等需要精确输出的任务，但计算量 ×k

**随机采样方案**：
- **Temperature Sampling**：logits / T 后 softmax，T<1 更确定，T>1 更随机
- **Top-k Sampling**：只在概率最高的 k 个 token 中采样
- **Top-p (Nucleus) Sampling**：在累积概率达到 p 的最小集合中采样，自适应候选数量
- **Min-p Sampling**：过滤掉概率 < p × max_prob 的 token

**进阶方案**：
- **Speculative Decoding**：小模型快速生成 draft tokens，大模型一次性验证并行接受/拒绝。不改变输出分布，但加速 2-3×
- **Repetition Penalty**：对已出现 token 降低 logits，减少重复
- **Structured Output**：通过 constrained decoding（如 Outlines/LMQL）强制输出符合 JSON schema

**系统实现考虑**：
- Top-k/Top-p 需要排序 vocab（通常 32K-128K），是 decode 中非 trivial 的计算
- Speculative Decoding 需要 draft model 与 target model 共享 KV Cache 管理

**考察点**：不只知道 greedy vs sampling，要理解各策略的适用场景和系统实现代价。

---

## Q：Prefill-Decode 分离（PD 分离）的大致流程

> 来源：百度 / AI Infra 暑期实习一面

**普通回答**：把 prefill 和 decode 分开执行。

**更好的回答**：

LLM 推理有两个截然不同的阶段：

| 阶段 | 特征 | 瓶颈 |
|------|------|------|
| Prefill | 一次处理整个 prompt，大矩阵乘 | Compute-bound |
| Decode | 逐 token 生成，每步只算一行 | Memory-bound |

**为什么要分离**：
- 混合调度时，prefill 的大计算会阻塞 decode 请求的 TTFT（Time To First Token）
- 两个阶段的 GPU 利用模式完全不同，混在一起都达不到最优

**PD 分离架构**：
1. **Prefill 节点**：专门处理 prompt 编码，高 compute 利用率，batch 大 prompt
2. **Decode 节点**：专门做自回归生成，优化 memory bandwidth，大 batch 小计算
3. **KV Cache 传输**：Prefill 完成后将 KV Cache 通过高速互联（NVLink/RDMA）传给 Decode 节点
4. **调度器**：根据请求到达模式动态分配 prefill/decode 资源比例

**挑战**：
- KV Cache 传输延迟（可能几十 GB）
- Prefill/Decode 比例随负载变化，资源分配需要动态调整
- 实现：DistServe、Splitwise、Mooncake 等系统

**考察点**：理解 prefill 和 decode 的计算特征差异，以及分离后的系统架构挑战。

---

## Q：讲讲 MoE 模型

> 来源：百度 / AI Infra 暑期实习一面

**普通回答**：MoE 是混合专家模型，每次只激活部分专家。

**更好的回答**：

MoE（Mixture of Experts）用稀疏激活实现"大参数量、小计算量"：

**核心结构**：
- 将 Transformer 的 FFN 替换为多个 Expert（每个 Expert 是独立的 FFN）
- 加一个 Router/Gate 网络，为每个 token 选择 top-k 个 Expert 执行
- 典型配置：8/16/64 个 Expert，每 token 激活 2 个

**优势**：
- 总参数量大（知识容量大），但单 token 计算量 ≈ 等效稠密模型的 1/N × k
- DeepSeek-V2 256 Expert 激活 6，Mixtral 8×7B 激活 2（等效计算量 ≈ 12B）

**推理挑战（AI Infra 视角）**：
- **显存**：所有 Expert 权重都要加载，即使每次只用 top-2 → 显存需求 = 全参数量
- **负载不均**：某些 Expert 被频繁选中（hot expert），造成 GPU 利用不均
- **通信**：Expert Parallelism 需要 All-to-All 通信，将 token 路由到对应 GPU
- **batch 效率**：每个 Expert 实际处理的 token 数不确定，影响 kernel 效率

**优化手段**：
- Expert Parallelism：不同 Expert 放不同 GPU，通过 All-to-All 交换 token
- Expert offloading：冷 Expert 放 CPU/SSD，热 Expert 常驻 GPU
- Load balancing loss：训练时加辅助损失鼓励均匀路由

**考察点**：MoE 面试重点不是模型效果，而是推理部署的工程挑战和解决方案。

---

## Q：讲讲多种 Attention 头以及特点

> 来源：百度 / AI Infra 暑期实习一面

**普通回答**：有 MHA、GQA、MQA。

**更好的回答**：

不同 Attention 头设计本质是在"模型表达力"和"KV Cache 显存/带宽"之间做 trade-off：

| 类型 | Q heads | KV heads | KV Cache 大小 | 代表模型 |
|------|---------|----------|--------------|---------|
| MHA | H | H | 基准 | GPT-3、LLaMA-1 |
| MQA | H | 1 | 1/H | PaLM、Falcon |
| GQA | H | G (1<G<H) | G/H | LLaMA-2 70B、Mistral |
| MLA | H | 压缩 latent | 极小 | DeepSeek-V2 |

**MHA（Multi-Head Attention）**：
- 每个 head 有独立的 Q、K、V 投影
- 表达力最强，但 KV Cache 最大

**MQA（Multi-Query Attention）**：
- 所有 Q head 共享同一组 K、V
- KV Cache 缩小到 1/H（H=32 则缩小 32×）
- 代价：模型质量略有下降

**GQA（Grouped-Query Attention）**：
- 折中方案：每 G 个 Q head 共享一组 KV
- LLaMA-2 70B 用 8 组 GQA（32 Q heads / 8 KV heads），KV Cache 缩小 4×

**MLA（Multi-head Latent Attention）**：
- DeepSeek-V2 的方案：将 KV 压缩到低维 latent 空间
- KV Cache 存 latent（维度 512），解码时投影回 KV
- 显存极小，但增加了解码时的计算（latent → KV 投影）

**推理系统影响**：
- KV Cache 越小 → batch size 可以越大 → 吞吐越高
- GQA/MQA 使 decode 阶段更加 memory-bound（读的数据少了）
- MLA 增加了 decode 计算但大幅减少显存，适合长序列

**考察点**：理解各方案的设计动机（KV Cache 压缩）和对推理系统的具体影响。

---

## Q：KV Cache 的维度一般是多少

> 来源：百度 / AI Infra 暑期实习一面

**普通回答**：和 hidden_dim 有关，MLA 是 512。

**更好的回答**：

KV Cache 每个 token 存储的维度取决于 Attention 类型：

**标准 MHA/GQA**：
- 每个 KV head 存 head_dim 维的 K 和 V
- 单层单 token：`num_kv_heads × head_dim × 2`（K 和 V）
- head_dim 通常 = hidden_dim / num_heads
  - LLaMA 系列：head_dim = 128（hidden 4096 / 32 heads）
  - GPT-3：head_dim = 128（hidden 12288 / 96 heads）

**GQA 下的 KV 维度**：
- LLaMA-2 70B：8 KV heads × 128 head_dim = 1024 维（per K and per V）
- LLaMA-3 8B：8 KV heads × 128 = 1024 维
- 对比 MHA 的 LLaMA-1：32 KV heads × 128 = 4096 维

**MLA（DeepSeek-V2）**：
- 不存原始 KV，存压缩后的 latent vector
- latent 维度 = 512（远小于原始 KV 的数千维）
- 解码时通过 up-projection 恢复到完整 KV
- 等效压缩率：512 vs 原始 ~4096+，约 8× 压缩

**对推理的意义**：
- KV 维度直接决定每 token 的显存开销和读取带宽
- Decode 阶段是 memory-bound，KV 维度越小 → 读取越快 → 每步延迟越低

**考察点**：能否结合具体模型配置快速算出 KV Cache 维度，理解不同设计的量级差异。

---

## Q：Decode 阶段属于 compute-bound 还是 memory-bound

> 来源：遂原科技 / AI Infra 实习一面

**普通回答**：Decode 是 memory-bound。

**更好的回答**：

**结论：Decode 阶段是典型的 memory-bound**。

**原因分析**：
- Decode 每步只生成 1 个 token，但需要读取整个模型权重 + 全部 KV Cache
- 计算量：batch_size × hidden² 级别（一个向量 × 矩阵）
- 数据读取量：全部权重（几十 GB）+ KV Cache
- Arithmetic Intensity = FLOPs / Bytes ≈ batch_size × 2（远低于 GPU 的计算/带宽比）

**Roofline 分析**：
- A100 的计算/带宽比（ops:byte ratio）≈ 312 TFLOPS / 2 TB/s ≈ 156
- Decode 的 arithmetic intensity ≈ 2 × batch_size
- 只有 batch_size > 78 时才可能变成 compute-bound（实际受 KV Cache 显存限制很难达到）

**KV Cache 量化提升的是什么**：
- 减少每步需要从 HBM 读取的字节数 → 直接提升 memory bandwidth 利用效率
- INT8 KV Cache：读取量减半 → decode latency 理论减半（在 memory-bound 下）
- 附带好处：显存占用减少 → 可以支持更大 batch 或更长序列

**对比 Prefill**：
- Prefill 是 compute-bound：大矩阵乘（seq_len × hidden × hidden），arithmetic intensity 高

**考察点**：理解 roofline model，以及 KV Cache 量化的收益机制（不是减少计算，而是减少读取）。

---

## Q：Attention 模块在端到端延迟中所占比例

> 来源：遂原科技 / AI Infra 实习一面

**普通回答**：Attention 是主要耗时模块。

**更好的回答**：

需要分 Prefill 和 Decode 两阶段讨论：

**Prefill 阶段**：
- Attention 计算复杂度 O(n² × d)，FFN 复杂度 O(n × d²)
- 当 seq_len > hidden_dim 时（如 seq_len=4096, hidden=4096），Attention 和 FFN 耗时接近
- 使用 FlashAttention 后，Attention 耗时大幅减少（减少 HBM 访问）
- 典型占比：Attention 30-40%，FFN 50-60%，其余（LayerNorm、Residual）<10%

**Decode 阶段**：
- Attention：读取 KV Cache（大量数据）+ 一个向量点积
- FFN：一个向量 × 权重矩阵
- 两者都是 memory-bound，耗时正比于需要读取的参数量
- FFN 参数量通常是 Attention 的 2-3×（因为 FFN 有 gate/up/down 三个大矩阵）
- 典型占比：Attention 25-35%，FFN 55-65%

**关键洞察**：
- Attention 并非总是瓶颈——在 decode 阶段，FFN 的参数读取往往更耗时
- 但 Attention 的 KV Cache 占显存大，限制了 batch size，间接影响吞吐
- FlashAttention 主要优化 Prefill 的 Attention，对 Decode 改善有限

**考察点**：不要想当然说"Attention 是瓶颈"，要能区分不同阶段的实际耗时分布。

---

## Q：A100 的理论显存带宽上限

> 来源：遂原科技 / AI Infra 实习一面

**普通回答**：A100 带宽是 2 TB/s。

**更好的回答**：

| 指标 | A100 80GB (SXM) | A100 40GB (PCIe) | H100 (SXM) |
|------|-----------------|------------------|-------------|
| HBM 带宽 | 2.0 TB/s | 1.6 TB/s | 3.35 TB/s |
| 显存容量 | 80 GB HBM2e | 40 GB HBM2e | 80 GB HBM3 |
| FP16 算力 | 312 TFLOPS | 312 TFLOPS | 989 TFLOPS |
| INT8 算力 | 624 TOPS | 624 TOPS | 1979 TOPS |
| NVLink | 600 GB/s | 无 | 900 GB/s |

**为什么带宽重要**：
- Decode 阶段是 memory-bound，性能上限 = 带宽 / 每 token 需读取的数据量
- 7B 模型 FP16 权重 = 14 GB，每 decode step 需读取全部权重
- 理论上限：2000 GB/s ÷ 14 GB ≈ 143 tokens/s（单请求）
- 实际因 KV Cache 读取、kernel overhead 等，约 70-80% 利用率

**Compute-Bandwidth 比（Ops:Byte）**：
- A100：312 TFLOPS / 2 TB/s = 156 ops/byte
- 含义：如果每读 1 byte 做不到 156 次运算，就是 memory-bound
- Decode 的 ops/byte ≈ 2 × batch_size，远低于 156

**实际面试追问方向**：
- 给定模型大小和带宽，估算理论最大 decode 速度
- 量化（INT8）如何同时减少读取量和利用 INT8 算力（双重收益）

**考察点**：将硬件参数和实际推理性能联系起来，不是背数字而是能做估算。

---

## Q：MLA 为什么比 MHA 好 / 权重吸收是什么

> 来源：蚂蚁 / AI Infra 校招一面 · 字节 / AI Infra 校招一面 · 阿里 / AI Infra 实习

**普通回答**：MLA 用低维 latent 压缩 KV Cache，显存更小。

**更好的回答**：

**MLA（Multi-head Latent Attention，DeepSeek-V2）核心设计**：

标准 MHA 每层 KV Cache 维度 = num_heads × head_dim（如 128 × 128 = 16384 维）。MLA 的做法：

```
编码阶段：
  c = W_dkv × h       # h (hidden) → c (latent, 维度 d_c=512)
  KV Cache 只存 c！   # 512 vs 16384，压缩 32×

解码阶段：
  K = W_uk × c        # c → K (恢复到 num_heads × head_dim)
  V = W_uv × c        # c → V
  然后做标准 attention
```

**权重吸收（Absorption）优化**：

直接实现：每步 decode 需要 c → K → Q×K 三步。但可以数学变换：

```
Q × K^T = Q × (W_uk × c)^T = Q × c^T × W_uk^T = (Q × W_uk^T) × c^T
                                                    ↑ 吸收到 Q 的投影中
```

将 W_uk 吸收到 Q 的投影矩阵中：
- 预计算 W_q' = W_q × W_uk^T（offline，一次性）
- 推理时：score = Q' × c^T（Q' = h × W_q'）
- 不需要从 c 恢复完整 K！直接用 latent c 做 attention

**权重吸收的问题**：
- 吸收后的权重矩阵 W_q' 维度变大
- 但如果同时用了 RoPE（旋转位置编码），RoPE 部分的 KV 不能吸收（因为 RoPE 是 position-dependent）
- DeepSeek 的解法：将 head 分为 RoPE 部分（单独存）+ 非 RoPE 部分（可吸收）
- 实际 KV Cache = latent c (512) + RoPE 部分的 K (少量)

**为什么比 MHA 好**：
- KV Cache 显存减少 ~30×（512 vs 16384）→ 支持更长序列/更大 batch
- Decode 阶段读取 KV Cache 带宽大幅减少 → 延迟降低
- 代价：decode 时多了 latent→KV 的投影计算（但计算可以被吸收优化掉）

**MLA Decode 计算访存比分析**：
- 无吸收：读 c → 投影得 K,V → attention。访存 = batch × seq × d_c
- 有吸收：读 c → 直接算 score。计算量与 seq_len 相关，访存与 batch × seq × d_c 相关
- 与 seq_len 相关：seq 越长，读 cache 越多，越 memory-bound
- 与 batch_size 相关：batch 越大，矩阵乘越大，可能转 compute-bound

**考察点**：理解 MLA 的 latent 压缩 + 权重吸收的数学原理，以及 RoPE 导致的不完全吸收问题。

---

## Q：Self-Attention 为什么要除以 √d

> 来源：科大讯飞 / AI Infra 校招

**普通回答**：防止点积值过大。

**更好的回答**：

**数学原因**：
- 假设 Q 和 K 的每个元素独立同分布，均值 0，方差 1
- 点积 Q·K = Σ(q_i × k_i)，i=1..d
- 每个 q_i × k_i 的方差 = Var(q)×Var(k) = 1
- 求和 d 项：Var(Q·K) = d
- 标准差 = √d → 点积值的量级随 d 增长

**除以 √d 的效果**：
- 将点积的方差归一化为 1，不随 head_dim 变化
- Softmax 输入的量级稳定 → 梯度健康

**如果不除**：
- d=128 时，点积标准差 ≈ 11
- Softmax(11) ≈ 1.0，Softmax(-11) ≈ 0.0 → 分布接近 one-hot
- 梯度接近 0（softmax 饱和区）→ 训练困难

**考察点**：从方差分析角度解释，而非只说"防止过大"。

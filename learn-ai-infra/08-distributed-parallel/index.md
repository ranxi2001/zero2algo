---
layout: default
title: 分布式训练与推理
description: TP/PP/EP/SP 并行策略、All-reduce/All-to-all 通信、ZeRO、通信优化
parent: AI Infra
---

# 分布式训练与推理

大模型单卡放不下，分布式并行是必修课。本文覆盖各种并行策略、通信算子原理和工程优化，是 AI Infra 面试的核心考点之一。

---

## Q：并行策略有哪些 / TP 怎么切的

> 来源：数坤科技 / AI Infra 实习 · 阿里控股 / AI Infra 一面 · 快手 / AI Infra 一面 · 面经总结 · 拼多多 / AI Infra

**普通回答**：有数据并行、模型并行、流水线并行。

**更好的回答**：

**并行策略总览**：

| 策略 | 切分维度 | 通信算子 | 适用场景 |
|------|---------|---------|---------|
| DP | 数据（batch） | All-reduce 梯度 | 模型放得下单卡 |
| TP | 模型（hidden dim） | All-reduce / All-gather | 单层太大放不下 |
| PP | 模型（layer） | 点对点 Send/Recv | 层数多、卡间带宽有限 |
| EP | MoE Expert | All-to-all | MoE 模型 |
| SP | 序列（seq_len） | All-gather / Reduce-scatter | 长序列激活值大 |

**Tensor Parallelism（TP）详解**：

以 Transformer 的一个线性层 Y = XW 为例（X: [B, S, H], W: [H, H']）：

**MLP 层的切法**：
```
Gate/Up projection (W1: H→4H):
  列切（Column Parallel）: W1 切成 [H, 4H/tp] × tp 份
  每卡算 Y_local = X × W1_local → 输出 [B, S, 4H/tp]
  无需通信!

Down projection (W2: 4H→H):
  行切（Row Parallel）: W2 切成 [4H/tp, H] × tp 份
  每卡算 Y_local = X_local × W2_local → 输出 [B, S, H]
  需要 All-reduce 求和!
```

**为什么 W1 列切 + W2 行切**：
- W1 列切后输出正好是分片的（每卡 4H/tp），可直接喂给下一层
- W2 行切后每卡得到 partial sum，All-reduce 合并得到完整输出
- **一个 Transformer 层只需 2 次 All-reduce**（Attention 一次 + MLP 一次）

**Attention 层**：
- QKV 投影按 head 切分（每卡负责 H/tp 个 head）
- Output projection 行切 → All-reduce

**TP 的 All-reduce 次数**：
- 每个 Transformer 层：2 次（Attention output + MLP output）
- 对于有 GQA 的模型：KV heads < Q heads 时，某些卡可能共享 KV

**考察点**：能画出 TP 的切分方式，解释为什么用列切+行切的组合来最小化通信。

---

## Q：Reduce-scatter 和 All-to-all 通信

> 来源：鹅厂 / AI Infra 实习 · 面经总结

**普通回答**：Reduce-scatter 是先 reduce 再 scatter，All-to-all 是全交换。

**更好的回答**：

**All-reduce = Reduce-scatter + All-gather**：

```
Reduce-scatter（每卡得到部分结果）:
  GPU0: [A0, A1, A2, A3]
  GPU1: [B0, B1, B2, B3]
  GPU2: [C0, C1, C2, C3]
  GPU3: [D0, D1, D2, D3]
  → GPU0 得到 A0+B0+C0+D0
    GPU1 得到 A1+B1+C1+D1
    GPU2 得到 A2+B2+C2+D2
    GPU3 得到 A3+B3+C3+D3

All-gather（每卡收集所有部分）:
  → 每卡都得到完整的 [sum0, sum1, sum2, sum3]
```

**通信量**：
- All-reduce = Reduce-scatter + All-gather
- 数据量 N，p 卡：每卡发送 2×N×(p-1)/p（ring 实现）

**All-to-all（全交换）**：
```
输入: 每卡有发给所有卡的数据
  GPU0: [to_0, to_1, to_2, to_3]
  GPU1: [to_0, to_1, to_2, to_3]
  ...
输出: 每卡收到所有卡发给自己的数据
  GPU0: [from_0, from_1, from_2, from_3]
```

**应用场景**：
- **Reduce-scatter**：ZeRO-2/3 分片梯度、SP 中 TP 层输出分片
- **All-gather**：ZeRO-3 临时收集完整参数、SP 中进入 TP 层前收集完整序列
- **All-to-all**：MoE 的 EP（token 路由到对应 Expert 所在的 GPU）

**EP（Expert Parallelism）中的 All-to-all**：
```
Dispatch（前向 All-to-all）:
  每卡的 tokens → 根据 Router 决策 → All-to-all 发送到 Expert 所在卡

Combine（反向 All-to-all）:
  Expert 计算完成 → All-to-all 将结果发回原卡
```
- DeepSeek V3 的 DeepEP：优化 All-to-all 的通信 overlap 和 load balancing

**考察点**：区分各通信算子的数据流模式和适用场景。

---

## Q：通信延迟的影响因素与优化

> 来源：数坤科技 / AI Infra 实习 · 快手 / AI Infra 一面 · 鹅厂 / AI Infra 实习

**普通回答**：带宽和延迟。

**更好的回答**：

**通信延迟 = α + n/BW**（α 启动延迟，n 数据量，BW 带宽）

**影响因素**：
1. **互联拓扑与带宽**：
   - NVLink（节点内）：600-900 GB/s
   - PCIe：64 GB/s（Gen5）
   - InfiniBand/RDMA（节点间）：400 Gb/s ≈ 50 GB/s
   - 节点内 >> 节点间，通信密集的并行（TP）应在节点内

2. **通信量**：取决于并行策略和模型大小
3. **通信算子效率**：Ring vs Tree vs Direct 实现
4. **通信-计算 overlap 程度**

**优化手段**：

| 方法 | 原理 |
|------|------|
| 计算-通信 overlap | 一边算当前层一边传上一层的梯度 |
| 通信量压缩 | 梯度压缩（TopK/随机稀疏化）、FP16 通信 |
| 拓扑感知 | TP 放节点内（NVLink），PP/EP 放节点间 |
| 流水线 | PP 的 1F1B 减少空泡 |
| Async AllReduce | 不等通信完成就继续下一层计算 |

**场景题：节点内 NVLink + 节点间部分 RDMA 的集群设计**：
- 节点内 8 卡 NVLink → TP=8
- 节点间 RDMA → PP 跨节点 或 EP 跨节点
- 无 RDMA 的节点 → 只用 DP（通信量最小）
- 整体：TP8 × PP × DP

**考察点**：能根据硬件拓扑设计并行策略，理解通信与计算的 overlap 技术。

---

## Q：ZeRO（DeepSpeed）的通信量分析

> 来源：面经总结

**普通回答**：ZeRO 把参数/梯度/优化器状态切分到多卡。

**更好的回答**：

**ZeRO 三个 Stage**：

| Stage | 切分内容 | 显存节省 | 额外通信 |
|------|---------|---------|---------|
| Stage 1 | 优化器状态 | ~4× | 无额外 |
| Stage 2 | + 梯度 | ~8× | 无额外 |
| Stage 3 | + 参数 | ~N× (N=GPU数) | All-gather 参数 |

**通信量对比**（假设模型参数量 Ψ）：

- **标准 DP**：All-reduce 梯度 = 2Ψ（reduce-scatter Ψ + all-gather Ψ）
- **ZeRO-1**：通信量 = 2Ψ（和标准 DP 相同！只是分片了优化器状态）
- **ZeRO-2**：通信量 = 2Ψ（reduce-scatter 梯度后不需要 all-gather 梯度——每卡只更新自己的分片）
- **ZeRO-3**：通信量 = 3Ψ（前向 all-gather Ψ + 反向 all-gather Ψ + reduce-scatter 梯度 Ψ）

**论文 vs 实现的 gap**：
- 论文说 Stage 1 和 2 通信量不增加，但实现中 reduce-scatter 和 all-reduce 的效率可能不同
- ZeRO-3 的 all-gather 与计算 overlap：每层 forward 前 all-gather 本层参数，用完释放
- 实际吞吐取决于 overlap 效率和网络带宽

**考察点**：准确区分三个 stage 的通信量差异，理解 Stage 3 增加通信量但换来最大显存节省。

---

## Q：流水线并行（PP）与调度策略

> 来源：小厂 / AI Infra RL · 面经总结

**普通回答**：PP 把模型按层切分到多卡，用 1F1B 减少空泡。

**更好的回答**：

**基本概念**：
- 将 L 层模型切成 P 段，每段放一张卡
- 朴素实现：一个 micro-batch 从头到尾走完 → (P-1)/P 的时间是空泡

**调度策略演进**：

**GPipe**：
- 将 mini-batch 切成 M 个 micro-batch
- 所有前向 → 所有反向
- 空泡率 = (P-1)/(P+M-1)，M 越大空泡越小

**1F1B（One Forward One Backward）**：
- 稳态时每卡交替执行一个前向和一个反向
- 不需要同时缓存所有 micro-batch 的激活 → 显存更优
- 空泡率同 GPipe，但 peak memory 更低

**Interleaved 1F1B（Megatron）**：
- 每卡负责多个不连续的层段（如卡 0 负责 layer 0-3 和 layer 8-11）
- 空泡率降低到 1/(P+M-1) × 常数

**Zero-Bubble（零气泡）**：
- 将反向分为 B（计算梯度）和 W（权重更新）两部分
- B 优先级高于 W，灵活插入 W 到空泡中
- 理论上可实现接近零空泡

**DualPipe（DeepSeek V3）**：
- 结合双向流水线 + 计算/通信 overlap
- 在 MoE 的 EP all-to-all 通信期间执行其他计算

**考察点**：理解不同调度策略如何减少 pipeline bubble，以及显存和计算效率的 trade-off。

---

## Q：Megatron SP（Sequence Parallelism）的实现方式

> 来源：面经总结

**普通回答**：SP 把序列维度切分到多卡。

**更好的回答**：

**动机**：
- TP 只并行了 Attention 和 MLP 的矩阵乘
- LayerNorm 和 Dropout 在 TP 中每卡都存完整的 [B, S, H] 激活 → 显存冗余
- SP 将这些非 TP 区域也按序列维度切分

**Megatron SP 实现**：

```
TP 区域（矩阵乘）:  每卡有完整序列的部分 hidden → All-reduce

切换为 SP 时:
- TP 输出 All-reduce → 改为 Reduce-scatter（每卡得到部分序列）
- 进入 TP 前 → All-gather 恢复完整序列

非 TP 区域（LayerNorm、Dropout）:
- 每卡只处理 S/tp 长度的序列 → 显存节省 tp×
```

**数据流**：
```
[完整序列] → All-gather → [TP: Attention/MLP] → Reduce-scatter → [SP: LayerNorm/Dropout]
```

**通信量**：
- 不增加额外通信！只是将 All-reduce 拆成 Reduce-scatter + All-gather 并在不同位置执行
- 数学上等价，但中间可以以分片形式存在 → 节省显存

**与 Ring Attention / Ulysses SP 的区别**：
- Megatron SP 是配合 TP 使用的，不改变计算逻辑
- Ring Attention：在 Attention 计算内部按 KV 序列维度分布
- DeepSpeed Ulysses：按 head 维度分布 → All-to-all 交换

**考察点**：理解 Megatron SP 不增加通信量但节省非 TP 区域的激活显存。

---

## Q：DualPipe 原理（DeepSeek V3）

> 来源：百度 / AI Infra 实习 · 面经总结 · 小厂 / AI Infra RL

**普通回答**：DualPipe 是 DeepSeek 的双向流水线。

**更好的回答**：

**问题背景**：
- DeepSeek V3 用了 EP（Expert Parallelism），Expert 分布在不同节点
- EP 的 All-to-all 通信量大、延迟高
- 传统 PP 的 bubble 浪费计算资源

**DualPipe 核心思想**：

将 Transformer 层的计算分为 4 个阶段：
```
一层 Transformer = [Attention] + [A2A dispatch] + [FFN/Expert] + [A2A combine]
```

DualPipe 的关键：**将 All-to-all 通信与相邻层的 Attention 计算重叠**：
```
时间→
GPU A: [Attn_L1] [A2A_L1 + Attn_L2] [FFN_L1 + A2A_L2] [FFN_L2] ...
         ↑ 计算    ↑ 通信和计算重叠    ↑ 通信和计算重叠
```

**双向的含义**：
- 将 micro-batch 分为两组，从流水线两端同时灌入
- 正向从 stage 0→N，反向从 stage N→0
- 中间 stage 同时处理来自两个方向的 micro-batch
- 效果：bubble 进一步减少

**与 Zero-Bubble 的关系**：
- Zero-Bubble：将 backward 拆为 B（梯度计算）和 W（权重更新），W 填充 bubble
- DualPipe：更聚焦于 MoE 的 All-to-all 通信 overlap

**实际效果**：
- DeepSeek V3 训练中 All-to-all 通信被几乎完全隐藏在计算中
- 计算效率接近无通信开销的理想值

**考察点**：理解 DualPipe 本质是利用计算-通信 overlap 消除 MoE 的 All-to-all 开销。

---

## Q：RL 训练全流程与 Rollout 优化

> 来源：小厂 / AI Infra RL · 百度 / AI Infra 实习 · 美团 / AI Infra 实习

**普通回答**：用 PPO 训练，有 Actor 和 Critic。

**更好的回答**：

**RLHF/RL 训练全流程（以 PPO 为例）**：

```
参与的模型:
- Actor (Policy): 生成回答
- Critic (Value): 估算 value
- Reference Model: 计算 KL 惩罚
- Reward Model: 打分

每一步训练:
1. Rollout: Actor 生成 N 条回答 (推理)
2. Scoring: Reward Model 对回答打分
3. Advantage 计算: Critic 估值 + GAE
4. Policy Update: PPO loss 更新 Actor
5. Value Update: MSE loss 更新 Critic
```

**PPO vs GRPO**：

| 特性 | PPO | GRPO (DeepSeek) |
|------|-----|------|
| Critic | 需要（额外模型） | 不需要 |
| Baseline | Critic 的 value 估计 | 同 prompt 多条回答的均值 |
| 显存 | 4 个模型 | 3 个模型（省 Critic） |
| 方差 | 低（Critic 学习 value） | 稍高但足够 |

**Rollout 是训练瓶颈**：
- Rollout 占 RLHF 训练 70-80% 的时间
- 因为是自回归推理（每 token 串行），不是矩阵乘（无法并行加速）
- Policy MFU（Model FLOPs Utilization）通常只有 30-40%

**MFU 计算公式**：
```
MFU = 实际 FLOPs / (训练时间 × GPU 峰值 FLOPs)
6Nd 公式: 单次 forward+backward 的 FLOPs ≈ 6 × N(参数量) × D(token数)
```

**Rollout 优化手段**：
1. **推理引擎加速**：用 vLLM/SGLang 做 rollout（Continuous Batching + PagedAttention）
2. **Rollout 量化**：Actor 在推理时用 INT8/FP8（不影响梯度计算）
3. **异步 Rollout**：rollout 和 training 在不同 GPU 上并行执行
   - Actor weight 同步延迟 → on-policy 变 slightly off-policy
   - 需要重要性采样修正（`π_new/π_old`）
4. **Speculative Decoding**：加速 rollout 的 decode 阶段
5. **预训练权重同步**：每次 Actor 更新后，将新权重广播到推理引擎（通常通过共享内存或 NCCL broadcast）

**考察点**：理解 RL 训练的四个模型角色和资源瓶颈，知道 rollout 优化是关键。

---

## Q：训练稳定性与容错

> 来源：阶跃星辰 / AI Infra 实习

**普通回答**：用 checkpoint 做断点续训。

**更好的回答**：

**大规模训练的稳定性挑战**：
- 几千张 GPU 训练几周，任何单卡故障都会中断全部计算
- 平均无故障时间（MTBF）：千卡集群约几小时到一天

**容错手段**：

| 层次 | 方法 | 代价 |
|------|------|------|
| Checkpoint | 定期保存全量状态到远端存储 | 保存时训练暂停（几分钟） |
| Async Checkpoint | 后台异步保存（DeepSpeed, Megatron） | 额外显存存 snapshot |
| Redundant Compute | 多份副本计算相同数据 | 2× 资源浪费 |
| Elastic Training | 动态增减节点（不中断训练） | 实现复杂 |
| In-memory Checkpoint | 状态存在其他节点的内存中 | 占用通信带宽 |

**具体实践**：
- **快速恢复**：故障后只需从最近 checkpoint 恢复，而非从头训练
- **梯度累积容错**：某步的 gradient 异常（NaN/Inf）→ 跳过该步
- **Loss spike 检测**：监控 loss 突然上升，自动回退到上一个 checkpoint
- **数据确定性**：保存 dataloader 状态，恢复后数据顺序一致

**FP8 训练的稳定性问题**（DeepSeek V3）：
- FP8 动态范围有限，训练中 activation/gradient 可能溢出
- 需要 per-tensor dynamic scaling
- DeepSeek 的做法：延迟 scaling（用上一步的 absmax 估算本步的 scale）

**考察点**：从故障概率出发理解为什么容错是大规模训练的刚需，了解 checkpoint 的优化变体。
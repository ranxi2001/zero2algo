---
layout: default
title: 推理服务框架与调度
description: FlashAttention、PagedAttention、Chunk Prefill、Continuous Batching、Speculative Decoding、CUDAGraph
parent: AI Infra
---

# 推理服务框架与调度

大模型推理从单次 forward 到高吞吐线上服务，需要一整套调度和内存管理机制。本文覆盖 vLLM/SGLang 体系中最高频的面试考点。

---

## Q：FlashAttention 为什么能加速，计算过程是什么

> 来源：京东 / AI Infra 一面 · 快手 / AI Infra 一面 · 快手 / AI Infra 二面 · 阿里国际 / AI Infra 实习 · 混元 / AI Infra · 面经总结

**普通回答**：FlashAttention 通过 tiling 减少 HBM 访问来加速。

**更好的回答**：

**标准 Attention 的问题**：
```
Q (N×d) × K^T (d×N) → S (N×N) → softmax → P (N×N) × V (N×d) → O (N×d)
```
- 中间矩阵 S 和 P 大小 O(N²)，必须写回 HBM
- N=4096, heads=32 → S 和 P 各占 ~2 GB，反复读写 HBM 是瓶颈

**FlashAttention 核心思想——IO-Aware Tiling**：

不存储完整的 N×N attention matrix，改为分块计算 + online softmax：

```
将 Q 分成 Br 大小的块，K/V 分成 Bc 大小的块
for each Q_block (外层循环):
    初始化 O_block = 0, m = -inf, l = 0  (running max 和 running sum)
    for each K_block, V_block (内层循环):
        1. 从 HBM 加载 Q_block, K_block, V_block 到 SRAM (Shared Memory)
        2. 计算局部 S_block = Q_block × K_block^T  (在 SRAM 中)
        3. Online Softmax 更新:
           - m_new = max(m_old, rowmax(S_block))
           - 修正旧累加: O_block *= exp(m_old - m_new) * (l_old / l_new)
           - 累加新块: O_block += exp(S_block - m_new) × V_block
           - 更新 l_new
        4. 不写回 S_block 到 HBM!
    写回 O_block 到 HBM
```

**为什么能加速**：
1. **HBM 访问量从 O(N²) 降到 O(N²d/M)**：M 是 SRAM 大小，分块后每块在 SRAM 内完成
2. **不存储 N×N 矩阵**：空间从 O(N²) 降到 O(N)
3. 实际收益：2-4× wall-clock 加速（Attention 部分），长序列收益更大

**V1 → V2 → V3 的改进**：

| 版本 | 关键改进 |
|------|---------|
| V1 | 基础 tiling + online softmax |
| V2 | 外层循环改为 Q（非 K/V）→ 减少非矩阵乘 FLOPs，更好利用 Tensor Core；前向减少一次 rescale |
| V3 | 利用 Hopper 的 TMA + warpgroup，producer-consumer 异步流水线，WGMMA 指令 |

**FlashDecoding**（decode 阶段的 FlashAttention）：
- Decode 时 Q 只有 1 个 token，但 KV 很长
- 在 KV 的 seq_len 维度并行（多个 block 各算一段），最后 combine kernel 合并
- Combine kernel 占总耗时很小（<5%），因为只是对 block 数做 reduction

**考察点**：核心是 online softmax + tiling 避免 N×N 矩阵写回 HBM，追问通常到 V2 为何外层改 Q。

---

## Q：PagedAttention 具体是什么 / 为什么效果好

> 来源：京东 / AI Infra 一面 · 字节 / AI Infra 后端 · 快手 / AI Infra 实习 · 美团北斗 / AI Infra · 小厂 / AI Infra

**普通回答**：PagedAttention 像操作系统的虚拟内存一样管理 KV Cache。

**更好的回答**：

**问题背景**：
- KV Cache 是推理最大的显存消耗（7B 模型 batch=32 可占 40+ GB）
- 传统方式为每个请求预分配最大序列长度的 KV 空间 → 巨大浪费（平均利用率仅 20-40%）
- 不同请求长度不同，预分配导致内存碎片，batch size 受限

**PagedAttention 设计**：

借鉴 OS 虚拟内存的分页机制：
1. 将 KV Cache 分成固定大小的 **block**（如 16 tokens 一块）
2. 维护一个 **block table**（类似页表）记录每个请求的逻辑 block → 物理 block 映射
3. 物理 block 从全局 **block pool** 按需分配，用完归还

```
逻辑视图（请求A）:  [block0] → [block1] → [block2] → ...
                       ↓           ↓           ↓
物理显存:           [slot 5]   [slot 12]   [slot 3]  ← 不连续!
```

**为什么效果好**：
- **消除碎片**：物理 block 不需要连续，任意空闲 slot 都能用 → 显存利用率接近 100%
- **按需分配**：生成几个 token 就分配几个 block，不预分配最大长度
- **Prefix Cache / Copy-on-Write**：相同 prefix 的请求可共享 block（如 system prompt），写时复制
- **结果**：同等显存下 batch size 提升 2-4×，吞吐翻倍

**Attention 计算变化**：
- 标准 Attention 要求 KV 在内存中连续
- PagedAttention kernel 需要根据 block table 做间接寻址——按 block 读取 KV，计算局部 attention 后合并
- 计算量不变，多了一次查表开销（可忽略）

**vLLM 如何结合**：
- Scheduler 管理 block 分配/回收/抢占
- 抢占（preemption）：显存不够时 swap out 某个请求的 KV block 到 CPU，后续 swap in 恢复
- Prefix Caching：相同 prompt 前缀的 block 可以复用（hash 匹配）

**考察点**：理解分页思想如何解决 KV Cache 碎片和预分配浪费，以及对 attention kernel 的影响。

---

## Q：Chunk Prefill 是什么 / 为了解决什么问题

> 来源：京东 / AI Infra 一面 · 鹅厂 / AI Infra 实习

**普通回答**：把 prefill 分成多个 chunk 执行。

**更好的回答**：

**问题**：
- 长 prompt 的 prefill 一次计算耗时很长（如 32K tokens 可能几秒）
- 在此期间，正在 decode 的请求被阻塞 → 它们的 TPOT（Time Per Output Token）剧增
- 用户体验：已经在对话中的请求突然"卡住"

**Chunk Prefill 解决方案**：

将一个长 prefill 请求切分为多个 chunk，与 decode 请求交替执行：

```
时间线（无 Chunk Prefill）:
|---- Prefill A (32K tokens, 3s) ----|  Decode B  |  Decode B  |
                                       ↑ B 被阻塞 3 秒

时间线（有 Chunk Prefill）:
|Prefill A chunk1|Decode B|Prefill A chunk2|Decode B|Prefill A chunk3|Decode B|
                  ↑ B 每隔一个 chunk 就能执行一次
```

**关键设计点**：
- **Chunk size 选择**：太小 → prefill 效率低（每块 kernel launch overhead）；太大 → decode 延迟高
- 通常 chunk_size = 512~2048 tokens
- **不是** A 的 prefill 和 A 的 decode 交替——是 A 的 prefill chunk 和 **其他请求** B 的 decode 交替
- 一个 iteration 内：scheduler 将一个 prefill chunk + 多个 decode 请求组成一个 batch

**与 PD 分离的关系**：
- PD 分离：prefill 和 decode 分到不同 GPU 节点
- Chunk Prefill：在同一节点上通过时间片共享解决延迟问题
- 两者可以结合：prefill 节点内部也可以用 chunk prefill 来服务多个请求

**SGLang 的实现**：
- RadixAttention + Chunk Prefill 结合
- Scheduler 每个 step 决定分配多少 budget 给 prefill vs decode

**考察点**：理解 prefill 阻塞 decode 的问题本质，以及 chunk granularity 的 trade-off。

---

## Q：Continuous Batching / vLLM 的调度机制

> 来源：小厂 / AI Infra RL · 快手 / AI Infra 实习 · 阿里国际 / AI Infra 实习

**普通回答**：Continuous Batching 是动态加入和移除请求。

**更好的回答**：

**Static Batching 的问题**：
- 传统方式：凑齐一个 batch 的请求，等所有请求都生成完（最长的那个）才开始下一个 batch
- 短请求早完成但要等长请求 → GPU 利用率低，延迟高

**Continuous Batching（iteration-level scheduling）**：
- 每个 decode step 都是一个调度点
- 一个请求完成（生成 EOS）→ 立即移出 batch，新请求立即加入
- batch 大小动态变化，GPU 始终满载

**vLLM Scheduler 流程**：
```
每个 step:
1. 检查 waiting 队列有无新请求需要 prefill
2. 检查 running 队列中有无请求完成（EOS / max_tokens）
3. 检查显存是否够分配新 block
4. 如果不够 → preempt（抢占优先级最低的请求，swap out 其 KV Cache）
5. 组装本步 batch: prefill tokens + decode tokens
6. 执行 forward → 输出 token
7. 更新状态
```

**抢占（Preemption）策略**：
- 显存不够时，选择一个请求暂停
- Swap：将其 KV Cache 从 GPU 搬到 CPU（后续 swap in 恢复）
- Recompute：直接丢弃 KV Cache，恢复时重新 prefill
- 策略：FCFS（先来先服务）或 priority-based

**SGLang 的 RadixAttention**：
- 用 Radix Tree 管理所有请求的 KV Cache prefix
- 自动识别共享 prefix 并复用（比 vLLM 的 hash-based prefix cache 更灵活）
- 对多轮对话、few-shot 场景效果好

**考察点**：理解 iteration-level 调度的核心优势，以及显存不够时的抢占策略。

---

## Q：Speculative Decoding（投机采样）原理

> 来源：面经总结 · 小厂 / AI Infra

**普通回答**：小模型先生成 draft，大模型验证。

**更好的回答**：

**动机**：
- Decode 阶段是 memory-bound，每步都要读全部权重但只算 1 token
- 如果能一次验证多个 token → 有效利用 compute → 加速

**流程**：
```
1. Draft Phase: 小模型（如 68M 参数）快速自回归生成 K 个 draft tokens
   - 小模型速度极快（参数少 → 读取少）
   - 生成 draft tokens: t1, t2, ..., tk 以及对应概率 q(t_i)

2. Verify Phase: 大模型一次性 forward 处理 K+1 个 position
   - 输入: [context, t1, t2, ..., tk]
   - 输出: 每个位置的概率分布 p(t_i)

3. Accept/Reject (保证分布不变):
   for i = 1 to K:
       if p(t_i) >= q(t_i): accept t_i
       else: 以概率 p(t_i)/q(t_i) accept, 否则 reject 并 resample from (p - q)+
   第一个 reject 之后的所有 token 丢弃

4. 期望加速: 如果 acceptance rate = α, 期望每次生成 1/(1-α) 个 token
```

**关键性质**：
- 数学上保证输出分布与大模型独立解码完全一致（无质量损失）
- 加速倍数取决于 draft model 与 target model 的分布接近程度

**变体**：
- **Medusa**：不用独立 draft model，在大模型上加多个 head 并行预测未来 token
- **EAGLE**：用大模型的 hidden states 训练轻量 draft head
- **Self-speculative**：大模型跳层做 draft
- **MTP（Multi-Token Prediction）**：DeepSeek V3 的方案，训练时就学多步预测

**系统挑战**：
- Draft model 和 target model 需要共享 KV Cache 管理
- Verify 的 batch 维度不规则（每个请求 accept 数量不同）
- vLLM 和 SGLang 的实现差异在于 KV Cache 管理和 batch 组织方式

**考察点**：理解 accept/reject 保证分布不变的数学原理，以及与 MTP/Medusa 的对比。

---

## Q：CUDAGraph 是什么 / 为什么更适合 decode 阶段

> 来源：小马智行 / AI Infra 实习

**普通回答**：CUDAGraph 把一系列 kernel 录制下来重放，减少 launch overhead。

**更好的回答**：

**问题**：
- 每次 kernel launch 有 CPU 端开销（~5-10μs/kernel）
- 大模型一次 forward 有数百个小 kernel
- Decode 阶段计算量很小（每步只处理 1 token），但 kernel 数不变
- 当计算时间接近 launch overhead 时，CPU 成为瓶颈

**CUDAGraph 原理**：
```cpp
// 录制阶段（只执行一次）
cudaStreamBeginCapture(stream);
  kernel_A<<<...>>>();
  kernel_B<<<...>>>();
  kernel_C<<<...>>>();
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph);

// 重放阶段（每次 decode step）
cudaGraphLaunch(graphExec, stream);  // 一次 launch 执行整个图
```

- 录制时不真正执行，只记录 kernel 序列和参数
- 重放时整个图作为一个单元提交给 GPU → CPU 只有一次 launch 开销

**为什么更适合 decode**：
- Decode 每步 input shape 固定（batch_size × 1）→ 图可以复用
- Prefill 的 seq_len 每次不同 → 每次 shape 变化需要重新录制 → 不适合
- Decode 的单步计算快，launch overhead 占比大 → CUDAGraph 收益明显

**为什么占更多显存**：
- CUDAGraph 录制时绑定了所有 kernel 的输入输出地址
- 这些 tensor 在图的生命周期内不能被释放或移动
- 等于把 decode 一步需要的所有中间 tensor "钉住"了
- 对比正常执行：PyTorch 可以在 kernel 间复用显存

**vLLM 中的实践**：
- 为不同 batch_size 预录制多个 CUDAGraph
- Decode 时根据当前 batch size 选择对应的 graph 重放
- Batch size 变化时切换 graph（有少量 overhead）

**考察点**：理解 CUDAGraph 的 trade-off：减少 launch overhead vs 增加显存占用。

---

## Q：Prefix Cache 机制

> 来源：字节 / AI Infra 后端

**普通回答**：缓存相同 prefix 的 KV Cache 避免重复计算。

**更好的回答**：

**应用场景**：
- System prompt：同一个应用的所有请求共享相同的 system prompt（可能几千 token）
- Few-shot examples：相同的 few-shot 示例
- 多轮对话：前几轮的上下文

**实现方式**：

1. **Hash-based（vLLM）**：
   - 将 token sequence hash 为 key
   - 全局维护 `hash → KV Cache blocks` 的映射
   - 新请求来时，检查 prefix 的 hash 是否命中 → 命中则直接复用 block

2. **Radix Tree（SGLang）**：
   - 用 Radix Tree（前缀树）组织所有活跃/缓存的 KV Cache
   - 自动找到最长公共前缀
   - 支持更细粒度的 prefix 共享（不限于完整 prompt 匹配）

**收益**：
- 4K token 的 system prompt × 100 并发 → 节省 99% 的重复 prefill 计算
- 多轮对话第 N 轮只需 prefill 新消息部分

**LRU 淘汰**：
- 显存有限，缓存的 prefix 需要 LRU 淘汰
- 热门 prefix（高命中率）常驻，冷 prefix 被回收

**考察点**：理解 prefix cache 的收益场景和 hash vs radix tree 的实现区别。

---

## Q：AF 分离是什么 / 为什么有了 PD 分离还要 AF 分离

> 来源：快手 / AI Infra 一面

**普通回答**：AF 分离是把 Attention 和 FFN 分开。

**更好的回答**：

**PD 分离的局限**：
- PD 分离解决了 prefill（compute-bound）和 decode（memory-bound）的资源争抢
- 但 decode 节点内部，每个 Transformer 层仍包含 Attention 和 FFN，它们特征不同：
  - **Attention**：需要读 KV Cache（随 seq_len 线性增长），是 decode 的显存瓶颈
  - **FFN**：固定大小的权重矩阵乘，计算模式固定

**AF 分离（Attention-FFN Disaggregation）**：
- 将 Attention 计算和 FFN 计算分配到不同的硬件
- Attention 节点：配大显存（存 KV Cache），带宽优先
- FFN 节点：配高算力，对显存需求相对固定

**为什么有价值**：
- MoE 模型中 FFN 参数量巨大（DeepSeek V3 有 256 个 Expert），但每 token 只激活 8 个
- Attention 参数少但 KV Cache 大（长序列场景）
- 分离后：FFN 节点可以用 EP 做专家并行，Attention 节点专注 KV Cache 管理

**Mooncake 系统的 KV-Cache Centric 设计**：
- KV Cache 作为独立的分布式存储层
- Prefill 节点生成 KV Cache → 存入分布式 KV Store
- Decode 节点从 KV Store 拉取需要的 KV Cache
- 好处：KV Cache 可以跨请求共享、按需加载、独立扩缩容

**考察点**：理解 PD 分离只是第一步，AF 分离进一步细分了 decode 内部的异构计算需求。

---

## Q：如何设计高吞吐低延迟的推理服务架构

> 来源：网易 / AI Infra 校招 · OPPO / AI Infra 实习 · 阿里云 / AI Infra

**普通回答**：用 vLLM 部署，加量化和 Tensor Parallel。

**更好的回答**：

**系统设计需要考虑的层面**：

**1. 请求调度层**：
- Continuous Batching（iteration-level scheduling）
- 优先级调度：延迟敏感请求优先
- Admission control：过载时拒绝新请求而非所有请求变慢

**2. 显存管理层**：
- PagedAttention：消除 KV Cache 碎片
- Prefix Cache：复用公共前缀的 KV Cache
- 动态 batch size：根据当前 KV Cache 占用调整并发数

**3. 计算优化层**：
- FlashAttention：减少 Attention 的 HBM 访问
- 量化（INT8/FP8）：减少带宽需求 + 利用低精度 Tensor Core
- CUDAGraph：decode 阶段减少 launch overhead
- Speculative Decoding：提升单请求 decode 速度

**4. 分布式层**：
- Tensor Parallel（节点内 NVLink）
- Pipeline Parallel（跨节点）
- PD 分离 + Chunk Prefill

**5. 可观测性**：
- 指标：TTFT（首 token 延迟）、TPOT（每 token 延迟）、Throughput（tokens/s）
- Profiling：哪个阶段是瓶颈
- SLO 监控：延迟 P99 是否达标

**延迟 vs 吞吐 Trade-off**：
- 大 batch → 高吞吐但每请求延迟增加（排队 + 共享 compute）
- 小 batch → 低延迟但 GPU 利用率低
- 解法：根据 SLO 动态调整 batch 上限

**考察点**：系统性地覆盖从调度到计算到分布式的全链路，而不是单点技术。
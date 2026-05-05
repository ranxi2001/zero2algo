---
layout: default
title: CUDA 编程与 GPU 架构
description: GPU 硬件架构、CUDA 内存模型、GEMM 优化、Reduce 优化、Triton 算子开发
parent: AI Infra
---

# CUDA 编程与 GPU 架构

AI Infra 面试中 CUDA/GPU 相关题出现频率极高。本文覆盖 GPU 架构、CUDA 内存模型、经典算子优化（GEMM、Reduce）和 Triton 开发。

---

## Q：CUDA 编程中并行性（Parallelism）与并发性（Concurrency）的区别

> 来源：沐曦 / AI Infra 实习一面

**普通回答**：并行是同时执行，并发是交替执行。

**更好的回答**：

在 CUDA 语境下这两个概念有具体含义：

**并行（Parallelism）— 同一 kernel 内部**：
- 数千个线程真正同时执行同一段代码（SIMT 模型）
- 一个 block 内的 32 个线程组成 warp，在同一时钟周期执行相同指令
- 这是 GPU 的核心优势：大规模数据并行

**并发（Concurrency）— 多个任务之间**：
- 多个 kernel 或操作在时间上重叠执行
- 实现方式：CUDA Streams
  - 不同 stream 的 kernel 可以并发执行（如果 SM 资源够）
  - 同一 stream 内的操作顺序执行
- 还有：kernel 执行与 H2D/D2H 数据传输的 overlap

**具体例子**：
```cuda
// 并行：一个 kernel 内 1024 个线程同时做向量加法
add_kernel<<<blocks, 1024>>>(a, b, c);

// 并发：两个独立 kernel 在不同 stream 上重叠
kernel_A<<<grid, block, 0, stream1>>>();
kernel_B<<<grid, block, 0, stream2>>>();

// 并发：计算与传输 overlap
cudaMemcpyAsync(d_b, h_b, size, H2D, stream1);  // 传输 batch 2
compute_kernel<<<grid, block, 0, stream2>>>();     // 计算 batch 1
```

**硬件层面**：
- 并行：一个 SM 同时执行多个 warp（warp scheduler 交替调度隐藏延迟）
- 并发：多个 SM 可以分别执行不同 kernel（需要资源不冲突）

**考察点**：CUDA 编程中"并行"指 SIMT 数据并行，"并发"指多 stream/多任务的时间重叠。

---

## Q：GPU 硬件架构 / SM 的具体结构

> 来源：百度 / AI Infra 暑期实习一面 · 太初 / AI Infra 实习一面

**普通回答**：GPU 有很多 SM，每个 SM 有很多 CUDA core。

**更好的回答**：

以 A100（Ampere 架构）为例：

**整体层次**：
```
GPU
├── GPC (Graphics Processing Cluster) × 8
│   └── TPC (Texture Processing Cluster) × 2
│       └── SM (Streaming Multiprocessor) × 2
└── 总计 108 个 SM（A100 实际启用）
```

**单个 SM 内部结构**（A100）：

```
SM
├── 4 个 Sub-partition（Processing Block）
│   ├── Warp Scheduler × 1（每个 sub-partition）
│   ├── Dispatch Unit × 1
│   ├── INT32 units × 16
│   ├── FP32 units × 16
│   ├── FP64 units × 8
│   └── Load/Store units × 8
├── Tensor Core × 4（第三代，支持 TF32/BF16/INT8）
├── Register File：256 KB（每 SM）
├── Shared Memory / L1 Cache：192 KB（可配置比例）
└── L2 Cache：40 MB（全局共享）
```

**关键概念**：
- **Warp**：32 个线程，SIMT 执行，是 SM 调度的基本单位
- **Occupancy**：SM 上活跃 warp 数 / 最大 warp 数，越高越好隐藏延迟
- **Tensor Core**：专用矩阵乘单元，一次做 4×4 或更大的矩阵乘累加
- **Register File**：超快但有限（256KB/SM），每线程用太多 register → occupancy 下降

**线程层次映射**：
```
Grid (所有 block) → GPU 整体
Block → 映射到一个 SM
Warp (32 threads) → SM 的调度单位
Thread → CUDA core 执行
```

**考察点**：能画出 SM 内部结构，理解 warp/block/grid 到硬件的映射关系。

---

## Q：CUDA 内存结构

> 来源：百度 / AI Infra 暑期实习一面

**普通回答**：有全局内存、共享内存、寄存器。

**更好的回答**：

**CUDA 内存层次（从快到慢）**：

| 层次 | 作用域 | 大小 | 带宽 | 延迟 |
|------|--------|------|------|------|
| Register | 单线程 | ~256KB/SM | ~20 TB/s | 0 cycle |
| Shared Memory | Block 内共享 | 最大 164KB/SM (A100) | ~19 TB/s | ~20 cycles |
| L1 Cache | SM 级 | 与 Shared Memory 共享 192KB | ~19 TB/s | ~30 cycles |
| L2 Cache | 全 GPU 共享 | 40MB (A100) | ~5 TB/s | ~200 cycles |
| Global Memory (HBM) | 全 GPU + Host 可见 | 80GB (A100) | 2 TB/s | ~400 cycles |
| Constant Memory | 只读，全局 | 64KB + cache | 接近 L1（命中时） | cache miss 时同 Global |

**编程中如何使用**：

```cuda
__global__ void kernel() {
    // Register: 局部变量
    int local_var = threadIdx.x;

    // Shared Memory: __shared__ 声明
    __shared__ float smem[256];

    // Global Memory: 通过指针访问
    float val = global_array[idx];
}

// Constant Memory: __constant__ 声明
__constant__ float weights[1024];
```

**优化原则**：
1. **数据复用 → Shared Memory**：多个线程需要同一数据，先从 Global 加载到 Shared，再多次读
2. **合并访问（Coalesced Access）**：相邻线程访问连续地址，一次内存事务完成
3. **Bank Conflict 避免**：Shared Memory 分 32 个 bank，同 warp 线程访问同 bank 会串行化
4. **Register Pressure**：用太多 register → 编译器 spill 到 Local Memory（实际在 Global）→ 巨慢

**AI Infra 实践**：
- GEMM tiling：将矩阵子块加载到 Shared Memory 做乘法
- FlashAttention：利用 Shared Memory 做 online softmax，避免写 attention matrix 到 HBM
- KV Cache：放在 Global Memory（必须的），通过合并访问和量化减少读取量

**考察点**：理解内存层次和每层的访问代价，能解释为什么 kernel 优化本质是在管理数据搬运。

---

## Q：Block 级规约过程（求一组线程的最大值）

> 来源：百度 / AI Infra 暑期实习一面 · 太初 / AI Infra 实习一面

**普通回答**：用树形规约，每步线程数减半。

**更好的回答**：

**完整的 Block-level Reduction 流程**：

```cuda
__global__ void blockMax(float* input, float* output, int n) {
    __shared__ float smem[256];  // 假设 blockDim = 256

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Step 1: 每个线程从 Global Memory 加载到 Shared Memory
    smem[tid] = (gid < n) ? input[gid] : -INFINITY;
    __syncthreads();

    // Step 2: 树形规约（Shared Memory 内）
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();  // 每步同步
    }

    // Step 3: Warp-level reduction（最后 32 个线程，无需 __syncthreads）
    if (tid < 32) {
        volatile float* vsmem = smem;
        vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 32]);
        vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 16]);
        vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 8]);
        vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 4]);
        vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 2]);
        vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 1]);
    }
    // 或者用 __shfl_down_sync 做 warp reduction

    // Step 4: Block 结果写回 Global
    if (tid == 0) output[blockIdx.x] = smem[0];
}
```

**优化要点**：

1. **避免 warp divergence**：用 `tid < stride` 而非 `tid % (2*stride) == 0`
2. **Warp-level 无需 sync**：同一 warp 内线程隐式同步（SIMT）
3. **使用 shuffle 指令**（更优）：
```cuda
float val = smem[tid];
val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, 16));
val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, 8));
val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, 4));
val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, 2));
val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, 1));
```
- `__shfl_down_sync` 直接在 register 间交换数据，不经过 Shared Memory

4. **Grid-level reduction**：多 block 的结果需要第二轮 kernel 或 atomic

**复杂度**：O(n/p) 计算 + O(log p) 规约步数，p = 线程数

**考察点**：能否写出正确且高效的 reduction kernel，理解 warp-level 优化。

---

## Q：GEMM（通用矩阵乘法）的常见优化方法

> 来源：三星 / AI Infra 实习一面 · 太初 / AI Infra 实习一面

**普通回答**：用 tiling 和 shared memory。

**更好的回答**：

GEMM (C = A × B) 是 AI 推理/训练最核心的算子，优化层次：

**1. Tiling（分块）— 最核心**：
```
将 M×K × K×N 的大矩阵乘分为小块：
- Thread Block Tile: 128×128（一个 block 负责 C 的一个子块）
- Warp Tile: 64×64（一个 warp 负责的子块）
- Thread Tile: 8×8（单线程通过寄存器累加）
```
- 将 A、B 的子块加载到 Shared Memory → 减少 Global Memory 访问次数
- K 维度循环：每次加载 BK 宽的 A 和 B 条带到 smem，做局部乘累加

**2. 数据预取（Double Buffering）**：
- Shared Memory 分两份：一份正在计算，另一份预取下一轮数据
- 隐藏 Global → Shared 的加载延迟

**3. 向量化加载（Vectorized Load）**：
- 用 `float4`（128 bit）一次加载 4 个 float → 减少 load 指令数
- LDG.128 指令：一次从 Global Memory 读 128 bits

**4. Register Tiling**：
- 每个线程在 register 中维护 8×8 的局部累加器
- 减少 Shared Memory 读取次数

**5. Tensor Core（硬件加速）**：
- `wmma::mma_sync` 或 `mma.sync` PTX 指令
- A100 第三代 Tensor Core：一次做 16×16×16 的 FP16 矩阵乘
- 需要数据 layout 对齐：column-major A、row-major B

**6. 避免 Bank Conflict**：
- Shared Memory 的 layout 需要 padding 或 swizzle
- 如 128×(128+8) 的 smem 布局，加 8 列 padding 消除 conflict

**7. Split-K**：
- K 维度太大时，多个 block 各算一部分 K，最后 reduce
- 增加并行度（适合 K 远大于 M×N 的场景）

**性能参考**：
- 手写优化 GEMM 可达 cuBLAS 的 90-95%
- cuBLAS/CUTLASS 是 production 级实现，已高度优化

**考察点**：能否逐层解释 tiling 策略，理解为什么 GEMM 优化本质是最大化数据复用。

---

## Q：CUDA 优化方法总览

> 来源：文远知行 / AI Infra 一面

**普通回答**：用 shared memory、减少分支、合并访存。

**更好的回答**：

CUDA kernel 优化从以下维度系统化思考：

**1. 内存访问优化**：
- Coalesced Access：同 warp 线程访问连续 128B → 一次内存事务
- Shared Memory Tiling：Global→Shared 加载，多次复用
- 向量化 Load/Store：float4 一次 128 bit
- 避免 Bank Conflict：padding 或 swizzle

**2. 计算优化**：
- Warp-level primitives：`__shfl_sync`、`__ballot_sync` 减少 shared memory 通信
- Tensor Core：矩阵乘用 wmma API 或 CUTLASS
- 快速数学函数：`__expf`、`__rsqrtf`（精度略低但快很多）
- 循环展开：`#pragma unroll`

**3. 占用率与延迟隐藏**：
- 控制 register 用量（别超 255 个/线程）
- 控制 shared memory 用量（别占满 → 限制 block 数）
- 保持足够 warp 让 scheduler 切换隐藏延迟

**4. 算子融合（Kernel Fusion）**：
- 将多个小 kernel 合为一个，避免反复读写 Global Memory
- 例：LayerNorm = mean + variance + normalize → 一个 fused kernel

**5. 异步与并发**：
- 多 stream 并发 kernel
- `cp.async`（Ampere+）：异步从 Global 拷贝到 Shared，不占用 compute 资源
- Pipeline：prefetch + compute overlap

**6. Profiling 驱动优化**：
- NSight Compute：kernel 级分析（occupancy、memory throughput、compute utilization）
- NSight Systems：系统级时间线（kernel 间 gap、H2D overlap）
- 关键指标：achieved bandwidth / peak bandwidth、compute utilization

**考察点**：系统性地组织优化手段，并能根据 profiling 结果判断瓶颈在哪。

---

## Q：Triton 算子的实现逻辑 / CUDA vs Triton 的选择

> 来源：遂原科技 / AI Infra 实习一面

**普通回答**：Triton 是 Python 写 GPU kernel 的工具，比 CUDA 简单。

**更好的回答**：

**Triton 是什么**：
- OpenAI 开源的 GPU 编程语言/编译器
- 用 Python 语法编写 kernel，编译到 PTX/SASS
- 抽象层级介于 CUDA（太底层）和 PyTorch（太高层）之间

**核心设计理念**：
- 以 **Block** 为编程单元而非 Thread → 不需要手动管理 thread 协作
- 自动处理 shared memory tiling、coalesced access、warp-level 优化
- 程序员只需指定 block_size 和数据访问 pattern

**Triton 实现 Softmax 示例**：
```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # 加载一行
    row = tl.load(input_ptr + row_idx * n_cols + col_offsets, mask=mask, other=-float('inf'))

    # 计算 softmax
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    tl.store(output_ptr + row_idx * n_cols + col_offsets, result, mask=mask)
```

**CUDA vs Triton 对比**：

| 维度 | CUDA | Triton |
|------|------|--------|
| 编程语言 | C++/PTX | Python |
| 抽象级别 | Thread-level | Block-level |
| 开发效率 | 低（几百行） | 高（几十行） |
| 极限性能 | 最高（手调每一条指令） | 达到 CUDA 的 80-95% |
| 调试难度 | 高 | 低 |
| 适用场景 | 极致性能、复杂 kernel | 快速原型、中等复杂度算子 |

**什么时候选 Triton**：
- 快速验证算法思路
- 中等复杂度算子（softmax、layernorm、attention）
- 不需要极致性能的场景

**什么时候必须 CUDA**：
- 需要手动控制 register、shared memory layout
- Warp-level 精细优化（如 Tensor Core MMA 指令）
- 极致性能（如 CUTLASS 级别的 GEMM）
- Triton 编译器不支持的 pattern

**考察点**：理解 Triton 的抽象层级定位，以及在实际工作中如何选型。

---

## Q：Profiling 与性能指标——如何判断 kernel 是否达到上限

> 来源：遂原科技 / AI Infra 实习一面

**普通回答**：看运行时间和 GPU 利用率。

**更好的回答**：

**核心指标体系**：

**1. Roofline Model（天花板分析）**：
- 计算 Arithmetic Intensity = FLOPs / Bytes_accessed
- 对比硬件的 compute ceiling 和 bandwidth ceiling
- AI < crossover point → memory-bound；AI > crossover → compute-bound

**2. Memory 指标**：
- **Achieved Bandwidth**：实际 bytes/s vs 峰值带宽
  - A100 HBM 峰值 2 TB/s，好的 memory-bound kernel 能达到 80-90%（~1.6-1.8 TB/s）
- **L1/L2 Hit Rate**：cache 命中率，低命中说明访问模式差
- **Bank Conflict**：shared memory 的 bank conflict 数

**3. Compute 指标**：
- **SM Utilization**：SM 被利用的比例
- **Achieved FLOPS**：vs 峰值算力百分比
- **Warp Stall Reasons**：为什么 warp 在等待（memory dependency、sync、full scoreboard）

**4. Occupancy**：
- 活跃 warp 数 / SM 最大 warp 数
- 不是越高越好——有时低 occupancy + 多 register 也能高性能（如 GEMM）

**Profiling 工具**：
- `nsys profile`：系统级时间线，找 kernel 间 gap 和 overlap
- `ncu --set full`：kernel 级详细分析
- 关键命令：`ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`

**判断优化空间**：
- Memory-bound kernel：achieved bandwidth < 80% peak → 优化访问模式
- Compute-bound kernel：achieved FLOPS < 70% peak → 看 warp stall 原因
- Latency-bound：occupancy 低 + stall on dependency → 增加并行度

**考察点**：能否用数据驱动的方式判断 kernel 瓶颈，而不是盲目优化。

---

## Q：Bank Conflict 是什么 / 怎么解决

> 来源：鹅厂 / AI Infra 实习 · 阿里国际 / AI Infra 实习 · 网易 / AI Infra 校招

**普通回答**：多个线程访问同一个 bank 会串行化。

**更好的回答**：

**Shared Memory 的 Bank 结构**：
- Shared Memory 被分为 32 个 bank（对应 warp 的 32 个线程）
- 连续的 4 bytes 分配到连续的 bank（bank 0, 1, 2, ..., 31, 0, 1, ...）
- 同一 warp 内线程同时访问不同 bank → 并行（一个时钟周期）
- 同一 warp 内多线程访问同一 bank 的不同地址 → **串行**（N-way conflict = N 倍延迟）
- 例外：所有线程访问同一 bank 的**同一地址** → broadcast（无 conflict）

**常见 conflict 场景**：
```cuda
__shared__ float smem[32][32];

// 按列访问 → 32-way bank conflict！
// thread i 访问 smem[i][col], 这些地址间隔 32×4=128 bytes = 每隔 32 个 bank 一轮
// 所有线程落在同一个 bank
float val = smem[threadIdx.x][fixed_col];  // ❌

// 按行访问 → 无 conflict
float val = smem[fixed_row][threadIdx.x];  // ✓
```

**解决方法**：

1. **Padding**：
```cuda
__shared__ float smem[32][32 + 1];  // 加一列 padding
// 现在 stride = 33×4 bytes，错开 bank 映射
```

2. **Swizzle（地址变换）**：
```cuda
// 用 XOR 变换索引，使不同线程映射到不同 bank
int new_col = col ^ (row % 32);
```

3. **调整访问模式**：使同 warp 线程访问连续地址

**GEMM 中的 Bank Conflict**：
- Tiling 时 A 子块存 smem，如果 K 维度 stride 恰好是 32 的倍数 → conflict
- 解决：smem 布局加 padding，或用 swizzle

**考察点**：理解 bank 映射规则，能判断给定访问模式是否有 conflict 并给出解决方案。

---

## Q：怎么减少 Launch Kernel Overhead

> 来源：鹅厂 / AI Infra 实习

**普通回答**：用 CUDAGraph 或算子融合。

**更好的回答**：

每次 kernel launch 的 CPU 端开销约 5-10μs，包括：参数设置、driver 调用、kernel 入队。当 kernel 本身执行时间很短（<100μs）时，launch overhead 成为瓶颈。

**优化方法**：

| 方法 | 原理 | 收益 |
|------|------|------|
| CUDAGraph | 录制 kernel 序列一次性提交 | Launch 开销从 N×5μs 降为 1×5μs |
| Kernel Fusion | 多个小 kernel 合为一个 | 减少 launch 次数 + 省 HBM 读写 |
| Persistent Kernel | Kernel 不退出，循环处理多批数据 | 只 launch 一次 |
| CUDA Streams | 多 stream 并发提交，重叠 CPU launch 和 GPU 执行 | 隐藏 launch 延迟 |
| Host-side Batching | CPU 侧攒多个 launch 再一次性 flush | 减少 driver 交互 |

**Persistent Kernel 示例**：
```cuda
__global__ void persistent_kernel(WorkQueue* queue) {
    while (true) {
        Task task = queue->dequeue();  // 等待新任务
        if (task.done) break;
        process(task);                  // 处理任务
    }
}
// 只 launch 一次，kernel 内部循环处理
```

**CUDA_DEVICE_MAX_CONNECTIONS**：
- 控制 GPU 上并发 stream 数量（默认 8）
- 更多 connections → 更好的 kernel 并发 → 但占用更多硬件资源
- 与 launch_bound 关系：launch_bound 限制 kernel 的 register 使用（影响 occupancy），不直接相关

**考察点**：理解 launch overhead 的来源和多种解决方案的适用场景。

---

## Q：Warp Shuffle 指令详解

> 来源：阿里国际 / AI Infra 实习 · 网易 / AI Infra 校招

**普通回答**：Warp shuffle 在 warp 内线程间直接交换 register 数据。

**更好的回答**：

**为什么需要 Warp Shuffle**：
- 同一 warp 的 32 个线程需要共享数据时，传统方法是写入 shared memory 再读出
- Warp shuffle 直接在 register 间传递 → 不经过内存 → 零延迟

**四种 Shuffle 指令**（mask = 0xFFFFFFFF 表示 32 线程全参与）：

```cuda
// 1. shfl_sync: 从指定 lane 读
int val = __shfl_sync(mask, x, src_lane);
// 所有线程拿到 lane src_lane 的 x 值（broadcast）

// 2. shfl_up_sync: 从低 delta 位的 lane 读
int val = __shfl_up_sync(mask, x, delta);
// lane i 拿到 lane (i-delta) 的值，前 delta 个 lane 值不变

// 3. shfl_down_sync: 从高 delta 位的 lane 读
int val = __shfl_down_sync(mask, x, delta);
// lane i 拿到 lane (i+delta) 的值

// 4. shfl_xor_sync: 从 XOR 配对的 lane 读
int val = __shfl_xor_sync(mask, x, lane_mask);
// lane i 拿到 lane (i^lane_mask) 的值 → butterfly pattern
```

**经典应用——Warp Reduction**：
```cuda
float val = thread_data;
val += __shfl_down_sync(0xFFFFFFFF, val, 16);
val += __shfl_down_sync(0xFFFFFFFF, val, 8);
val += __shfl_down_sync(0xFFFFFFFF, val, 4);
val += __shfl_down_sync(0xFFFFFFFF, val, 2);
val += __shfl_down_sync(0xFFFFFFFF, val, 1);
// lane 0 拿到所有 32 个线程的 sum
```
- 5 步完成 32 个线程的 reduction（vs shared memory 需要 5 步 + __syncthreads）

**Butterfly Reduction（shfl_xor）**：
```cuda
val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
val += __shfl_xor_sync(0xFFFFFFFF, val, 8);
val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
val += __shfl_xor_sync(0xFFFFFFFF, val, 1);
// 所有 lane 都拿到完整 sum!
```

**优势对比 Shared Memory**：
- 延迟更低（register 直接传递）
- 不占 shared memory 空间
- 不需要 __syncthreads（warp 内隐式同步）

**考察点**：能写出 warp reduction 代码，理解 shfl_down vs shfl_xor 的区别（部分结果 vs 全部结果）。

---

## Q：CUDA 编译流程与 PTX

> 来源：小马智行 / AI Infra 实习

**普通回答**：CUDA 代码编译成 PTX 再编译成机器码。

**更好的回答**：

**CUDA 编译流水线**：
```
.cu 文件
  → nvcc 前端：分离 host code（CPU）和 device code（GPU）
  → host code → 普通 C++ 编译器（gcc/clang）→ .o
  → device code → CUDA 编译器 (cicc)
    → PTX (Parallel Thread Execution) — 虚拟 ISA
      → ptxas → SASS (实际机器码，特定 GPU 架构)
  → 最终链接为可执行文件
```

**PTX（虚拟指令集）**：
- NVIDIA 定义的中间表示，类似 LLVM IR
- 向前兼容：旧 PTX 可在新 GPU 上运行（JIT 编译为新架构 SASS）
- 文本格式，可读，可手写优化

```ptx
.reg .f32 %f<3>;
ld.global.f32 %f0, [%rd1];    // 从 global memory 加载
add.f32 %f2, %f0, %f1;        // 浮点加法
st.global.f32 [%rd2], %f2;    // 存回 global memory
```

**SASS（实际机器码）**：
- 特定于 GPU 架构（sm_80 = A100，sm_90 = H100）
- 二进制格式，用 `cuobjdump` 可反汇编
- 包含具体的 register 分配、指令调度、bank 分配

**-arch vs -code（nvcc 参数）**：
```bash
nvcc -arch=sm_80 -code=sm_80    # 生成 A100 的 SASS
nvcc -arch=sm_80 -code=compute_80  # 生成 PTX（运行时 JIT）
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_90,code=sm_90  # Fat binary: 多架构
```

**MLIR 的定位**（追问）：
- 编译器基础设施，定义多层次 IR（dialect）
- 可以表示从高层算子图到底层硬件指令的整个链路
- 比 LLVM IR 更适合 ML 编译器（支持 tensor 类型、循环变换等 ML 特有操作）
- Triton 后端、IREE、TensorFlow MLIR 都基于它

**考察点**：理解从 .cu 到 GPU 执行的完整链路，PTX 的作用和跨代兼容性。

---

## Q：H100 相比 A100 有哪些改进 / H 卡 vs L 卡区别

> 来源：讯飞飞星 / AI Infra 校招 · 快手 / AI Infra 校招

**普通回答**：H100 更快，有 FP8 支持。

**更好的回答**：

**H100（Hopper）vs A100（Ampere）核心差异**：

| 维度 | A100 (sm_80) | H100 (sm_90) | 提升 |
|------|-------------|-------------|------|
| HBM | 80GB HBM2e, 2 TB/s | 80GB HBM3, 3.35 TB/s | 带宽 +67% |
| FP16 算力 | 312 TFLOPS | 989 TFLOPS | +3.2× |
| FP8 | 不支持 | 1979 TFLOPS | 全新 |
| Tensor Core | 第 3 代 | 第 4 代（支持 FP8） | |
| NVLink | 600 GB/s (NVLink 3) | 900 GB/s (NVLink 4) | +50% |
| L2 Cache | 40 MB | 50 MB | +25% |

**H100 关键新特性**：
1. **FP8 Tensor Core**：E4M3/E5M2 原生支持，训练+推理都可用
2. **TMA（Tensor Memory Accelerator）**：异步数据搬运单元，不占 SM 计算资源
3. **Thread Block Cluster**：多个 block 可组成 cluster 协作，shared memory 可跨 block 访问
4. **WGMMA 指令**：Warpgroup 级别的矩阵乘指令（4 个 warp 协作做大矩阵乘）
5. **异步执行引擎**：cp.async 更强，支持 bulk copy

**H 卡 vs L 卡（H100 vs L40/L4）**：

| 维度 | H100 (数据中心训练/推理) | L40 (推理/图形) | L4 (边缘推理) |
|------|------------------------|----------------|-------------|
| 定位 | 训练 + 推理旗舰 | 推理/视频/图形 | 低功耗推理 |
| FP8 | 支持 | 支持（Ada Lovelace） | 支持 |
| HBM | 80GB HBM3 | 48GB GDDR6X | 24GB GDDR6 |
| 带宽 | 3.35 TB/s | 864 GB/s | 300 GB/s |
| NVLink | 有 | 无 | 无 |
| 功耗 | 700W | 300W | 72W |
| 多卡互联 | NVLink + NVSwitch | PCIe only | PCIe only |

**选型指南**：
- 训练/大 batch 推理 → H100（需要 HBM + NVLink + 高算力）
- 中等规模推理/多模态 → L40（大显存 + Ada 架构 + 性价比）
- 边缘/嵌入式推理 → L4（低功耗小尺寸）

**考察点**：不只背参数表，要理解每个新特性对 AI workload 的实际影响（TMA → FlashAttention V3，FP8 → 训练加速）。

---

## Q：NHWC vs NCHW 数据布局 / 训练推理怎么选

> 来源：OPPO / AI Infra 实习二面

**普通回答**：NCHW 是 PyTorch 默认，NHWC 对 Tensor Core 更友好。

**更好的回答**：

**内存布局差异**：
```
NCHW: [batch][channel][height][width]
  → 同一 channel 的所有像素在内存中连续

NHWC: [batch][height][width][channel]
  → 同一像素的所有 channel 在内存中连续
```

**为什么 NHWC 对 GPU 更友好**：
- Tensor Core 做 Conv/GEMM 时，需要读取多个 channel 的数据做内积
- NHWC 布局下，一个像素的所有 channel 连续 → 向量化加载（float4/float8）
- NCHW 下读取同一像素的不同 channel → 跨步访问 → 无法合并

**各框架默认**：

| 框架 | 默认布局 | 原因 |
|------|---------|------|
| PyTorch | NCHW | 历史原因（CPU 上 NCHW 对 cache 友好） |
| TensorFlow | NHWC | GPU 优化优先 |
| TensorRT | NHWC (推理) | Tensor Core 最优 |
| cuDNN | 两者都支持，NHWC 更快 | |

**实践选择**：
- **训练**：PyTorch 用 `channels_last` memory format（`x.to(memory_format=torch.channels_last)`）可无缝切换到 NHWC
- **推理**：TensorRT/TVM 通常自动转为 NHWC 或更优布局
- **LLM**：主要是 GEMM 而非 Conv，布局影响不大（用 row-major/column-major 描述）

**考察点**：理解数据布局对 GPU 访存效率的影响，以及何时该切换布局。

---

## Q：C++ 四种 Cast 转换的区别

> 来源：蔚来 / AI Infra 实习

**普通回答**：有 static_cast、dynamic_cast、const_cast、reinterpret_cast。

**更好的回答**：

| Cast | 编译期/运行期 | 用途 | 安全性 |
|------|-------------|------|-------|
| static_cast | 编译期 | 已知安全的类型转换 | 不检查运行时类型 |
| dynamic_cast | 运行期 | 多态类型的安全向下转型 | 失败返回 nullptr/抛异常 |
| const_cast | 编译期 | 去除/添加 const/volatile | 修改真正的 const 是 UB |
| reinterpret_cast | 编译期 | 底层位模式重新解释 | 最不安全 |

**详细说明**：

```cpp
// static_cast: 编译器能验证合理性的转换
double d = 3.14;
int i = static_cast<int>(d);         // 截断
Base* b = static_cast<Base*>(derived); // 上行转换（安全）
Derived* d = static_cast<Derived*>(base); // 下行（不检查！）

// dynamic_cast: 运行时类型检查（需要虚函数）
Base* b = getObject();
Derived* d = dynamic_cast<Derived*>(b);
if (d) { /* 转换成功 */ }
// 内部通过 RTTI (typeid/vtable) 检查实际类型

// const_cast: 唯一能去掉 const 的方式
const int* cp = &x;
int* p = const_cast<int*>(cp);  // 去 const
// 如果原始对象本身是 const → 写入是 UB

// reinterpret_cast: 位级别重解释
float f = 1.0f;
int bits = *reinterpret_cast<int*>(&f);  // 查看 float 的二进制表示
// 常用于：指针↔整数、不同指针类型间转换
```

**父类转子类的安全性**：
- `static_cast<Derived*>(base)`：不检查，如果 base 实际不是 Derived → UB
- `dynamic_cast<Derived*>(base)`：运行时检查 vtable，不匹配返回 nullptr
- 性能：dynamic_cast 有开销（遍历继承链），hot path 避免使用

**AI Infra 中的使用**：
- `reinterpret_cast`：CUDA 中 float→half 的位操作
- `static_cast`：基类 Tensor* → 具体 CUDATensor*（已知类型时）
- `const_cast`：旧接口接受非 const 但不会修改时

**考察点**：四种 cast 各自的适用场景和风险，尤其是 static_cast 下行转换的不安全性。

---

## Q：Online Softmax 的实现原理

> 来源：飞腾 / AI Infra 实习 · 快手 / AI Infra 校招 · 科大讯飞 / AI Infra · 面经总结

**普通回答**：先减最大值再求 softmax，分块做。

**更好的回答**：

**标准 Softmax（三遍扫描）**：
```
Pass 1: m = max(x[0..N])           // 找最大值
Pass 2: d = Σ exp(x[i] - m)        // 求分母
Pass 3: y[i] = exp(x[i] - m) / d   // 归一化
```
- 必须先扫完所有元素才能开始计算 → 不能分块

**Online Softmax（一遍或两遍扫描）**：

核心思想：在遍历过程中维护 running max 和 running sum，每次看到新元素时修正之前的累加值。

```python
# 算法（单遍计算 denominator）:
m = -inf   # running max
d = 0      # running sum of exp

for i in range(N):
    m_new = max(m, x[i])
    d = d * exp(m - m_new) + exp(x[i] - m_new)  # 修正旧 sum + 加新项
    m = m_new

# 最终: y[i] = exp(x[i] - m) / d
```

**为什么能修正**：
```
旧的 d = Σ exp(x[j] - m_old), j < i
新的 d 应该 = Σ exp(x[j] - m_new), j <= i
     = Σ exp(x[j] - m_old) * exp(m_old - m_new) + exp(x[i] - m_new)
     = d_old * exp(m_old - m_new) + exp(x[i] - m_new)  ✓
```

**在 FlashAttention 中的应用**：

FlashAttention 将 Online Softmax 扩展到矩阵级别：
```
对 Q 的每个块:
    m = -inf, l = 0, O = 0
    for each K,V block:
        S_block = Q_block × K_block^T
        m_new = max(m, rowmax(S_block))
        l_new = l * exp(m - m_new) + rowsum(exp(S_block - m_new))
        O = O * (l * exp(m - m_new) / l_new) + exp(S_block - m_new) / l_new × V_block
        m, l = m_new, l_new
```
- 不需要存储完整 N×N 的 attention matrix
- 每个 block 在 SRAM 中完成 → 只读写最终结果到 HBM

**数值稳定性**：
- 始终减去 running max → exp 的输入 ≤ 0 → 不会溢出
- 乘以 exp(m_old - m_new) ≤ 1 → 不会放大误差

**考察点**：能写出 online softmax 的伪代码，理解"修正因子"exp(m_old - m_new) 的数学推导。

---

## Q：DMA 与 RDMA 的区别

> 来源：阶跃星辰 / AI Infra 实习 · 面经总结

**普通回答**：DMA 是直接内存访问，RDMA 是远程直接内存访问。

**更好的回答**：

**DMA（Direct Memory Access）**：
- CPU 发起传输请求后，DMA 控制器独立完成数据搬运（内存↔设备）
- CPU 在此期间可以做其他工作
- 应用：磁盘 IO、网络包收发、GPU H2D/D2H 传输
- `cudaMemcpyAsync` 就是 DMA 传输（需要 pinned memory）

**RDMA（Remote DMA）**：
- 跨网络节点的 DMA：直接从节点 A 的内存读写节点 B 的内存
- **Bypass kernel**：不经过远端的 CPU 和 OS 协议栈
- **Zero-copy**：数据直接 NIC→应用内存（不经过内核缓冲区）
- 延迟：~1-2 μs（vs TCP ~50-100 μs）

**RDMA 实现方式**：
- InfiniBand：专用 RDMA 网络（数据中心主流）
- RoCE（RDMA over Converged Ethernet）：在以太网上实现 RDMA
- iWARP：基于 TCP 的 RDMA（性能稍差但兼容性好）

**RDMA 操作类型**：
```
单边操作（Remote 端 CPU 不参与）:
  - RDMA Write: 直接写入远端内存
  - RDMA Read: 直接从远端内存读取

双边操作:
  - Send/Recv: 类似传统消息传递（远端 CPU 需要 post recv）
```

**为什么 cudaMemcpy 需要 Pinned Memory**：
- DMA 控制器使用物理地址
- Pageable memory 的物理地址可能随时变（OS page out）
- Pinned memory 锁定物理地址 → DMA 安全进行
- RDMA 同理：注册的内存必须是 pinned（`ibv_reg_mr`）

**AI Infra 中的应用**：
- NCCL All-reduce：节点内 NVLink（P2P DMA），节点间 InfiniBand（RDMA）
- GPUDirect RDMA：NIC 直接访问 GPU 显存（跳过 CPU 内存）
- DeepEP 的 All-to-all：利用 RDMA 高带宽做 Expert 间 token 传输

**考察点**：理解 DMA 和 RDMA 的层次关系，以及为什么 RDMA 能大幅降低延迟。

---

## Q：CUDA 实现矩阵转置与优化

> 来源：小鹏 / AI Infra 实习 · 面经总结

**普通回答**：每个线程读一个元素写到转置位置。

**更好的回答**：

**Naive 实现**：
```cuda
__global__ void transpose_naive(float* out, float* in, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        out[col * M + row] = in[row * N + col];
    }
}
```
问题：写入 `out[col * M + row]` → 同 warp 线程写入不连续地址 → 非合并写入

**优化：使用 Shared Memory**：
```cuda
__global__ void transpose_smem(float* out, float* in, int M, int N) {
    __shared__ float tile[TILE][TILE + 1];  // +1 避免 bank conflict!

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // 合并读取: in[y][x] → tile[threadIdx.y][threadIdx.x]
    if (x < N && y < M)
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();

    // 转置输出: 交换 block 坐标
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    // 合并写入: tile[threadIdx.x][threadIdx.y] → out[y][x]
    if (x < M && y < N)
        out[y * M + x] = tile[threadIdx.x][threadIdx.y];
}
```

**为什么 shared memory 能解决问题**：
1. 读 Global Memory：`in[y*N + x]` → 同 warp 线程 x 连续 → 合并读取 ✓
2. 写 Shared Memory：`tile[ty][tx]` → 按行写入 → 无 bank conflict
3. 读 Shared Memory：`tile[tx][ty]` → 按列读取 → 可能 bank conflict → +1 padding 解决
4. 写 Global Memory：`out[y*M + x]` → 因为交换了 block 坐标，x 连续 → 合并写入 ✓

**为什么 TILE+1**：
- Shared Memory 32 banks，stride=TILE=32 时按列读 → 32-way conflict
- Padding 后 stride=33 → 错开 bank → 无 conflict

**性能对比**：
- Naive：约 50% peak bandwidth（非合并写入的惩罚）
- Shared Memory + padding：约 90%+ peak bandwidth

**考察点**：完整写出优化版本，解释为什么+1 padding 和为什么需要交换 block 坐标。

---

## Q：CUDA Stream 的使用与 H2D/D2H 是否可以重叠

> 来源：OPPO / AI Infra 实习 · 快手 / AI Infra 校招

**普通回答**：不同 stream 可以并发，H2D 和 D2H 可以重叠。

**更好的回答**：

**CUDA Stream 基本规则**：
- 同一 stream 内的操作按顺序执行
- 不同 stream 的操作可以并发（如果硬件资源允许）
- 默认 stream (stream 0) 与所有其他 stream 同步

**三种操作可以重叠**：
```
1. H2D 传输（DMA Copy Engine 1）
2. Kernel 执行（SM）
3. D2H 传输（DMA Copy Engine 2）
```

**H2D 和 D2H 可以重叠吗？→ 可以！**
- 现代 GPU 有两个独立的 DMA 引擎（一个负责 H2D，一个负责 D2H）
- 条件：使用不同 stream + pinned memory

```cuda
cudaStream_t s1, s2, s3;
// Pipeline: 处理 batch 1 的同时传输 batch 2
cudaMemcpyAsync(d_in2, h_in2, size, H2D, s1);     // H2D batch 2
kernel<<<grid, block, 0, s2>>>(d_in1, d_out1);      // 计算 batch 1
cudaMemcpyAsync(h_out0, d_out0, size, D2H, s3);    // D2H batch 0
// 三者在时间上重叠执行!
```

**使用前提**：
1. 必须用 `cudaMemcpyAsync`（异步版本）
2. Host 端必须是 pinned memory
3. 不同操作在不同 stream 上
4. 操作之间无数据依赖

**同一 stream 内不能重叠**：
```cuda
// 这三个操作必须顺序执行（同一 stream）
cudaMemcpyAsync(d_in, h_in, size, H2D, stream);
kernel<<<..., stream>>>(d_in, d_out);
cudaMemcpyAsync(h_out, d_out, size, D2H, stream);
```

**Event 同步**：
```cuda
cudaEvent_t event;
cudaEventRecord(event, stream1);
cudaStreamWaitEvent(stream2, event);  // stream2 等 stream1 的 event 完成
```

**考察点**：理解 GPU 的双 DMA 引擎架构，以及 stream + event 实现细粒度同步。

---

## Q：如何确定最优线程数 / Block 大小

> 来源：OPPO / AI Infra 实习二面 · 快手 / AI Infra 校招

**普通回答**：看 occupancy，越高越好。

**更好的回答**：

**Block 大小选择的约束条件**：

1. **Warp 对齐**：block 大小必须是 32 的倍数（否则浪费 warp 中的线程）
2. **SM 资源限制**：
   - 最大线程数/SM（A100: 2048）
   - 最大 block 数/SM（A100: 32）
   - Register file（256KB/SM，每线程 255 个 register 上限）
   - Shared Memory（最大 164KB/SM）

3. **Occupancy 计算**：
```
active_warps = min(
    max_threads_per_SM / block_size × warps_per_block,
    max_blocks_per_SM × warps_per_block,
    register_限制,
    shared_memory_限制
)
occupancy = active_warps / max_warps_per_SM
```

**不是 occupancy 越高越好**：
- GEMM：低 occupancy（如 25%）但每线程用大量 register 做 tiling → 性能更好
- Memory-bound kernel：高 occupancy 帮助隐藏延迟 → 越高越好
- Compute-bound kernel：足够的 occupancy 即可，更多 register 更重要

**实践指南**：
- 起步：256 线程/block（8 warps，大多数 kernel 的安全选择）
- Memory-bound：512-1024 线程/block（最大化 occupancy）
- Compute-bound / register-heavy：128-256 线程/block（腾出 register）
- 用 `__launch_bounds__(maxThreads, minBlocks)` 提示编译器优化 register 分配

**Profiling 验证**：
- `ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active`
- 如果 stall 原因是 "not selected"（warp 太多竞争）→ 降 block 大小
- 如果 stall 原因是 "wait"（延迟没被隐藏）→ 增 block 大小

**考察点**：不是背公式，而是根据 kernel 特征（compute/memory bound）选择策略。

---

## Q：显存越界如何排查

> 来源：科大讯飞 / AI Infra 校招

**普通回答**：用 CUDA memcheck。

**更好的回答**：

**症状**：CUDA kernel 结果错误、随机崩溃、`illegal memory access`、结果不可复现

**排查工具和方法**：

1. **compute-sanitizer（推荐首选）**：
```bash
compute-sanitizer --tool memcheck ./my_program
# 报告越界访问的 kernel 名、行号、地址
```
- 检测：越界读写、未初始化读取、race condition
- 代价：10-100× 减速

2. **cuda-memcheck（旧版本）**：
```bash
cuda-memcheck ./my_program
```

3. **CUDA_LAUNCH_BLOCKING=1**：
```bash
CUDA_LAUNCH_BLOCKING=1 python train.py
# 让 kernel 同步执行 → 错误在发生时立即报出（而非延迟）
```

4. **assert 和 bounds check**：
```cuda
__global__ void kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    assert(idx < n);  // debug 模式下生效
    if (idx >= n) return;  // 安全退出
    data[idx] = ...;
}
```

5. **逐步缩小范围**：
- 先确定是哪个 kernel（二分注释法）
- 打印 grid/block 配置和 index 范围
- 检查 index 计算是否对 edge case 正确（最后一个 block 可能越界）

**常见原因**：
- Grid/Block 配置计算错误：忘记向上取整 `(n + BLOCK - 1) / BLOCK`
- Shared memory 索引溢出
- 异步 kernel 改了后续 kernel 依赖的数据（缺少同步）
- Host 端 free 了 GPU 正在使用的显存

**考察点**：知道 compute-sanitizer 是首选工具，能说出常见越界原因。

---

## Q：如何判断优化是否到瓶颈 / 还有多少空间

> 来源：快手校招 / AI Infra

**普通回答**：看 GPU 利用率是不是 100%。

**更好的回答**：

**Roofline Model 判断法**：

```
计算 Arithmetic Intensity (AI) = FLOPs / Bytes_accessed

if AI < GPU 的 ops:byte ratio (如 A100 = 156):
    → Memory-bound，上限 = 带宽 × AI
    → 优化方向：减少内存访问（fusion、量化、cache）
    
if AI > ops:byte ratio:
    → Compute-bound，上限 = 峰值算力
    → 优化方向：Tensor Core、更好的指令调度
```

**量化评估**：
- **Memory-bound kernel**：achieved_bandwidth / peak_bandwidth
  - > 80%：已接近瓶颈，空间不大
  - < 60%：访问模式有问题（非合并、bank conflict）
  
- **Compute-bound kernel**：achieved_FLOPS / peak_FLOPS
  - > 70%：已接近极限
  - < 50%：看 warp stall 原因

**系统级判断**：
- 端到端 throughput vs 理论上限的比值
- Pipeline bubble 率（PP 并行时）
- 通信/计算 overlap 率
- 各阶段（prefill/decode/communication）的时间占比

**工具**：
- Kernel 级：NSight Compute（roofline chart 直接显示）
- 系统级：NSight Systems（时间线看 gap 和 overlap）
- 框架级：PyTorch Profiler、DeepSpeed Flops Profiler

**已到瓶颈的信号**：
- Bandwidth 利用率 >85% 且无明显 bank conflict
- FLOPS 利用率 >75%
- Pipeline bubble <5%
- 通信完全被计算 overlap

**考察点**：能用 roofline + profiling 数据定量判断优化空间，而非凭感觉。
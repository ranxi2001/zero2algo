---
layout: default
title: PyTorch 性能优化机制
description: PyTorch 如何从 Python 层调度 C++ 算子、torch.compile、算子融合
parent: AI Infra
---

# PyTorch 性能优化机制

PyTorch 用户接口是 Python，但实际计算全部发生在 C++/CUDA 层。本文解析 PyTorch 如何解决 Python 慢的问题，以及现代优化手段。

---

## Q：PyTorch 的代码是用 Python 写的（慢），它是如何优化这个问题的

> 来源：抖音 / AI Infra 一面
**普通回答**：PyTorch 底层用 C++ 写的，Python 只是接口层。

**更好的回答**：

PyTorch 的设计哲学是"Python 做调度，C++/CUDA 做计算"。Python 慢的影响被控制在极小范围内：

**1. C++ 后端算子（ATen 库）**
- 所有 tensor 操作（matmul、conv、softmax…）的实际计算由 C++ 编写的算子完成
- Python 层只负责调用 dispatch（几微秒），实际计算（几毫秒到几秒）在 C++ 中
- 当算子计算时间 >> Python dispatch 开销时，Python 的慢可以忽略

**2. GPU 异步执行**
- Python 只是向 CUDA stream 提交 kernel，GPU 异步执行
- Python 发完指令就返回，不等 GPU 完成
- 只要 Python 提交速度 > GPU 消费速度（不成为瓶颈），就没有性能损失

**3. torch.compile（PyTorch 2.0+）**
- 将 Python 代码通过 TorchDynamo 捕获为计算图（FX Graph）
- 用 TorchInductor 后端编译为优化的 Triton/C++ kernel
- 消除了 Python 解释器的逐行开销，实现算子融合
- 相当于 JIT 编译，首次慢但后续调用走编译后的代码

**4. torch.jit（TorchScript，旧方案）**
- 将 Python 代码编译为 TorchScript IR，脱离 Python 解释器执行
- 已逐渐被 torch.compile 取代

**5. C++ Extension / Custom CUDA Kernel**
- 热点代码直接用 C++/CUDA 重写，通过 pybind11 暴露给 Python
- FlashAttention、各种 fused kernel 都是这个模式

**6. DataLoader 多进程**
- 数据预处理用多进程绕过 GIL
- `num_workers > 0` 时，数据加载在独立进程中

**Python 开销真正成为瓶颈的场景**：
- 极小的 tensor 操作（overhead > compute）→ 用 torch.compile 或合并操作
- 复杂的控制流（大量 if/for）→ torch.compile 的 graph break 会退化
- CPU-bound 的自定义 Python 逻辑 → 改用 C++ extension

**考察点**：理解 PyTorch 的分层架构和"Python overhead 被异步执行掩盖"这个核心洞察。

---

## Q：AI 框架前端中算子注册的机制 / 为什么通过添加宏定义即可完成算子注册

> 来源：三星 / AI Infra 实习一面

**普通回答**：用宏定义注册算子到全局表中。

**更好的回答**：

**算子注册的目的**：
- 框架需要维护一张"算子名 → 实现函数"的映射表
- 用户调 `torch.add(a, b)` 时，dispatch 系统查表找到对应设备（CPU/CUDA）的实现并执行

**PyTorch 的注册机制**：

```cpp
// 用户只需写一个宏：
TORCH_LIBRARY(mylib, m) {
    m.def("my_op(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(mylib, CUDA, m) {
    m.impl("my_op", my_cuda_implementation);
}
```

**为什么宏能完成注册（核心机制）**：

宏展开后生成一个 **全局静态变量**，其构造函数在 `main()` 之前执行：

```cpp
// 宏展开后大致等价于：
static auto __register_op_42 = []() {
    OpRegistry::getInstance().registerOp("my_op", &my_impl);
    return 0;
}();
// 或者用全局静态对象的构造函数
```

**原理**：
1. C++ 保证全局/静态对象在 `main()` 之前构造
2. 宏生成一个静态对象，构造函数里执行注册逻辑
3. 链接时只要 .o 文件被链入，注册就会发生
4. 这就是"自注册模式"（self-registration pattern）

**完整的 dispatch 流程**：
```
Python: torch.add(a, b)
  → C++: Dispatcher::call("aten::add", ...)
    → 根据 tensor 的 DispatchKey（CPU/CUDA/AutogradCUDA）查表
      → 找到注册的具体 kernel 执行
```

**模板函数的编译过程**（相关追问）：
- 模板在编译期实例化：编译器看到具体类型调用时生成对应的函数代码
- 模板定义必须在头文件中（编译器需要看到完整定义才能实例化）
- 隐式实例化 vs 显式实例化（`template class Foo<int>;`）

**考察点**：理解 C++ 的静态初始化机制如何实现"声明即注册"的设计模式。

---

## Q：torch.repeat 和 torch.expand 的区别

> 来源：面经总结

**普通回答**：expand 不拷贝数据，repeat 拷贝。

**更好的回答**：

```python
x = torch.tensor([[1, 2, 3]])  # shape: (1, 3)

# expand: 不拷贝数据，只改 stride（虚拟扩展）
y = x.expand(4, 3)  # shape: (4, 3), stride: (0, 1)
# 内存中仍只有 [1,2,3]，4 行共享同一数据
# y.storage().data_ptr() == x.storage().data_ptr()
# y 不是 contiguous（stride[0]=0 表示该维度"广播"）

# repeat: 真正拷贝数据
z = x.repeat(4, 1)  # shape: (4, 3), stride: (3, 1)
# 内存中有 [1,2,3,1,2,3,1,2,3,1,2,3]
# z.is_contiguous() == True
```

**关键区别**：

| 特性 | expand | repeat |
|------|--------|--------|
| 内存占用 | 不增加 | 按倍数增加 |
| 返回值 | view（共享数据） | 新 tensor（独立数据） |
| 可写性 | 写入会广播覆盖 | 安全独立写入 |
| 速度 | O(1)（只改 metadata） | O(n)（复制数据） |
| 参数含义 | 目标 shape | 每维重复次数 |

**tensor.view vs tensor.contiguous**：
```python
x = torch.randn(3, 4)
y = x.t()            # 转置 → stride 变了，不 contiguous
# y.view(12)         # ❌ RuntimeError: view requires contiguous
y.contiguous().view(12)  # ✓ 先拷贝为连续布局再 view
y.reshape(12)        # ✓ reshape = 能 view 就 view，不能就 copy
```

- **contiguous**：检查 stride 是否递减且紧密，不是则拷贝为标准布局
- **view**：只改 shape/stride，要求内存连续（零拷贝）
- **reshape**：view 的安全版本，不连续时自动 contiguous + view

**考察点**：理解 PyTorch 的 stride-based tensor 设计——expand 只改 stride 不拷数据。

---

## Q：Graph Fusion（算子融合）的原理与决策

> 来源：字节 / AI Infra 实习 · OPPO / AI Infra 实习 · 快手 / AI Infra 校招

**普通回答**：把多个小算子合成一个大 kernel 减少内存读写。

**更好的回答**：

**为什么要融合**：
```
未融合: x → kernel1 → 写HBM → 读HBM → kernel2 → 写HBM → 读HBM → kernel3
融合后: x → fused_kernel(1+2+3) → 写HBM
```
- 每次 kernel 之间需要将中间结果写回 HBM（因为 kernel 退出后 register/smem 丢失）
- 融合后中间结果留在 register/shared memory → 省掉大量 HBM 读写
- 同时减少 kernel launch overhead

**融合类型**：

| 类型 | 模式 | 例子 |
|------|------|------|
| Element-wise fusion | 逐元素操作链 | bias + relu + dropout |
| Reduce fusion | reduce 前后的 element-wise | sum + scale |
| Matmul + epilogue | GEMM + 后处理 | GEMM + bias + activation |
| 复杂 fusion | 自定义 pattern | FlashAttention (QKV+softmax+output) |

**什么时候适合融合**：
- 多个 memory-bound 小算子连续 → 融合收益大
- 中间 tensor 生命周期短（只被下一个算子用）→ 不需要物化到 HBM
- 算子间数据依赖简单（element-wise 或同维 reduce）

**什么时候不适合融合**：
- 融合后 register 压力过大 → occupancy 下降 → 可能更慢
- 算子本身是 compute-bound（如大 GEMM）→ 已经在充分利用计算，融合收益有限
- 不同 shape/layout 需要转换 → 融合反而复杂

**torch.compile 的自动融合**：
- TorchInductor 后端自动识别可融合 pattern
- 生成 Triton kernel 实现融合
- Horizontal fusion：独立的小算子合并为一个 kernel（并行执行）
- Vertical fusion：数据依赖链上的算子合并

**考察点**：判断融合收益需要考虑 memory-bound vs compute-bound 和 register 压力。

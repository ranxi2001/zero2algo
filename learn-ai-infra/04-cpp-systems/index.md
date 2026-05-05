---
layout: default
title: C++ 系统编程高频题
description: 内存管理方式、智能指针线程安全、RAII 等 AI Infra 岗 C++ 高频考点
parent: AI Infra
---

# C++ 系统编程高频题

AI Infra 岗位（推理框架、训练框架、CUDA 开发）对 C++ 底层有硬性要求。本文覆盖面试最常考的内存管理和智能指针问题。

---

## Q：C++ 申请内存的方式有哪些

> 来源：抖音 / AI Infra 一面 · 沐曦 / AI Infra 实习一面 · 文远知行 / AI Infra 一面 · 太初 / AI Infra 实习一面
**普通回答**：new、malloc。

**更好的回答**：

C++ 中申请内存的方式按层次从高到低：

**1. 栈内存（自动存储）**
```cpp
int arr[1024];  // 编译器自动管理，函数退出即释放
```
- 快速（只移动栈指针），但大小有限（通常 1-8 MB）

**2. new / delete（C++ 操作符）**
```cpp
int* p = new int(42);      // 分配 + 构造
int* arr = new int[100];   // 数组
delete p;
delete[] arr;
```
- 调用构造/析构函数
- 底层通常调 `operator new` → `malloc`
- 可以重载 `operator new` 自定义分配策略

**3. malloc / free（C 库函数）**
```cpp
void* p = malloc(sizeof(int) * 100);
free(p);
```
- 不调构造/析构，返回 `void*`
- 失败返回 NULL（new 抛异常）

**4. mmap（系统调用）**
```cpp
void* p = mmap(nullptr, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
```
- 直接向 OS 申请虚拟内存页
- 大块内存分配（glibc 的 malloc 对 >128KB 的请求内部就用 mmap）
- 适合内存映射文件、共享内存

**5. placement new（在已有内存上构造）**
```cpp
char buf[sizeof(Obj)];
Obj* p = new(buf) Obj();  // 不分配内存，只调构造
p->~Obj();                // 需手动析构
```
- 内存池、Arena Allocator 的核心手段

**6. 自定义分配器（Allocator / Memory Pool）**
- AI 框架常用：预分配大块显存/内存，内部管理分配释放
- PyTorch 的 CUDACachingAllocator、TensorFlow 的 BFC Allocator
- 减少 malloc/cudaMalloc 的开销和碎片

**AI Infra 场景联系**：
- GPU 显存：`cudaMalloc`、`cudaMallocAsync`（CUDA stream-ordered）
- 统一内存：`cudaMallocManaged`（Unified Memory）
- 锁页内存：`cudaHostAlloc`（加速 Host↔Device 传输）

**考察点**：不只是背 API，要理解从用户态到内核态的层次关系，以及 AI 框架为什么需要自定义内存管理。

---

## Q：共享指针（shared_ptr）是线程安全的吗

> 来源：抖音 / AI Infra 一面 · 沐曦 / AI Infra 实习一面 · 文远知行 / AI Infra 一面 · 太初 / AI Infra 实习一面
**普通回答**：引用计数是线程安全的，但指向的对象不是。

**更好的回答**：

`std::shared_ptr` 的线程安全性需要分三个层面理解：

**1. 引用计数（control block）— 线程安全**
- 引用计数的增减使用原子操作（`std::atomic`）
- 多个线程同时拷贝/销毁同一个 shared_ptr 的副本，引用计数不会出错

**2. 同一个 shared_ptr 对象的读写 — 不线程安全**
```cpp
std::shared_ptr<T> global_ptr = ...;

// Thread A
global_ptr = another_ptr;  // 写

// Thread B
auto local = global_ptr;   // 读

// ❌ 数据竞争！shared_ptr 内部有两个字段（指针 + control block 指针）
// 一个线程写的同时另一个读，可能读到不一致的状态
```

**3. 指向的对象（managed object）— 不线程安全**
```cpp
auto ptr = std::make_shared<std::vector<int>>();

// Thread A
ptr->push_back(1);  // 修改对象

// Thread B
ptr->push_back(2);  // 修改对象

// ❌ vector 本身不是线程安全的
```

**总结**：

| 操作 | 线程安全？ |
|------|-----------|
| 多线程各自持有副本，独立拷贝/析构 | 安全 |
| 多线程读写**同一个** shared_ptr 变量 | 不安全 |
| 多线程通过各自副本访问同一对象 | 取决于对象本身 |

**解决方案**：
- C++20：`std::atomic<std::shared_ptr<T>>`
- C++11/14/17：`std::atomic_load` / `std::atomic_store`（已 deprecated）
- 实践中：加 mutex，或使用 read-copy-update（RCU）模式

**AI Infra 场景联系**：
- 推理框架中模型权重用 shared_ptr 共享给多个 worker，权重本身只读所以安全
- 但如果涉及热更新模型（替换 shared_ptr 指向），就必须用 atomic 或锁

**考察点**：不能只回答"引用计数安全"就停，要区分三个层面，并给出实际解决方案。

---

## Q：new 与 malloc 的底层实现原理

> 来源：沐曦 / AI Infra 实习一面

**普通回答**：new 调 malloc 再调构造函数。

**更好的回答**：

**malloc 的底层实现（glibc ptmalloc2）**：

```
malloc(size)
  → size < 128KB: 从 arena 的 bins/freelist 分配（用户态）
  → size >= 128KB: 调 mmap 直接向内核申请
  → 小块不够时: 调 brk/sbrk 扩展堆顶（用户态→内核态切换）
```

关键数据结构：
- **Bins**：空闲链表，按大小分类（fast bins、small bins、large bins、unsorted bin）
- **Arena**：多线程各自有 arena，减少锁竞争
- **Chunk**：每个分配块带 header（记录大小、前后 chunk 状态），用于合并空闲块

**new 的底层实现**：

```cpp
// new expression: T* p = new T(args);
// 编译器展开为：
void* mem = operator new(sizeof(T));  // step 1: 分配内存
T* p = static_cast<T*>(mem);
p->T(args);                           // step 2: 调构造函数
```

`operator new` 默认实现：
```cpp
void* operator new(size_t size) {
    void* p = malloc(size);     // 底层调 malloc
    if (!p) throw std::bad_alloc();  // 失败抛异常（malloc 返回 NULL）
    return p;
}
```

**关键区别**：

| 特性 | malloc | new |
|------|--------|-----|
| 失败处理 | 返回 NULL | 抛 bad_alloc |
| 构造/析构 | 不调 | 调 |
| 可重载 | 不能 | 可以重载 operator new |
| 类型 | void* 需强转 | 返回正确类型 |
| 数组 | malloc(n*sizeof(T)) | new T[n] + delete[] |

**AI Infra 应用**：
- 重载 `operator new` 实现定制分配器（如 Arena Allocator、显存池）
- `aligned_alloc` / `posix_memalign` 用于 SIMD 对齐（AVX 需 32 byte 对齐）

**考察点**：从系统调用层面理解内存分配的层次，以及 new 作为 C++ 抽象在 malloc 之上做了什么。

---

## Q：C++ 面向对象编程的三大特性

> 来源：沐曦 / AI Infra 实习一面 · 太初 / AI Infra 实习一面

**普通回答**：封装、继承、多态。

**更好的回答**：

**1. 封装（Encapsulation）**：
- 将数据和操作绑定在类中，通过 public/protected/private 控制访问
- AI Infra 场景：`Tensor` 类封装数据指针、shape、dtype、device，外部不能直接操作裸指针

**2. 继承（Inheritance）**：
- 子类复用父类接口和实现
- AI 框架常用：`Module` 基类 → `Linear`、`Conv2d` 等子类
- 注意虚析构函数：基类指针删除派生对象需要 `virtual ~Base()`

**3. 多态（Polymorphism）**：
- **编译期多态**：函数重载、模板（CRTP）
- **运行期多态**：虚函数 + 基类指针/引用
- AI 框架 dispatch 机制：通过虚函数表分发到不同设备（CPU/CUDA/XPU）的实现

**虚函数底层实现**：
- 每个含虚函数的类有一个 vtable（虚函数表），存储函数指针
- 每个对象有一个 vptr 指向其类的 vtable（占 8 bytes）
- 虚函数调用 = 读 vptr → 查 vtable → 间接跳转（比直接调用多一次内存访问）
- `final` 关键字可让编译器去虚拟化（devirtualize），消除间接调用开销

**构造函数中调用虚函数**（高频追问）：
- 构造函数中虚函数不会多态分发——调用的是当前正在构造的类的版本
- 原因：基类构造时，派生类尚未初始化，vptr 指向基类 vtable

**考察点**：不只背概念，要能联系到 AI 框架中的实际设计和底层机制。

---

## Q：std::map 与 std::unordered_map 的底层实现原理对比

> 来源：沐曦 / AI Infra 实习一面

**普通回答**：map 是红黑树，unordered_map 是哈希表。

**更好的回答**：

| 特性 | std::map | std::unordered_map |
|------|----------|-------------------|
| 底层 | 红黑树（自平衡 BST） | 哈希表（桶 + 链表/开放寻址） |
| 有序性 | key 有序（中序遍历） | 无序 |
| 查找/插入 | O(log n) | 平均 O(1)，最差 O(n) |
| 迭代器失效 | 插入不失效，删除仅当前 | rehash 时全部失效 |
| 内存布局 | 节点分散，cache 不友好 | 桶连续，单桶内链式 |
| key 要求 | 需要 `operator<` 或自定义比较器 | 需要 `std::hash` + `operator==` |

**选择指南**：
- 需要有序遍历 → map
- 纯查找、不关心顺序 → unordered_map（快 5-10×）
- key 是自定义类型且难写 hash → map 更方便
- 数据量小（<100）→ 性能差异不大，随意

**unordered_map 的陷阱**：
- 负载因子 > 1 时触发 rehash → 所有迭代器失效 + O(n) 重建
- 哈希冲突严重时退化为链表 O(n)
- 可以 `reserve(n)` 预分配避免频繁 rehash

**AI Infra 中的使用**：
- 算子注册表：`unordered_map<string, OpFactory>`
- 图节点邻接关系：`map<NodeId, vector<NodeId>>`（需要确定性遍历顺序保证可复现）
- Tensor 缓存：`unordered_map<shape+dtype, Tensor>`

**考察点**：不只说数据结构名字，要理解性能特征和选型依据。

---

## Q：C++ 中 lambda 表达式的语法与使用方式

> 来源：沐曦 / AI Infra 实习一面

**普通回答**：`[capture](params) { body }`

**更好的回答**：

**语法**：
```cpp
[capture_list](parameter_list) mutable -> return_type { body }
```

**捕获方式**：
```cpp
int x = 10;
auto f1 = [x]() { return x; };        // 值捕获（拷贝，lambda 内不可修改除非 mutable）
auto f2 = [&x]() { x++; };            // 引用捕获
auto f3 = [=]() { return x; };        // 隐式值捕获所有
auto f4 = [&]() { x++; };             // 隐式引用捕获所有
auto f5 = [x = std::move(obj)]() {};  // C++14 init capture（移动捕获）
```

**底层实现**：
- 编译器将 lambda 转为一个匿名类（闭包类型）
- 捕获的变量成为该类的成员
- `operator()` 就是 lambda body
- 无捕获的 lambda 可隐式转为函数指针

```cpp
// auto f = [x](int y) { return x + y; };
// 等价于编译器生成：
struct __lambda_42 {
    int x;  // 捕获的变量
    int operator()(int y) const { return x + y; }
};
```

**常见使用场景**：
```cpp
// STL 算法
std::sort(v.begin(), v.end(), [](int a, int b) { return a > b; });

// 回调
threadPool.submit([&tensor]() { tensor.compute(); });

// 立即调用（IIFE）
const auto result = [&]() {
    // 复杂初始化逻辑
    return computeResult();
}();
```

**注意事项**：
- 引用捕获注意生命周期（lambda 比被捕获对象活得长 → 悬垂引用）
- `mutable` 允许修改值捕获的副本
- 泛型 lambda（C++14）：`[](auto x) { return x * 2; }`

**考察点**：理解 lambda 的本质是匿名仿函数，捕获就是成员变量。

---

## Q：Pinned Memory（锁页内存）

> 来源：文远知行 / AI Infra 一面

**普通回答**：Pinned memory 是不会被 swap 的内存，GPU 传输更快。

**更好的回答**：

**什么是 Pinned Memory**：
- 普通内存（pageable）可以被 OS 换出到磁盘（page out）
- Pinned memory（page-locked）被锁定在物理内存中，OS 不会将其 swap out
- 分配方式：`cudaHostAlloc()` 或 `cudaMallocHost()`

**为什么 Host↔Device 传输需要它**：
```
普通 DMA 传输流程：
  pageable memory → OS 分配临时 pinned buffer → DMA 到 GPU
                    ↑ 额外拷贝！

Pinned memory 传输：
  pinned memory → DMA 直接到 GPU
                  ↑ 零拷贝！
```
- 原因：DMA 控制器需要物理地址不变，pageable 内存的物理地址可能随时变（被 swap/重映射）
- 所以 CUDA driver 必须先拷贝到 pinned staging buffer，再启动 DMA

**性能提升**：
- Host→Device 带宽：pageable ~12 GB/s → pinned ~25 GB/s（PCIe Gen4 x16 上限 ~32 GB/s）
- 约 2× 提升
- 可以与 CUDA stream overlap（异步传输）

**使用方式**：
```cpp
float* h_data;
cudaMallocHost(&h_data, size);          // 分配 pinned memory
cudaMemcpyAsync(d_data, h_data, size,   // 异步传输（需要 pinned）
                cudaMemcpyHostToDevice, stream);
cudaFreeHost(h_data);                   // 释放
```

**注意事项**：
- Pinned memory 不能被 swap → 过多分配会挤压其他进程的可用内存
- 分配/释放比普通 malloc 慢（涉及内核 mlock 系统调用）
- PyTorch 中：`tensor.pin_memory()` 或 DataLoader 设置 `pin_memory=True`

**考察点**：理解 DMA 传输对物理地址固定的要求，以及 pinned memory 的收益和代价。

---

## Q：智能指针的实现原理

> 来源：太初 / AI Infra 实习一面

**普通回答**：用引用计数管理对象生命周期。

**更好的回答**：

**三种智能指针的实现**：

**1. `unique_ptr`**：
```cpp
template<typename T, typename Deleter = std::default_delete<T>>
class unique_ptr {
    T* ptr;          // 唯一拥有的裸指针
    Deleter deleter; // 析构时调用（默认 delete）
public:
    ~unique_ptr() { if (ptr) deleter(ptr); }
    // 禁止拷贝，只允许移动
    unique_ptr(const unique_ptr&) = delete;
    unique_ptr(unique_ptr&& other) : ptr(other.ptr) { other.ptr = nullptr; }
};
```
- 零开销抽象：和裸指针大小相同（Deleter 为空类时 EBO 优化）
- 移动语义转移所有权

**2. `shared_ptr`**：
```cpp
// 简化版核心结构
struct ControlBlock {
    std::atomic<int> strong_count;  // shared_ptr 计数
    std::atomic<int> weak_count;    // weak_ptr 计数
    void* ptr;                      // 管理的对象
    Deleter deleter;
};

class shared_ptr<T> {
    T* ptr;                  // 对象指针（方便直接解引用）
    ControlBlock* ctrl;      // 控制块指针
};
```
- 拷贝 → `strong_count++`（原子操作）
- 析构 → `strong_count--`，降为 0 时 delete 对象
- `make_shared` 优化：对象和 ControlBlock 一次分配（减少内存碎片 + cache 友好）

**3. `weak_ptr`**：
- 不增加 strong_count，只增加 weak_count
- `lock()` 尝试获取 shared_ptr：检查 strong_count > 0 则返回 shared_ptr，否则返回空
- 用途：打破循环引用、缓存（观察但不拥有）

**智能指针能否管理连续内存**：
```cpp
// unique_ptr 支持数组特化
std::unique_ptr<int[]> arr(new int[100]);  // 自动调 delete[]

// shared_ptr C++17 起支持数组
std::shared_ptr<int[]> arr(new int[100]);  // C++17

// C++11/14 的 shared_ptr 需要自定义 deleter
std::shared_ptr<int> arr(new int[100], std::default_delete<int[]>());
```

**考察点**：不只知道"RAII + 引用计数"，要能说清楚控制块结构、原子操作、以及数组管理。

---

## Q：STL 迭代器在遍历中删除元素的正确方式

> 来源：三星 / AI Infra 实习一面

**普通回答**：用 erase 返回下一个迭代器。

**更好的回答**：

**错误写法**（迭代器失效）：
```cpp
for (auto it = vec.begin(); it != vec.end(); ++it) {
    if (*it == target) {
        vec.erase(it);  // ❌ erase 后 it 失效，++it 是 UB
    }
}
```

**正确写法**：

```cpp
// 方法 1：使用 erase 返回值（序列容器）
for (auto it = vec.begin(); it != vec.end(); ) {
    if (*it == target) {
        it = vec.erase(it);  // erase 返回下一个有效迭代器
    } else {
        ++it;
    }
}

// 方法 2：erase-remove idiom（最高效，一次 O(n)）
vec.erase(std::remove_if(vec.begin(), vec.end(),
    [](int x) { return x == target; }), vec.end());

// C++20：std::erase_if
std::erase_if(vec, [](int x) { return x == target; });
```

**不同容器的迭代器失效规则**：

| 容器 | erase 后失效范围 |
|------|-----------------|
| vector/deque | 被删元素及之后所有迭代器失效 |
| list | 仅被删元素的迭代器失效 |
| map/set | 仅被删元素的迭代器失效 |
| unordered_map | 仅被删元素失效（rehash 时全部失效） |

**关联容器（map/set）的删除**：
```cpp
for (auto it = m.begin(); it != m.end(); ) {
    if (it->second == 0) {
        it = m.erase(it);  // C++11 起 map::erase 也返回迭代器
    } else {
        ++it;
    }
}
```

**考察点**：理解不同容器的迭代器失效规则，选择正确的删除模式。

---

## Q：单例模式的实现方式

> 来源：三星 / AI Infra 实习一面

**普通回答**：私有构造 + 静态方法返回实例。

**更好的回答**：

**方法 1：Meyer's Singleton（C++11 推荐）**：
```cpp
class Singleton {
public:
    static Singleton& getInstance() {
        static Singleton instance;  // C++11 保证线程安全
        return instance;
    }
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
private:
    Singleton() = default;
};
```
- C++11 标准保证局部 static 变量初始化是线程安全的（编译器用 guard variable + mutex）
- 延迟初始化（首次调用时构造）
- 最简洁、最推荐

**方法 2：双重检查锁（DCLP，C++11 之前常用）**：
```cpp
class Singleton {
    static std::atomic<Singleton*> instance;
    static std::mutex mtx;
public:
    static Singleton* getInstance() {
        auto* tmp = instance.load(std::memory_order_acquire);
        if (!tmp) {
            std::lock_guard<std::mutex> lock(mtx);
            tmp = instance.load(std::memory_order_relaxed);
            if (!tmp) {
                tmp = new Singleton();
                instance.store(tmp, std::memory_order_release);
            }
        }
        return tmp;
    }
};
```
- C++11 前需要这种写法保证线程安全
- 注意 memory order：acquire/release 防止指令重排

**AI Infra 中单例的应用**：
- 全局设备管理器（`DeviceManager::getInstance()`）
- 算子注册表（`OpRegistry::getInstance()`）
- 内存池分配器

**考察点**：Meyer's Singleton 是标准答案，但要能解释 C++11 如何保证线程安全。

---

## Q：虚函数表中函数顺序是怎样的

> 来源：京东 / AI Infra 一面

**普通回答**：按声明顺序排列。

**更好的回答**：

**vtable 中的函数顺序（以 GCC/Clang Itanium ABI 为例）**：

```cpp
class Base {
    virtual void foo();     // vtable[0]
    virtual void bar();     // vtable[1]
    virtual void baz();     // vtable[2]
};

class Derived : public Base {
    void bar() override;    // 覆盖 vtable[1]
    virtual void qux();     // vtable[3] (新增虚函数追加在末尾)
};
```

**规则**：
1. **基类虚函数按声明顺序**排列
2. **派生类 override 的函数**替换基类对应位置（保持 index 不变）
3. **派生类新增虚函数**追加到 vtable 末尾
4. **多继承**时每个基类有自己的 vtable（或 vtable 组），主基类共享派生类 vtable

**为什么顺序重要**：
- 虚函数调用编译为 `vptr[offset](this)`
- offset 在编译期确定（根据声明顺序）
- 如果基类修改了虚函数声明顺序 → ABI 不兼容 → 需要全部重编译

**MSVC 差异**：
- MSVC 的 vtable layout 与 Itanium ABI 略有不同
- 但同样遵循"声明顺序 + override 不改 index"的基本规则

**考察点**：不只说"按声明顺序"，要能解释 override 和新增虚函数的处理方式。

---

## Q：右值引用和移动语义

> 来源：网易 / AI Infra 校招

**普通回答**：右值引用用 && 表示，可以"偷"资源避免拷贝。

**更好的回答**：

**核心问题**：
```cpp
vector<string> create() {
    vector<string> v = {"hello", "world"};
    return v;  // 拷贝？移动？
}
vector<string> result = create();  // 没有移动语义时：深拷贝所有 string
```

**右值引用（&&）**：
- 左值：有名字、有地址、可以取地址的表达式
- 右值：临时对象、字面值、将亡值（即将销毁）
- `T&&` 绑定到右值，表示"这个对象即将销毁，可以偷它的资源"

**移动语义**：
```cpp
class Vector {
    int* data;
    size_t size;
public:
    // 移动构造：偷资源，O(1)
    Vector(Vector&& other) noexcept
        : data(other.data), size(other.size) {
        other.data = nullptr;  // 源对象置空（合法但不可用）
        other.size = 0;
    }
    // 拷贝构造：深拷贝，O(n)
    Vector(const Vector& other) : size(other.size) {
        data = new int[size];
        memcpy(data, other.data, size * sizeof(int));
    }
};
```

**std::move**：
- 不移动任何东西！只是将左值强转为右值引用（xvalue）
- 让编译器选择移动构造/移动赋值而非拷贝

**push_back 中的应用**：
```cpp
vector<string> v;
string s = "hello";
v.push_back(s);            // 拷贝 s 到 vector 内部
v.push_back(std::move(s)); // 移动 s（之后 s 为空）
v.emplace_back("hello");   // 直接在 vector 内部构造，零拷贝
```

**完美转发（Perfect Forwarding）**：
```cpp
template<typename T>
void wrapper(T&& arg) {  // 万能引用（不是右值引用！）
    inner(std::forward<T>(arg));  // 保持原始的左/右值属性
}
```

**AI Infra 场景**：
- Tensor 的 move：只转移 data pointer 和 metadata，不拷贝几 GB 的数据
- `std::unique_ptr` 只能 move 不能 copy → 所有权转移

**考察点**：区分右值引用、std::move、完美转发，理解移动语义的性能收益。

---

## Q：内存对齐规则与 sizeof 计算

> 来源：网易 / AI Infra 校招

**普通回答**：成员按最大对齐数对齐。

**更好的回答**：

**对齐规则（x86-64, GCC）**：
1. 每个成员的偏移量必须是其自身对齐值的倍数
2. 结构体总大小必须是最大对齐值的倍数（尾部 padding）
3. 基本类型对齐值 = 自身大小（char=1, short=2, int=4, double=8, pointer=8）

**示例**：
```cpp
struct A {
    char a;     // offset 0, size 1
    // padding 7 bytes (下一个 double 需要 8 对齐)
    double b;   // offset 8, size 8
    int c;      // offset 16, size 4
    // padding 4 bytes (总大小需要是 8 的倍数)
};
// sizeof(A) = 24, alignof(A) = 8

struct B {
    double b;   // offset 0, size 8
    int c;      // offset 8, size 4
    char a;     // offset 12, size 1
    // padding 3 bytes
};
// sizeof(B) = 16, alignof(B) = 8
```

**调整成员顺序**可以节省空间（B 比 A 小 8 bytes）

**为什么需要对齐**：
- CPU 从内存读数据以字（word，通常 8 bytes）为单位
- 未对齐的访问可能需要两次内存读取 + 拼接 → 性能下降
- 某些架构（ARM）未对齐访问直接 fault
- SIMD 指令（AVX）要求 32/64 byte 对齐

**`#pragma pack(n)` / `__attribute__((packed))`**：
- 强制对齐为 n → 减少 padding 但牺牲性能
- 网络协议 struct 常用（需要精确控制布局）

**AI Infra 场景**：
- CUDA `__align__(16)` 保证 128-bit 向量化加载
- Tensor metadata struct 的 padding 影响 cache 效率

**考察点**：能手算给定 struct 的 sizeof，并解释为什么成员顺序影响大小。

---

## Q：进程与线程的核心区别

> 来源：阶跃星辰 / AI Infra 实习 · 蔚来 / AI Infra · 荣耀 / AI Infra 校招 · 飞腾 / AI Infra 实习 · 网易 / AI Infra 校招

**普通回答**：进程有独立地址空间，线程共享地址空间。

**更好的回答**：

| 维度 | 进程 (Process) | 线程 (Thread) |
|------|---------------|---------------|
| 地址空间 | 独立（互相隔离） | 共享（同进程内） |
| 创建开销 | 重（fork 需要复制页表/文件描述符） | 轻（只需新栈 + TLS） |
| 切换开销 | 重（切换页表 + TLB flush） | 轻（切换栈指针 + 寄存器） |
| 通信方式 | IPC（管道/共享内存/socket/消息队列） | 直接读写共享变量 |
| 容错性 | 一个崩不影响其他 | 一个崩全进程挂 |
| 同步 | 进程间锁（flock/信号量） | 互斥锁/条件变量/原子操作 |

**Linux 实现细节**：
- Linux 中进程和线程都是 `task_struct`，通过 `clone()` 的 flags 区分
- `CLONE_VM`：共享地址空间 → 线程
- `CLONE_FILES`：共享文件描述符表
- 内核不区分"进程/线程"，统称为 task

**AI Infra 场景**：
- **PyTorch DataLoader**：多进程（`num_workers`）绕过 GIL，各进程独立加载数据
- **推理服务**：多进程做 Tensor Parallel（各进程管一张 GPU）
- **CUDA 多线程**：多 CPU 线程可共享 GPU context，各用不同 stream

**考察点**：不只背对比表，要能联系到 Linux 内核实现和实际应用场景。

---

## Q：Python GIL（全局解释器锁）

> 来源：字节 / AI Infra 实习

**普通回答**：GIL 让 Python 多线程不能真正并行。

**更好的回答**：

**什么是 GIL**：
- CPython 解释器的全局互斥锁
- 同一时刻只有一个线程能执行 Python 字节码
- 原因：CPython 的引用计数（`ob_refcnt`）不是线程安全的，GIL 保护它

**影响**：
- CPU-bound 多线程：无法利用多核（线程交替执行，不并行）
- IO-bound 多线程：可以并发（线程在等 IO 时释放 GIL）
- C 扩展代码：可以手动释放 GIL（`Py_BEGIN_ALLOW_THREADS`）

**PyTorch/CUDA 为什么不受 GIL 影响**：
1. PyTorch 的底层计算在 C++ 中执行，进入 ATen 算子时释放 GIL
2. CUDA kernel 提交是异步的——Python 只是发 launch 命令
3. 真正的计算在 GPU 上，完全不涉及 Python 解释器
4. 多 GPU 训练用多进程（`torch.distributed`），每个进程独立的 GIL

**Python 多进程通信（IPC）**：
- `multiprocessing.Queue`：基于 pipe + pickle
- `multiprocessing.shared_memory`：共享内存段
- `torch.multiprocessing`：可以共享 CUDA tensor（通过 shared memory handle）
- `torch.distributed`：NCCL/Gloo 后端，直接 GPU-GPU 通信

**考察点**：理解 GIL 的存在原因和 PyTorch 为什么能绕过它。

---

## Q：深拷贝与浅拷贝

> 来源：网易 / AI Infra 校招

**普通回答**：浅拷贝只拷贝指针，深拷贝拷贝整个对象。

**更好的回答**：

**浅拷贝（Shallow Copy）**：
```cpp
class Buffer {
    int* data;
    size_t size;
public:
    // 默认拷贝构造 = 浅拷贝
    // Buffer b2(b1); → b2.data == b1.data (同一块内存!)
};
// 问题: b1 析构释放 data → b2.data 悬垂 → double free
```

**深拷贝（Deep Copy）**：
```cpp
Buffer(const Buffer& other) : size(other.size) {
    data = new int[size];
    memcpy(data, other.data, size * sizeof(int));  // 拷贝内容
}
```

**何时必须深拷贝**：
- 类中有裸指针管理的堆内存
- 不希望共享底层数据的场景
- Rule of Three/Five：如果需要自定义析构 → 几乎一定需要自定义拷贝构造和赋值

**现代 C++ 的解法**：
- `std::unique_ptr`：禁止拷贝，强制移动 → 不会意外浅拷贝
- `std::shared_ptr`：共享所有权（引用计数）→ 有意识的浅拷贝
- `std::vector`：拷贝构造自动深拷贝内部数据

**AI Infra 中的 Tensor 拷贝**：
- PyTorch `tensor.clone()`：深拷贝数据
- `tensor2 = tensor1`：浅拷贝（共享 Storage）
- `tensor.detach()`：共享数据但断开梯度图

**考察点**：不只说概念，要能指出默认拷贝的危险和现代 C++ 的正确做法。

---

## Q：虚拟内存解决了什么问题 / OS 内存管理

> 来源：字节 / AI Infra 实习 · 荣耀 / AI Infra 校招

**普通回答**：虚拟内存让每个进程以为自己有完整的地址空间。

**更好的回答**：

**虚拟内存解决的三个核心问题**：

1. **隔离性**：每个进程有独立的虚拟地址空间 → 进程 A 不能访问进程 B 的内存
2. **大于物理内存的使用**：虚拟空间可以比物理内存大（通过 swap 到磁盘）
3. **简化链接和加载**：每个进程从地址 0 开始编址，链接器不需要知道实际物理位置

**实现机制**：
- 页表（Page Table）：虚拟页号 → 物理帧号 的映射
- TLB（Translation Lookaside Buffer）：页表缓存，加速翻译
- 缺页中断（Page Fault）：访问未映射页 → 触发中断 → OS 分配物理页 or swap in

**内存不足时 OS 的处理**（缺页中断流程）：
1. 检查虚拟地址是否合法（segfault if not）
2. 如果物理内存有空闲帧 → 直接分配
3. 如果不足 → LRU 页面置换算法选择 victim 页
4. 如果 victim 是 dirty → 写回 swap 分区
5. 将需要的页从 swap/文件 读入物理帧
6. 更新页表，重新执行触发缺页的指令
7. 极端情况 → OOM Killer 杀掉占用最多内存的进程

**用户态 vs 内核态**：
- 用户态：应用程序代码，只能访问自己的虚拟地址空间
- 内核态：OS 内核，可以访问所有物理内存和硬件
- 切换：系统调用（syscall）、中断、异常

**考察点**：虚拟内存不只是"让内存看起来更大"，核心是隔离和抽象。

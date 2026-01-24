## 特性

### 丰富的算子库

FlagGems 拥有大量与 PyTorch 兼容的算子。
算子将根据 [算子列表](./operators.md) 进行实现。

### 针对选定算子的手工优化性能

下图展示了 FlagGems 在 eager 模式下与 PyTorch ATen 库相比的加速比。
加速比是通过平均每个形状上的加速比计算得出的，代表了算子的整体性能。

![算子加速比](./assets/speedup-20251225.png)

### 支持 Eager 模式，独立于 `torch.compile`

> 待补充 (TBD)

### 自动化代码生成

FlagGems 提供了一种自动化代码生成机制，使开发者能够轻松生成逐元素（pointwise）和融合（fused）算子。
该自动生成系统支持多种需求，包括标准逐元素计算、非张量参数以及指定输出类型。
详情请参考 [pointwise_dynamic](./pointwise_dynamic.md) 文档。

### 函数级 Kernel 调度

FlagGems 引入了 `LibEntry`，它可以独立管理 kernel 缓存，并绕过 `Autotuner`、`Heuristics` 和 `JitFunction` 的运行时。要使用此特性，只需使用 LibEntry 装饰 Triton kernel。

`LibEntry` 还支持直接包装 `Autotuner`、`Heuristics` 和 `JitFunction`，同时保留完整的调优功能。
然而，它避免了嵌套的运行时类型调用，消除了冗余的参数处理。
这意味着不需要绑定或类型包装，从而简化了缓存键格式并减少了不必要的键计算。

### 适用于多种平台的通用接口

<div id="platforms-supported"></div>

FlagGems 支持广泛的硬件平台，并已在不同的硬件配置上进行了广泛测试。

目前支持的平台包括：

| 厂商       | 状态           | float16 | float32 | bfloat16 |
| ---------- | -------------- | ------- | ------- | -------- |
| AIPU       | ✅ （部分支持） | ✅       | ✅       | ✅        |
| ARM(CPU)   | 🚧              |         |         |          |
| Ascend     | ✅ （部分支持） | ✅       | ✅       | ✅        |
| Cambricon  | ✅              | ✅       | ✅       | ✅        |
| Hygon      | ✅              | ✅       | ✅       | ✅        |
| Iluvatar   | ✅              | ✅       | ✅       | ✅        |
| Kunlunxin  | ✅              | ✅       | ✅       | ✅        |
| MetaX      | ✅              | ✅       | ✅       | ✅        |
| Mthreads   | ✅              | ✅       | ✅       | ✅        |
| NVIDIA     | ✅              | ✅       | ✅       | ✅        |
| TsingMicro | 🚧              |         |         |          |


### 后端支持

FlagGems 支持 10 多个后端。

### C++ Triton 函数分发器

C++ Triton 函数分发器正在开发中。

### C++ 运行时

FlagGems 既可以作为纯 Python 包安装，也可以作为带有 C++ 扩展的包安装。
C++ 运行时旨在解决 Python 运行时的开销并提高端到端性能。

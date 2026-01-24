![img_v3_02gp_8115f603-cc89-4e96-ae9d-f01b4fef796g](https://github.com/user-attachments/assets/97950fc6-62bb-4b6a-b8d5-5751c14492fa)

## 项目简介

FlagGems 是一个基于 [Triton](https://github.com/openai/triton) 语言实现的高性能通用算子库。
它旨在提供一系列算子内核（kernel）函数，以加速大语言模型（LLM）的训练和推理。

通过注册到 PyTorch 的 ATen 后端，FlagGems 实现了无缝过渡，
允许用户在不修改模型代码的情况下直接切换到 Triton 算子库。
用户可以沿用原有的 ATen 后端使用习惯，同时获得显著的性能提升。
Triton 语言在可读性、易用性上具有优势，且能提供媲美 CUDA 的性能。
这种便利性使得开发者能够以极低的学习成本参与到 FlagGems 的开发中。

## 特性

### 多后端硬件支持

FlagGems 支持广泛的硬件平台，并已在多种硬件配置上进行了充分验证。

### 自动代码生成 (Codegen)

FlagGems 提供了一套自动代码生成机制，使开发者能够轻松生成逐元素（pointwise）和融合（fused）算子。
该系统支持多种开发需求，包括标准元素级计算、非张量参数处理以及输出类型指定。
更多详细信息，请参考 [pointwise_dynamic](pointwise_dynamic.md)。

### LibEntry

FlagGems 引入了 `LibEntry` 机制，用于独立管理内核缓存，并绕过 `Autotuner`、`Heuristics` 和 `JitFunction` 的运行时开销。
使用时，只需使用 `LibEntry` 装饰 Triton 内核即可。

`LibEntry` 还支持直接包装 `Autotuner`、`Heuristics` 和 `JitFunction`，在保留完整调优功能的同时，
避免了嵌套的运行时类型调用，消除了冗余的参数处理。
这意味着无需进行繁琐的绑定或类型封装，从而简化了缓存键（cache key）格式并减少了不必要的计算开销。

### C++ 运行时

FlagGems 支持作为纯 Python 包安装，也支持包含 C++ 扩展的安装方式。
C++ 运行时旨在进一步降低 Python 运行时的开销，提升端到端性能。

## 更新日志

<!--TODO(Qiming) Drop this section-->

### v1.0

- 支持 BLAS 算子：addmm, bmm, mm
- 支持逐元素（pointwise）算子：abs, add, div, dropout, exp, gelu, mul, pow, reciprocal, relu, rsqrt, silu, sub, triu
- 支持归约（reduction）算子：cumsum, layernorm, mean, softmax

### v2.0

- 支持 BLAS 算子：mv, outer
- 支持逐元素（pointwise）算子：bitwise_and, bitwise_not, bitwise_or, cos, clamp, eq, ge, gt, isinf, isnan, le, lt, ne, neg, or, sin, tanh, sigmoid
- 支持归约（reduction）算子：all, any, amax, argmax, max, min, prod, sum, var_mean, vector_norm, cross_entropy_loss, group_norm, log_softmax, rms_norm
- 支持融合（fused）算子：fused_add_rms_norm, skip_layer_norm, gelu_and_mul, silu_and_mul, apply_rotary_position_embedding

### v2.1

- 支持 Tensor 操作算子：where, arange, repeat, masked_fill, tile, unique, index_select, masked_select, ones, ones_like, zeros, zeros_like, full, full_like, flip, pad
- 支持神经网络算子：embedding
- 支持基础数学算子：allclose, isclose, isfinite, floor_divide, trunc_divide, maximum, minimum
- 支持随机分布算子：normal, uniform\_, exponential\_, multinomial, nonzero, topk, rand, randn, rand_like, randn_like
- 支持科学计算算子：erf, resolve_conj, resolve_neg

## 快速入门

有关 FlagGems 的安装和使用，请参考 [快速入门](./getting-started.md) 文档。

## 支持的算子

算子将根据 [算子列表](./operators.md) 进行实现。

## 支持的模型

- Bert-base-uncased
- Llama-2-7b
- Llava-1.5-7b

## 支持的平台

|    平台    | float16 | float32 | bfloat16 |
| :--------: | :-----: | :-----: | :------: |
| Nvidia GPU |    ✓    |    ✓    |    ✓     |

## 性能表现

下图展示了 FlagGems 在 Eager 模式下相比 PyTorch 原生 ATen 库的加速效果。
加速比通过计算各形状下的平均值获得，代表了算子的整体性能表现。

![算子加速比](./assets/speedup-20250423.png)

## 贡献指南

如果您有兴趣参与 FlagGems 项目，请参考 [贡献指南](./contribution.md)。
我们非常欢迎任何形式的贡献。

## 联系我们

如果您对本项目有任何疑问，请提交 Issue，或通过邮件联系我们：<a href="mailto:flaggems@baai.ac.cn">flaggems@baai.ac.cn</a>。

我们也建立了 FlagGems 微信交流群。
扫描二维码即可加入群聊！获取最新发布动态，或进行技术交流与反馈。

<p align="center">
 <img src="https://github.com/user-attachments/assets/69019a23-0550-44b1-ac42-e73f06cb55d6" alt="bge_wechat_group" class="center" width="200">
</p>

## 开源协议

FlagGems 项目基于 [Apache 2.0](https://github.com/flagos-ai/FlagGems/blob/master/LICENSE) 协议开源。

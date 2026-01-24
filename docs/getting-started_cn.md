# FlagGems 快速入门

## 介绍

FlagGems 是一个使用 Triton 语言实现的高性能通用算子库。
它旨在提供一系列内核函数，以加速大语言模型（LLM）的训练和推理。

通过在 PyTorch 的 ATen 后端进行注册，FlagGems 实现了无缝切换，
允许用户在无需修改模型代码的情况下切换到 Triton 函数库。
FlagGems 支持用于不同 AI 芯片的 [FlagTree 编译器](https://github.com/flagos-ai/flagtree/)，
以及 OpenAI Triton 编译器（适用于 NVIDIA 和 AMD）。

## 快速安装

FlagGems 可以作为纯 Python 包安装，也可以作为带有 C 扩展的包安装，以获得更好的运行时性能。
默认情况下，它不会构建 C 扩展。有关如何使用 C++ 运行时的信息，请参阅[安装指南](./installation.md)。

### 安装构建依赖

```shell
pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
```

### 安装

将仓库克隆到本地环境：

```shell
git clone https://github.com/flagos-ai/FlagGems.git
```

然后使用以下命令触发安装：

```shell
cd FlagGems
# 如果您想使用原生 Triton 而不是 FlagTree，请跳过此步骤。
# 其他后端：替换为对应的 requirements_backendxxx.txt
pip install -r flag_tree_requirements/requirements_nvidia.txt
pip install --no-build-isolation .
```

您还可以使用以下命令进行可编辑模式安装：

```shell
cd FlagGems
pip install --no-build-isolation -e .
```

此外，您可以构建 wheel 包进行安装：

```shell
pip install -U build
git clone https://github.com/flagos-ai/FlagGems.git
cd FlagGems
python -m build --no-isolation --wheel .
```

## 如何使用 Gems

### 导入

```python
# 永久启用 flag_gems
import flag_gems
flag_gems.enable()

# 或者临时启用 flag_gems
with flag_gems.use_gems():
    pass
```

例如：

```python
import torch
import flag_gems

M, N, K = 1024, 1024, 1024
A = torch.randn((M, K), dtype=torch.float16, device=flag_gems.device)
B = torch.randn((K, N), dtype=torch.float16, device=flag_gems.device)
with flag_gems.use_gems():
    C = torch.mm(A, B)
```

## 如何使用实验性 Gems

`experimental_ops` 模块为尚未准备好正式发布的新算子提供了空间。
可以通过 `flag_gems.experimental_ops.*` 访问该模块中的算子。
这些算子遵循与核心算子相同的开发模式。

```python
import flag_gems

# 全局启用
flag_gems.enable()
result = flag_gems.experimental_ops.rmsnorm(*args)

# 或在特定范围内使用
with flag_gems.use_gems():
    result = flag_gems.experimental_ops.rmsnorm(*args)
```

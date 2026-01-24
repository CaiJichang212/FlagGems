# 安装指南

## 从 PyPI 安装

由于 FlagGems 尚未在 PyPI 上发布，目前只能从源码安装。

## 从源码编译安装

### 克隆源码

```shell
git clone https://github.com/FlagOpen/FlagGems
cd FlagGems/
```

### 安装 FlagTree

[FlagTree](https://github.com/flagos-ai/flagtree/) 是一个开源的、针对多种 AI 芯片的统一编译器项目。
如果你想使用原生 Triton 而不是 FlagTree，请跳过此步骤。

有关其他后端环境要求的详细信息，请查看 FlagTree 项目中的 [build](https://github.com/flagos-ai/FlagTree/blob/main/documents/build.md) 文档。

```shell
# 其他后端：替换为对应的 requirements_backendxxx.txt
pip install -r flag_tree_requirements/requirements_nvidia.txt
```

### 构建系统

FlagGems 遵循 [PEP 518](https://peps.python.org/pep-0518/) 标准，并包含一个 `pyproject.toml` 文件来指定如何构建包。

Python 包 `flag_gems` 使用 `scikit-build-core` 作为 [构建后端](https://peps.python.org/pep-0517/#build-backend-interface)。
简要介绍一下，[`scikit-build-core`](https://scikit-build-core.readthedocs.io/en/latest/) 是一个构建后端，它提供了 CMake 与 Python 构建系统之间的桥梁，使得使用 CMake 创建 Python 模块变得更加容易。
我们使用它来避免在 `setup.py` 中手动包装 CMake。

### 构建隔离 (Build-isolation)

遵循 PEP 517 中对 [构建前端的建议](https://peps.python.org/pep-0517/#recommendations-for-build-frontends-non-normative)，`pip` 或其他现代构建前端会使用隔离环境来构建包。
这包括在构建包之前创建一个虚拟环境并在其中安装构建要求。

如果你不想使用构建隔离（通常在可编辑安装的情况下），可以向 `pip install` 传递 `--no-build-isolation` 标志，但你需要预先在当前环境中安装 `build-requirements`。
检查 `pyproject.toml` 文件中的 `[build-system.requires]` 部分并安装所需的包。

示例命令：

```shell
pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
```

FlagGems 可以作为纯 Python 包或带有 C 扩展的包安装。
默认情况下，不构建 C 扩展，因为这仍然是一个实验性功能。

### 作为纯 Python 包安装

要将 FlagGems 作为纯 Python 包安装，请使用以下命令：

```shell
# 安装到 site-packages
pip install .

# 或以可编辑模式安装
pip install -e .
```

### 安装 C 扩展

要在 FlagGems 中启用 C 扩展构建，必须在配置阶段将 CMake 选项 `-DFLAGGEMS_BUILD_C_EXTENSION=ON` 传递给 CMake。
这可以通过 `SKBUILD_CMAKE_ARGS` 或 `CMAKE_ARGS` 环境变量将参数传递给 CMake 来实现。

请注意，对于环境变量 `SKBUILD_CMAKE_ARGS`，多个选项由分号 (`;`) 分隔；而对于 `CMAKE_ARGS`，它们由空格分隔。
这与 `scikit-build-core` 及其前身 `scikit-build` 之间的差异有关。

配置 FlagGems 的选项如下表所示：

| 选项                             | 描述                                | 默认值                                  |
| -------------------------------- | ----------------------------------- | --------------------------------------- |
| FLAGGEMS_USE_EXTERNAL_TRITON_JIT | 是否使用外部 Triton JIT 库          | OFF                                     |
| FLAGGEMS_USE_EXTERNAL_PYBIND11   | 是否使用外部 pybind11 库            | ON                                      |
| FLAGGEMS_BUILD_C_EXTENSIONS      | 是否构建 C 扩展                     | 当作为顶层项目时为 ON                   |
| FLAGGEMS_BUILD_CTESTS            | 是否构建 C++ 单元测试               | 与 FLAGGEMS_BUILD_C_EXTENSIONS 的值一致 |
| FLAGGEMS_INSTALL                 | 是否安装 FlagGems 的 CMake 配置文件 | 当作为顶层项目时为 ON                   |

FlagGems 的 C 扩展依赖于 [TritonJIT](https://github.com/iclementine/libtorch_example/)，这是一个在 C++ 中实现 Triton JIT 运行时并允许从 C++ 调用 Triton JIT 函数的库。
请注意，如果你使用外部 TritonJIT 构建 FlagGems，你应该预先构建并安装它，然后将选项 `-DTritonJIT_ROOT=<安装路径>` 传递给 CMake。

配置 `scikit-build-core` 的其他常用环境变量包括：

1. `SKBUILD_CMAKE_BUILD_TYPE`：用于配置项目的构建类型。
   有效值为 `Release`、`Debug`、`RelWithDebInfo` 和 `MinSizeRel`；

1. `SKBUILD_BUILD_DIR`：配置项目的构建目录。
   默认值为 `build/<cache_tag>`，在 `pyproject.toml` 中定义。

常用的 pip 选项包括：

1. `-v`：显示配置和构建过程的日志；
1. `-e`：创建可编辑安装。请注意，在可编辑安装中，C 部分（头文件、库、cmake 配置文件）将安装到 `site-packages` 目录，而 Python 部分保持在原位，并在 `site-packages` 目录中安装一个加载器来查找它。
   有关此安装模式的更多详细信息，请参阅 `scikit-build-core` 的 [文档](https://scikit-build-core.readthedocs.io/en/latest/configuration/index.html#editable-installs)。
1. `--no-build-isolation`：不创建单独的虚拟环境来构建项目。
   这通常与可编辑安装一起使用。
   请注意，在不隔离构建时，必须手动安装构建依赖项。
1. `--no-deps`：不安装包依赖项。
   当你不想更新依赖项时，这很有用。

下面提供了示例配方，方便复制粘贴。

使用外部 TritonJIT 的可编辑安装：

```shell
pip install -U scikit-build-core ninja cmake pybind11

CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DFLAGGEMS_USE_EXTERNAL_TRITON_JIT=ON -DTritonJIT_ROOT=<triton-jit 安装路径>" \
pip install --no-build-isolation -v -e .
```

通过 FetchContent 将 TritonJIT 作为子项目进行的可编辑安装：

```shell
CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON" \
pip install --no-build-isolation -v -e .
```

## 打包

创建源码分发版 (sdist) 或二进制分发版 (wheel) 与从源码构建安装类似。
它涉及调用构建前端（如 `pip` 或 `build`）并将命令传递给构建后端（此处为 `scikit-build-core`）。

### 使用构建前端：build

推荐使用 `build` 包来构建 wheel。

```sh
pip install -U build
python -m build --no-isolation --no-deps .
```

这将首先创建一个源码分发版 (sdist)，然后从源码分发版构建二进制分发版 (wheel)。

如果你想禁用默认行为（源码目录 -> sdist -> wheel），你可以：

- 传递 `--sdist` 以从源码构建源码分发版 (源码目录 -> sdist)；

- 或传递 `--wheel` 以从源码构建二进制分发版 (源码目录 -> wheel)；

- 或同时传递 `--sdist` 和 `--wheel` 以从源码同时构建源码和二进制分发版 (源码目录 -> sdist, 以及 源码目录 -> wheel)。

结果将放置在 `dist/` 目录中。

### 使用构建前端：pip

或者，你可以使用 `pip` 构建 wheel。

```sh
pip wheel --no-build-isolation --no-deps -w dist .
```

用于配置 `scikit-build-core` 的环境变量的工作方式与上述相同。

在二进制分发版 (wheel) 构建完成后，使用 `pip` 进行安装。

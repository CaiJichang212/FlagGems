# FlagGems 代码贡献指南

在提交 Pull Request (PR) 时，贡献者应当描述更改的内容及原因。
如果适用，请同时提供测试用例。
PR 在合并前需要至少 **一名成员** 的批准（Approval）。
此外，PR 必须通过持续集成（CI）检查。

目前，持续集成检查包含四个流水线：

## 代码格式检查 (Code Format Check)

在 FlagGems 中使用 pre-commit git hooks，可以在执行 `git commit` 命令时自动格式化 Python 源码并进行基础的代码预检查。

```bash
pip install pre-commit
pre-commit install
pre-commit
```

## 算子单元测试 (Op Unit Test)

单元测试用于检查算子的正确性。
在添加新算子时，您需要在 `tests` 目录下的相应文件中添加测试用例。
如果添加了新的测试文件，您还应该将测试命令添加到 `tools/coverage.sh` 文件中的 `cmd` 变量里。

对于算子测试，请在测试函数前使用 `@pytest.mark.{OP_NAME}` 装饰器，以便我们可以通过 `pytest -m` 运行指定算子的单元测试。

一个单元测试函数可以被多个自定义 mark 装饰。

如果您添加了 C++ 包装器，还应添加相应的 ctest。详情请参阅 [添加 C++ 包装器](add_a_cpp_wrapper.md)。

## 模型测试 (Model Test)

模型测试用于检查模型的正确性。添加新模型的过程与添加新算子类似。

## Python 代码覆盖率 (Python Coverage)

Python 覆盖率检查当前 PR 中新增代码的覆盖情况。它依赖于 `Op Unit Test` 和 `Model Test` 的成功执行；否则，该步骤将被跳过。代码覆盖率的计算方式为：

> PR 中新增且被测试覆盖的行数 / PR 中新增代码的总行数

注：由于 Triton JIT 函数实际上不在 Python 中运行，它们将从覆盖率计算中排除。

代码合并的要求是覆盖率达到 **90%** 或以上。有关代码覆盖率的详细信息可以在日志和给出的 URL 中查看。

如需在本地复现，您需要安装 `lcov`、`coverage` 和 `PyGithub` 等工具。

```bash
cd $FlagGemsROOT
PR_ID=your_pr_id
bash tools/op-unit-test.sh
bash tools/model-test.sh
tools/code_coverage/coverage.sh PR_ID
```

## 算子性能基准测试 (Operator Performance Benchmarking)

目前，流水线不自动检查算子的性能。您可以在 `benchmark` 目录下编写性能测试来评估您的优化结果。

`Op Benchmark` 用于评估算子的性能。如果您正在添加新算子，需要在 `benchmark` 目录下的适当文件中添加相应的测试用例。建议按照以下步骤为新算子添加测试用例：

1. **选择合适的测试文件**
   根据算子类型，在 `benchmark` 目录中选择相应的文件：

   - 对于规约（Reduction）算子，将测试用例添加到 `test_reduction_perf.py`。

   - 对于张量构造（Tensor Constructor）算子，将测试用例添加到 `test_tensor_constructor_perf.py`。

   - 如果算子不属于现有类别，可以将其添加到 `test_special_perf.py` 或为新的算子类别创建一个新文件。

2. **检查现有的基准测试类**
   确定文件后，查看继承自 `Benchmark` 结构的现有类，看是否有适合您算子测试场景的类，特别要考虑：

   - **指标收集（Metric collection）** 是否合适。

   - **输入生成函数（Input generation function）**（`input_generator` 或 `input_fn`）是否合适。

3. **添加测试用例**
   根据测试场景，按照以下方式之一添加测试用例：

   - **使用现有的指标和输入生成器**

     如果现有的指标收集和输入生成函数满足您算子的要求，您可以直接参考文件中的代码组织方式，添加一行 `pytest.mark.parametrize`。例如，参考 `test_binary_pointwise_perf.py` 中的算子。

   - **自定义输入生成器**

     如果指标收集合适但输入生成函数不满足算子要求，您可以实现自定义的 `input_generator`。可以参考 `test_special_perf.py` 中的 `topk_input_fn` 函数作为 `topk` 算子自定义输入函数的示例。

   - **自定义指标和输入生成器**

     如果现有的指标收集和输入生成函数都不满足算子需求，您可以创建一个新类。该类应定义算子特定的指标收集逻辑和自定义输入生成器。您可以参考 `benchmark` 目录下的各种 `Benchmark` 子类作为示例。

## 3. 项目结构

```cpp
FlagGems
├── src                  // Python 源代码
│   └──flag_gems
│       ├──utils         // Python 自动代码生成工具
│       ├──ops           // Python 单算子实现
│       ├──fused         // Python 融合算子实现
│       ├──testing       // Python 测试工具
├── tests                // Python 精度测试文件
├── benchmark            // Python 性能测试文件
├── examples             // Python 模型测试文件
├── cmake                // C-extension 的 C++ cmake 文件
├── include              // C++ 头文件
├── lib                  // 算子库的 C++ 源代码
├── ctest                // C++ 测试文件
├── triton_src           // 临时 Triton JIT 函数源码
├── docs                 // flag_gems 文档
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── ...
```

## 4. 许可证

您做出的任何贡献都将遵循 [Apache License](https://github.com/FlagOpen/FlagGems/blob/master/LICENSE)。

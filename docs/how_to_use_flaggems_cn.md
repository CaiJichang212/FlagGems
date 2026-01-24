# 如何使用 FlagGems

## 基础用法

要使用 `FlagGems` 算子库，只需在运行计算前导入并启用加速。您可以全局启用或在特定范围内启用。

### 方式 1：全局启用

要在整个脚本或交互式会话中应用 `FlagGems` 优化：

```python
import flag_gems

# 全局启用 flag_gems
flag_gems.enable()
```

启用后，代码中所有受支持的算子都将自动替换为优化后的 `FlagGems` 实现，无需进一步修改。

### 方式 2：局部（作用域）启用

为了实现更精细的控制，您可以使用上下文管理器仅在特定的代码块中启用 `FlagGems`：

```python
import flag_gems

# 临时启用 flag_gems
with flag_gems.use_gems():
    # 此代码块内的代码将使用 Gems 加速的算子
    ...
```

这种局部用法在以下场景中非常有用：
- 对比性能差异
- 验证实现的正确性
- 在复杂工作流中选择性地应用加速

## 高级用法

`flag_gems.enable(...)` 函数支持多个可选参数，让您能够更精细地控制加速的应用方式。这在复杂工作流的集成、调试或分析中非常有用。

### 参数概览

<!--TODO(Qiming): verify the list of parameters.-->

| 参数     | 类型      | 描述                                      |
| -------- | --------- | ----------------------------------------- |
| `unused` | List[str] | 禁用特定的算子                            |
| `record` | bool      | 记录算子调用情况，用于调试或分析          |
| `path`   | str       | 日志文件路径（仅在 `record=True` 时使用） |

### 示例：选择性禁用特定算子

您可以使用 `unused` 参数将某些算子排除在 `FlagGems` 加速之外。当某个算子在您的任务中表现不符合预期，或者您发现性能欠佳并希望暂时回退到原始实现时，这非常有用。

```python
flag_gems.enable(unused=["sum", "add"])
```

在此配置下，`sum` 和 `add` 将继续使用 PyTorch 原生实现，而其他受支持的算子将使用 `FlagGems` 版本。

### 示例：启用调试日志

设置 `record=True` 以在运行时记录算子使用情况，并通过 `path` 指定输出路径。

```python
flag_gems.enable(
    record=True,
    path="./gems_debug.log"
)
```

运行脚本后，检查日志文件（如 `gems_debug.log`）以查看通过 `flag_gems` 调用的算子列表。

样例日志内容：
```shell
$ cat ./gems_debug.log
[DEBUG] flag_gems.ops.fill: GEMS FILL_SCALAR_
[DEBUG] flag_gems.ops.fill: GEMS FILL_SCALAR_
[DEBUG] flag_gems.ops.mm: GEMS MM
[DEBUG] flag_gems.fused.reshape_and_cache: GEMS RESHAPE_AND_CACHE
```

## 在非 NVIDIA 硬件上运行 FlagGems

### 支持的平台

FlagGems 支持除 NVIDIA 之外的多种 AI 芯片。有关已验证平台的最新列表，请参考[支持平台](./features.md#platforms-supported)。

### 统一的使用接口

无论底层硬件如何，`flag_gems` 的用法完全相同。从 NVIDIA 切换到非 NVIDIA 平台时，无需修改应用代码。

一旦调用 `import flag_gems` 并通过 `flag_gems.enable()` 启用加速，算子分发将自动路由到正确的后端。这在异构环境中提供了高度一致的开发体验。

### 后端要求

虽然用法模式未变，但在非 NVIDIA 硬件上运行需要底层依赖——**PyTorch** 和 **Triton 编译器**——在目标平台上可用且配置正确。

获取兼容版本有两种常见方式：

1. **联系硬件厂商**
   硬件厂商通常会维护针对其芯片优化的 PyTorch 和 Triton 定制版本。请联系厂商获取相应版本。

2. **探索 FlagTree 项目**
   [FlagTree](https://github.com/flagos-ai/flagtree) 项目提供了一个统一的 Triton 编译器，支持多种 AI 芯片（包括 NVIDIA 和非 NVIDIA 平台）。它将厂商特定的补丁和增强功能整合到一个共享的开源后端中，简化了编译器维护并实现了多平台兼容。

   > [!Note]
   > FlagTree 仅提供 Triton。仍需单独准备匹配的 PyTorch 版本。

> [!Note]
> 某些平台可能需要额外的设置或补丁。

### 后端自动检测与手动设置

默认情况下，`flag_gems` 在运行时会自动检测当前硬件后端并选择相应的实现。在大多数情况下，无需手动配置，即可开箱即用。

但是，如果自动检测失败或与您的环境不兼容，您可以手动设置目标后端。在运行代码前设置以下环境变量：

```shell
export GEMS_VENDOR=<厂商名称>
```

> ⚠️ 该设置应与实际硬件平台匹配。手动设置错误的后端可能会导致运行时错误。

您可以在运行时通过以下方式验证当前激活的后端：

```python
import flag_gems
print(flag_gems.vendor_name)
```

## 与主流框架集成

为了帮助将 `flag_gems` 集成到实际场景中，我们提供了与常用深度学习框架的示例。这些集成仅需极少的代码改动，并保留了原有的工作流结构。

完整示例请参考源码仓库中的 [`examples/`](https://github.com/flagos-ai/FlagGems/tree/master/examples) 目录。

### 示例 1：Hugging Face Transformers

与 Hugging Face 的 `transformers` 库集成非常简单。您只需遵循前面介绍的基础用法模式即可。

在推理过程中，您可以在不修改模型或分词器逻辑的情况下激活加速。示例如下：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import flag_gems

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("sharpbai/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("sharpbai/Llama-2-7b-hf")

# 移动模型到正确设备并设置为评估模式
device = flag_gems.device
model.to(device).eval()

# 准备输入并在启用 flag_gems 的情况下运行推理
inputs = tokenizer(prompt, return_tensors="pt").to(device=device)
with flag_gems.use_gems():
    output = model.generate(**inputs, max_length=100, num_beams=5)
```

这种模式确保了生成过程中所有兼容的算子都将自动获得加速。更多示例见：
- `examples/model_llama_test.py`
- `examples/model_llava_test.py`

### 示例 2：vLLM

[vLLM](https://github.com/vllm-project/vllm) 是一个专为高效服务大语言模型设计的高吞吐推理引擎。它支持分页注意力 (paged attention)、连续批处理 (continuous batching) 和优化的内存管理等特性。

`flag_gems` 可以集成到 vLLM 中，替换标准 PyTorch (`aten`) 算子以及 vLLM 内部的自定义内核。

#### 替换 vLLM 中的标准 PyTorch 算子

要加速 vLLM 中的标准 PyTorch 算子（如 `add`, `masked_fill`），用法与其他框架一致：

- 在任何模型初始化或推理之前调用 `flag_gems.enable()`。
- 这将覆盖所有兼容的 PyTorch `aten` 算子，包括 vLLM 间接使用的算子。

#### 替换 vLLM 特有的自定义算子

为了进一步优化 vLLM 内部内核，`flag_gems` 提供了一个额外的 API：

```python
flag_gems.apply_gems_patches_to_vllm(verbose=True)
```

该函数会将特定的 vLLM 自定义 C++ 或 Triton 算子替换为 `flag_gems` 实现。当 `verbose=True` 时，它会记录哪些函数被替换了：

```none
Patched RMSNorm.forward_cuda with FLAGGEMS custom_gems_rms_forward_cuda
Patched RotaryEmbedding.forward_cuda with FLAGGEMS custom_gems_rope_forward_cuda
Patched SiluAndMul.forward_cuda with FLAGGEMS custom_gems_silu_and_mul
```

当需要更全面的 `flag_gems` 覆盖时，请使用此功能。

#### 完整示例：在 vLLM 推理中启用 `flag_gems`

```python
from vllm import LLM, SamplingParams
import flag_gems

# 步骤 1: 为 PyTorch (aten) 算子启用加速
flag_gems.enable()

# 步骤 2: (可选) 补丁化 vLLM 自定义算子
flag_gems.apply_gems_patches_to_vllm(verbose=True)

# 步骤 3: 像往常一样使用 vLLM
llm = LLM(model="sharpbai/Llama-2-7b-hf")
sampling_params = SamplingParams(temperature=0.8, max_tokens=128)

output = llm.generate("给我讲个笑话。", sampling_params)
print(output)
```

### 示例 3：Megatron

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 是一个高度优化的框架，用于大规模语言模型的预训练和微调。由于其与自定义训练循环和内部工具紧密集成，在 Megatron 中集成 `flag_gems` 需要更具针对性的方法。

由于 Megatron 的训练循环将分布式数据加载、梯度累加和流水线并行紧密结合，我们建议仅在正向和反向计算阶段周围应用 `flag_gems`。

#### 推荐的集成点

在 Megatron 中使用 `flag_gems` 最可靠的方法是修改 `train_step` 函数（位于 [`megatron/training/training.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/training.py#L1360)）。具体来说，使用 `flag_gems.use_gems()` 包裹调用 `forward_backward_func` 的代码块，如下所示：

```python
def train_step(forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config):
    """单次训练步。"""
    args = get_args()
    timers = get_timers()

     # 省略 CUDA Graph 捕获逻辑
    rerun_state_machine = get_rerun_state_machine()
    while rerun_state_machine.should_run_forward_backward(data_iterator):
    	# 省略梯度清零逻辑

        # 使用 flag_gems 加速的正向/反向传播
        import flag_gems
        with flag_gems.use_gems():
          forward_backward_func = get_forward_backward_func()
          losses_reduced = forward_backward_func(
              forward_step_func=forward_step_func,
              data_iterator=data_iterator,
              model=model,
              num_microbatches=get_num_microbatches(),
              seq_length=args.seq_length,
              micro_batch_size=args.micro_batch_size,
              decoder_seq_length=args.decoder_seq_length,
              forward_only=False,
              adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
          )

    should_checkpoint, should_exit, exit_code = rerun_state_machine.should_checkpoint_and_exit()
    if should_exit:
        return {}, True, should_checkpoint, should_exit, exit_code, None, None

    # 省略其他步后操作
```

这确保了只有正向和反向计算逻辑在 `flag_gems` 加速下运行，而其他组件（如数据加载和优化器步骤）保持不变。

#### 范围与限制

虽然 `flag_gems.enable()` 在大多数框架中已足够，但我们观察到，在 Megatron 流水线的早期应用它有时会导致意外行为，尤其是在数据加载阶段。为了更好的稳定性，我们建议将 `flag_gems.use_gems()` 上下文管理器限制在计算阶段。

如果您希望加速更广泛的组件（如优化器、预处理），可以尝试使用 `flag_gems.enable()` 全局启用。然而，这种方法测试较少，可能需要根据您的 Megatron 版本进行额外的验证。

我们鼓励社区贡献——请开启 `issue` 或提交 PR 以帮助改进更广泛的 Megatron 集成。

### 多 GPU 部署

在实际的 LLM 部署场景中，通常需要多 GPU 或多节点设置，以支持大模型尺寸和高吞吐量推理。`flag_gems` 通过跨多个 GPU 加速算子执行来支持这些场景。

#### 单节点与多节点使用

对于**单节点部署**，集成非常简单。您只需在脚本开头导入并调用 `flag_gems.enable()` 即可。这将在无需任何额外更改的情况下启用加速。

然而，在**多节点部署**中，这种方法是不够的。分布式推理框架（如 vLLM）会在各个节点上启动多个工作进程 (worker processes)，每个进程都必须独立初始化 `flag_gems`。如果仅在启动脚本中激活，远程节点上的工作进程将回退到默认实现，从而失去加速效果。

#### 集成示例：vLLM + DeepSeek

以下是在分布式 vLLM + DeepSeek 部署中启用 `flag_gems` 的方法：

1. **基准测试验证**

   在集成 `flag_gems` 之前，请验证模型在没有它的情况下能否正常加载和服务。例如，加载像 `Deepseek-R1` 这样的模型通常需要**至少两张 H100 GPU**，并且根据检查点大小和系统 I/O，初始化可能需要**长达 20 分钟**。

2. **将 `flag_gems` 注入 vLLM 工作进程代码**

   根据您的 vLLM 版本找到相应的模型运行脚本：

   - 如果您使用的是 **vLLM v1 架构**（适用于 vLLM ≥ 0.8），请修改 `vllm/v1/worker/gpu_model_runner.py`
   - 如果您使用的是**传统的 v0 架构**，请修改 `vllm/worker/model_runner.py`

   在任一文件中，在最后一个 `import` 语句后插入以下逻辑：

   ```python
   import os
   if os.getenv("USE_FLAGGEMS", "false").lower() in ("1", "true", "yes"):
       try:
           import flag_gems
           flag_gems.enable()
           flag_gems.apply_gems_patches_to_vllm(verbose=True)
           logger.info("Successfully enabled flag_gems as default ops implementation.")
       except ImportError:
           logger.warning("Failed to import 'flag_gems'. Falling back to default implementation.")
       except Exception as e:
           logger.warning(f"Failed to enable 'flag_gems': {e}. Falling back to default implementation.")
   ```

3. **在所有节点上设置环境变量**

   在启动服务之前，确保所有节点都设置了以下环境变量：

   ```shell
   export USE_FLAGGEMS=1
   ```

4. **启动分布式推理并确认加速**

   启动服务并检查每个节点上的启动日志，查看是否有表示算子已被覆盖的消息。

   ```none
   Overriding a previously registered kernel for the same operator and the same dispatch key
   operator: aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
     registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
   dispatch key: CUDA
   previous kernel: registered at /pytorch/aten/src/ATen/LegacyBatchingRegistrations.cpp:1079
        new kernel: registered at /dev/null:488 (Triggered internally at /pytorch/aten/src/ATen/core/dispatch/OperatorEntry.cpp:154.)
   self.m.impl(
   ```

   这证实了 `flag_gems` 已在所有 GPU 上成功启用。

## 使用 Gems 算子构建自定义模型

在某些场景下，用户可能希望从头开始构建自己的模型，或者修改现有模型以更好地适应特定需求。为了支持这一点，`flag_gems` 提供了一系列不断增加的高性能模块，这些模块在大语言模型 (LLM) 中常用。

这些组件是使用 `flag_gems` 加速的算子实现的，可以像任何标准的 `torch.nn.Module` 一样使用。您可以无缝地将它们集成到您的架构中，从而从算子级加速中受益，而无需编写自定义 CUDA 或 Triton 代码。

可用模块位于：
[flag_gems/modules](https://github.com/flagos-ai/FlagGems/tree/master/src/flag_gems/modules)

### 可用模块

<!--TODO(Qiming): Double check the list of modules. -->

| 模块                   | 描述                                | 支持特性                                   |
| ---------------------- | ----------------------------------- | ------------------------------------------ |
| `GemsRMSNorm`          | RMS LayerNorm                       | 融合残差加法, 支持 `inplace` 和 `outplace` |
| `GemsRope`             | 标准旋转位置编码 (RoPE)             | 支持 `inplace` 和 `outplace`               |
| `GemsDeepseekYarnRoPE` | 适用于 DeepSeek 类 LLM 的 Yarn RoPE | 支持 `inplace` 和 `outplace`               |
| `GemsSiluAndMul`       | 融合 SiLU 激活与逐元素乘法          | 仅支持 `outplace`                          |

我们鼓励用户使用这些模块作为等效 PyTorch 层的直接替换。更多组件（如融合注意力机制、MoE 层和 Transformer 块）正在开发中。

## 使用 Gems 获得最佳性能

虽然 `flag_gems` 的内核旨在实现高性能，但在完整的模型部署中实现最佳的端到端速度需要仔细的集成并考虑运行时行为。特别是，有两个常见的性能瓶颈：

- 生产环境中的**运行时自动调优开销**。
- 由于框架级算子分发或与 Triton 运行时的交互而导致的**次优分发**。

这些问题有时会抵消高度优化内核带来的好处。为了解决这些问题，我们提供了两条互补的优化路径，旨在确保 `flag_gems` 在实际推理场景中以最高效率运行。

### 为推理场景预调优模型形状

`flag_gems` 集成了 [`LibTuner`](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/utils/libentry.py#L139)，这是对 Triton 自动调优系统的一种轻量级增强。`libtuner` 引入了**持久化的单设备调优缓存**，有助于减轻 Triton 默认自动调优过程带来的运行时开销。

#### 为什么需要预调优？

Triton 通常在首次执行新输入形状时进行自动调优，这可能会导致延迟峰值——尤其是在对延迟敏感的推理系统中。`libtuner` 通过以下方式解决此问题：

- **持久化缓存**：最佳自动调优配置跨运行保存。
- **跨进程共享**：缓存可在同一设备上的不同进程之间共享。
- **减少运行时开销**：一旦调优完成，算子在未来的运行中将跳过调优。

这对于像 `mm` 和 `addmm` 这样经常触发 Triton 自动调优逻辑的算子特别有用。

#### 如何使用预调优

要主动预热您的系统并填充缓存：

1. 识别生产负载中使用的关键输入形状。
2. 运行预调优脚本进行基准测试并缓存最佳配置：`python examples/pretune.py`
3. 正常部署，`flag_gems` 将在推理过程中自动从缓存中选取最优配置。

> ✅ `pretune.py` 接受示例形状和工作负载，以模拟您模型的实际用例。您可以根据批次大小 (batch size)、序列长度 (sequence length) 等进行自定义。

> 💡 在像 **vLLM** (`v0.8.5+`) 这样的框架中，启用 `--compile-mode` 会自动执行预热步骤。如果集成了 `flag_gems`，这也会隐式触发基于 `libtuner` 的预调优。

有关更多详细信息或自定义调优缓存路径和设置，请参考 [`examples/pretune.py`](https://github.com/flagos-ai/FlagGems/blob/master/examples/pretune.py) 示例。

### 使用基于 C++ 的算子包装器以获得进一步的性能提升

`flag_gems` 中的另一个高级优化路径是为选定的算子使用 **C++ 包装器 (wrappers)**。虽然 Triton 内核提供了相当不错的计算性能，但 Triton 本身是一个嵌入 Python 的 DSL。这意味着算子定义和运行时分发都依赖于 Python，这在延迟敏感或高吞吐量场景中可能会引入**不可忽视的开销**。

为了解决这个问题，我们提供了一个 C++ 运行时解决方案，它将算子的包装逻辑、注册机制和运行时管理完全封装在 C++ 中，同时仍然复用底层的 Triton 内核进行实际计算。这种方法在保持 Triton 算子级效率的同时，显著减少了与 Python 相关的开销，从而能够与底层 CUDA 工作流更紧密地集成，并提高整体推理性能。

#### 安装与设置

要使用 C++ 算子包装器：

1. 按照 [安装指南](./installation.md) 编译并安装 C++ 版本的 `flag_gems`。

2. 使用以下代码段验证安装是否成功：

   ```python
   try:
       from flag_gems import c_operators
       has_c_extension = True
   except Exception as e:
       c_operators = None  # 如果 c_operators 不可用，避免导入错误
       has_c_extension = False
   ```

   如果 `has_c_extension` 为 `True`，则 C++ 运行时路径可用。

3. 安装成功后，在**补丁模式 (patch mode)** 下以及显式使用 `flag_gems` 定义的模块构建模型时，将自动首选 C++ 包装器。

   例如，`gems_rms_forward` 默认将使用 `rms_norm` 的 C++ 包装版本。您可以参考 [`normalization.py`](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/modules/normalization.py#L46) 中的实际用法，以更好地了解 C++ 算子包装器是如何集成和调用的。

#### 显式使用 C++ 算子

如果您想*直接调用 C++ 包装的算子*，绕过任何补丁逻辑或回退机制，请按如下方式使用 `torch.ops.flag_gems` 命名空间：

```python
output = torch.ops.flag_gems.fused_add_rms_norm(...)
```

这为您提供了对算子分发的*精确控制*，在性能敏感的上下文中非常有用。

#### 当前支持的 C++ 包装算子

<!--TODO(Qiming): Move this list elsewhere. -->

| 算子名称             | 描述                            |
| -------------------- | ------------------------------- |
| `add`                | 逐元素加法                      |
| `bmm`                | 批量矩阵乘法                    |
| `cat`                | 拼接                            |
| `fused_add_rms_norm` | 融合加法 + RMSNorm              |
| `mm`                 | 矩阵乘法                        |
| `nonzero`            | 返回非零元素的索引              |
| `rms_norm`           | 均方根归一化 (RMSNorm)          |
| `rotary_embedding`   | 旋转位置编码 (Rotary Embedding) |
| `sum`                | 跨维度的求和                    |

作为我们持续性能优化路线图的一部分，我们正在积极扩展此列表。

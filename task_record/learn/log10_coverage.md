# log10 测例覆盖清单（赛道一）

> 说明：本清单对应 `FlagGems/tests/test_log10_ops.py` 中的 log10 系列测例。
> 请在实际评测环境运行后，补充性能对比表。

## 覆盖维度

- 输入规模：
  - 小尺寸：`(1, 1)`, `(8, 8)`
  - 常规尺寸：`(64, 64)`, `(256, 256)`
  - 大尺寸：`(1024, 1024)`
- 输入维数：
  - 0 维：`()`（来自 `POINTWISE_SHAPES`）
  - 1 维：`(1,)`（来自 `POINTWISE_SHAPES`）
  - 2 维：`(1,1)`, `(8,8)`, `(64,64)`, `(256,256)`, `(1024,1024)`
  - 3/4/5 维：`(20,320,15)`, `(16,128,64,60)`, `(16,7,57,32,29)`（来自 `POINTWISE_SHAPES`）
- 参数模式：
  - 非原位：`torch.log10`
  - 原位：`torch.log10_`
  - out 参数：`torch.log10(..., out=...)`
- 功能分支：
  - 连续 / 非连续输入
  - 空张量
  - 负数 / 零值（应产生 `-inf`/`nan`）
  - 非法 dtype（int）错误提示
  - out 形状不匹配错误提示
  - out dtype 不匹配错误提示

## 测例对应关系

| 测例 | 覆盖项 |
|---|---|
| `test_accuracy_log10` | 维度覆盖（0/1/3/4/5D），基础正确性 |
| `test_accuracy_log10_size_coverage` | 输入规模覆盖（小/常规/大），2D |
| `test_accuracy_log10_edge_values` | 负数/零值/边界数值 |
| `test_accuracy_log10_empty` | 空张量 |
| `test_accuracy_log10_noncontiguous` | 非连续输入 |
| `test_accuracy_log10_` | 原位模式 |
| `test_accuracy_log10_out` | out 参数 |
| `test_log10_invalid_dtype` | 非法 dtype |
| `test_log10_out_invalid_shape` | out 形状错误 |
| `test_log10_out_invalid_dtype` | out dtype 错误 |

## 性能对比表（待填写）

| Shape | Dtype | PyTorch (ms) | FlagGems (ms) | Speedup |
|---|---|---:|---:|---:|
| (1,1) | fp16 |  |  |  |
| (8,8) | fp16 |  |  |  |
| (64,64) | fp16 |  |  |  |
| (256,256) | fp16 |  |  |  |
| (1024,1024) | fp16 |  |  |  |
| (1,1) | fp32 |  |  |  |
| (8,8) | fp32 |  |  |  |
| (64,64) | fp32 |  |  |  |
| (256,256) | fp32 |  |  |  |
| (1024,1024) | fp32 |  |  |  |

## 实现细节

- 入口函数：`flag_gems.ops.log10` / `log10_` / `log10_out`。
- 内核：`_log10_kernel` 使用 Triton 实现，计算路径为 `y = log(x) * (1/ln(10))`，中间在 fp32 计算，再回写为输入 dtype。
- dtype 约束：仅支持 `float16/bfloat16/float32`，complex 与其他 dtype 抛出明确错误。
- 非连续处理：输入非连续时先 `contiguous()`，计算后 `copy_` 回原张量或 out。
- out 约束：要求 out 的 shape 与 dtype 与输入一致，否则报错。

## 优化思路

- Autotune：内核启用多组 `BLOCK_SIZE/num_warps/num_stages`，按 `n_elements` 与 `dtype_size` 自动选择配置。
- 近似计算：对 fp16/bf16 启用快速近似 log（基于指数与尾数分解的多项式近似），fp32 使用 `tl.log` 保证精度。
- 统一内核：log10 / log10_ / log10.out 共用同一个 Triton kernel，减少重复开销。

## 使用方法

```python
import torch
import flag_gems

x = torch.rand((1024, 1024), dtype=torch.float16, device=flag_gems.device)

with flag_gems.use_gems():
    y = torch.log10(x)           # 非原位
    torch.log10_(x)              # 原位
    out = torch.empty_like(x)
    torch.log10(x, out=out)      # out 参数
```

注意事项：
- 仅支持 float16/bfloat16/float32。
- 输入为 0/负数时，结果为 -inf/nan，行为与 PyTorch 对齐。

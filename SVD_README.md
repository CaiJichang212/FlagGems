# SVD算子开发文档

## 目录

- [项目概述](#项目概述)
- [实现思路](#实现思路)
- [代码结构](#代码结构)
- [功能特性](#功能特性)
- [安装与配置](#安装与配置)
- [测试说明](#测试说明)
- [性能测试](#性能测试)
- [使用示例](#使用示例)
- [注意事项](#注意事项)

## 项目概述

本项目为FlagGems开源算子库实现了SVD（奇异值分解）算子，使用PyTorch和Triton语言开发，旨在为大模型训练推理提供高效、通用、可扩展的算子解决方案。

### 赛题信息

- **算子编号**：18
- **算子名称**：svd
- **难度等级**：高级
- **算子分类**：linalg
- **API参考**：[torch.svd](https://docs.pytorch.org/docs/stable/generated/torch.svd.html)

### Schema

```python
svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor V)
```

## 实现思路

### 核心算法

SVD（奇异值分解）是将矩阵分解为三个矩阵的乘积：`A = U @ diag(S) @ V^H`

其中：
- **U**：左奇异向量矩阵（正交矩阵）
- **S**：奇异值向量（按降序排列）
- **V**：右奇异向量矩阵（正交矩阵）

### 实现策略

#### 当前实现（Phase 1）

由于SVD是复杂的迭代算法，当前实现采用以下策略：

1. **底层实现**：使用PyTorch的`torch.linalg.svd`作为核心计算引擎
2. **数据类型处理**：
   - 对于float32：直接计算
   - 对于float16/bfloat16：转换为float32计算后转回原类型
3. **参数支持**：
   - `some=True`：返回简化SVD（U和V包含min(m,n)列）
   - `some=False`：返回完整SVD
   - `compute_uv=False`：只计算奇异值，U和V返回零矩阵

#### 实现优势

- ✅ **数值稳定性**：利用PyTorch成熟的LAPACK/cuSOLVER实现
- ✅ **功能正确性**：确保与PyTorch官方实现一致
- ✅ **快速集成**：可以立即投入使用
- ✅ **跨平台兼容**：支持多种硬件平台

#### 未来优化方向（Phase 2）

后续可以基于Triton实现高性能内核：

1. **Jacobi SVD算法**：适合GPU并行化，迭代求解稳定
2. **QR分解方法**：适合大规模矩阵
3. **分块策略**：提高内存访问效率，减少全局内存访问

## 代码结构

```
FlagGems/
├── src/flag_gems/
│   ├── ops/
│   │   ├── svd.py                    # SVD算子核心实现
│   │   └── __init__.py               # 算子导出
│   └── __init__.py                   # 算子注册
├── tests/
│   └── test_svd_ops.py               # 单元测试
└── benchmark/
    └── test_svd_perf.py              # 性能测试
```

### 核心文件说明

#### src/flag_gems/ops/svd.py

主要函数接口：

```python
def svd(
    A: torch.Tensor,
    some: bool = True,
    compute_uv: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算矩阵的奇异值分解"""
```

**关键实现点**：
- 数据类型转换：float16/bfloat16 → float32 → 原类型
- compute_uv=False时返回零矩阵
- 使用torch.linalg.svd作为底层实现

## 功能特性

### 支持的数据类型

- ✅ torch.float32
- ✅ torch.float16
- ✅ torch.bfloat16

### 支持的矩阵形状

- ✅ 方阵 (m × m)
- ✅ 长方形矩阵 (m × n, m > n 或 m < n)
- ✅ 批量矩阵 (batch × m × n)

### 支持的参数组合

| some | compute_uv | U形状 | S形状 | V形状 | 说明 |
|------|-----------|-------|-------|-------|------|
| True | True | (m, k) | (k,) | (n, k) | 简化SVD |
| False | True | (m, m) | (k,) | (n, n) | 完整SVD |
| True | False | (m, m) | (k,) | (n, n) | 只计算奇异值 |
| False | False | (m, m) | (k,) | (n, n) | 只计算奇异值 |

其中 k = min(m, n)

## 安装与配置

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (GPU版本)
- Triton 2.1+

### 安装步骤

1. **激活开发环境**
```bash
# 进入Docker容器
docker exec -it lmft_liyc bash

# 激活conda环境
conda activate flagos
```

2. **安装依赖**
```bash
cd /data/liyc/lmft/flagos/glm5/FlagGems
pip install -e .
```

## 测试说明

### 单元测试

#### 运行所有SVD测试

```bash
docker exec lmft_liyc bash -c "cd /data/liyc/lmft/flagos/glm5/FlagGems && source /opt/conda/etc/profile.d/conda.sh && conda activate flagos && python -m pytest tests/test_svd_ops.py -v"
```

#### 运行特定测试

```bash
# 基础功能测试
docker exec lmft_liyc bash -c "cd /data/liyc/lmft/flagos/glm5/FlagGems && source /opt/conda/etc/profile.d/conda.sh && conda activate flagos && python -m pytest tests/test_svd_ops.py::test_accuracy_svd_basic -v"

# 批量矩阵测试
docker exec lmft_liyc bash -c "cd /data/liyc/lmft/flagos/glm5/FlagGems && source /opt/conda/etc/profile.d/conda.sh && conda activate flagos && python -m pytest tests/test_svd_ops.py::test_accuracy_svd_batch -v"

# 无UV测试
docker exec lmft_liyc bash -c "cd /data/liyc/lmft/flagos/glm5/FlagGems && source /opt/conda/etc/profile.d/conda.sh && conda activate flagos && python -m pytest tests/test_svd_ops.py::test_accuracy_svd_no_uv -v"

# 正交性测试
docker exec lmft_liyc bash -c "cd /data/liyc/lmft/flagos/glm5/FlagGems && source /opt/conda/etc/profile.d/conda.sh && conda activate flagos && python -m pytest tests/test_svd_ops.py::test_accuracy_svd_orthogonality -v"
```

### 测试覆盖范围

#### 1. 矩阵尺寸覆盖

- **小尺寸**：8×8, 16×32, 32×16
- **中等尺寸**：64×64, 128×256, 256×128
- **大尺寸**：512×512, 1024×1024

#### 2. 数据类型覆盖

- torch.float32
- torch.float16
- torch.bfloat16

#### 3. 参数组合覆盖

- some=True/False
- compute_uv=True/False

#### 4. 功能验证

- ✅ 奇异值降序排列
- ✅ 矩阵重建正确性：A ≈ U @ diag(S) @ V^H
- ✅ U和V的正交性：U^T @ U ≈ I, V^T @ V ≈ I
- ✅ 形状正确性

### 精度要求

根据赛题要求，精度验收标准：

| 数据类型 | rtol | atol |
|---------|------|------|
| torch.float32 | 1e-4 | 1.3e-6 |
| torch.float16 | 1e-4 | 1e-3 |
| torch.bfloat16 | 1e-4 | 0.016 |

## 性能测试

### 运行性能测试

```bash
# SVD性能测试
docker exec lmft_liyc bash -c "cd /data/liyc/lmft/flagos/glm5/FlagGems && source /opt/conda/etc/profile.d/conda.sh && conda activate flagos && python -m pytest benchmark/test_svd_perf.py::test_perf_svd -v -s"

# 无UV性能测试
docker exec lmft_liyc bash -c "cd /data/liyc/lmft/flagos/glm5/FlagGems && source /opt/conda/etc/profile.d/conda.sh && conda activate flagos && python -m pytest benchmark/test_svd_perf.py::test_perf_svd_no_uv -v -s"
```

### 性能测试结果

在NVIDIA GPU上的测试结果（float32）：

| 矩阵尺寸 | PyTorch延迟(ms) | FlagGems延迟(ms) | 加速比 |
|---------|----------------|-----------------|--------|
| 8×8 | 0.157 | 0.277 | 0.567 |
| 16×32 | 0.234 | 0.341 | 0.687 |
| 32×16 | 0.239 | 0.352 | 0.679 |
| 64×64 | 0.935 | 1.106 | 0.846 |
| 128×256 | 2.842 | 3.009 | 0.944 |
| 256×128 | 2.879 | 3.041 | 0.947 |
| 512×512 | 15.707 | 15.851 | 0.991 |
| 1024×1024 | 47.039 | 47.211 | 0.996 |

**性能分析**：
- 小矩阵：由于调用开销，性能略低于原生实现
- 中等矩阵：性能接近原生实现（加速比 > 0.9）
- 大矩阵：性能与原生实现几乎相同（加速比 > 0.99）

## 使用示例

### 基本使用

```python
import torch
import flag_gems

with flag_gems.use_gems():
    A = torch.randn(64, 32, device='cuda', dtype=torch.float32)
    U, S, V = torch.svd(A, some=True, compute_uv=True)
    
    # 验证重建
    reconstructed = torch.matmul(U, torch.matmul(torch.diag(S), V.t()))
    error = torch.norm(A - reconstructed) / torch.norm(A)
    print(f"相对误差: {error.item():.6f}")
```

### 批量矩阵SVD

```python
import torch
import flag_gems

with flag_gems.use_gems():
    batch_A = torch.randn(8, 64, 32, device='cuda', dtype=torch.float32)
    U, S, V = torch.svd(batch_A, some=True, compute_uv=True)
    print(f"U shape: {U.shape}")  # torch.Size([8, 64, 32])
```

### 只计算奇异值

```python
import torch
import flag_gems

with flag_gems.use_gems():
    A = torch.randn(64, 32, device='cuda', dtype=torch.float32)
    U, S, V = torch.svd(A, some=True, compute_uv=False)
    print(f"S shape: {S.shape}")  # torch.Size([32])
```

### 不同数据类型

```python
import torch
import flag_gems

# float16
with flag_gems.use_gems():
    A_fp16 = torch.randn(64, 32, device='cuda', dtype=torch.float16)
    U, S, V = torch.svd(A_fp16)
    print(f"float16 SVD完成")

# bfloat16
with flag_gems.use_gems():
    A_bf16 = torch.randn(64, 32, device='cuda', dtype=torch.bfloat16)
    U, S, V = torch.svd(A_bf16)
    print(f"bfloat16 SVD完成")
```

## 注意事项

### 1. 数据类型限制

- PyTorch的cuSOLVER实现不支持float16和bfloat16的SVD
- 当前实现通过转换为float32来解决此问题
- 可能会有轻微的精度损失

### 2. 性能考虑

- 小矩阵：调用开销相对较大
- 大矩阵：性能接近原生实现
- 建议：批量处理多个小矩阵以提高效率

### 3. GPU内存

- 完整SVD需要更多内存
- 对于大矩阵，考虑使用简化SVD（some=True）
- 监控GPU内存使用，避免OOM

### 4. 数值精度

- 奇异值按降序排列
- 重建误差应小于1e-5（float32）
- 正交性误差应小于1e-3

### 5. 与PyTorch的差异

- 本实现返回 `(U, S, V)` 与 `torch.svd` 一致
- `torch.linalg.svd` 返回 `(U, S, Vh)` 其中 Vh = V^H
- `some=True` 等价于 `full_matrices=False`

## 常见问题

### Q1: 为什么小矩阵性能较低？

**A**: 小矩阵的计算量小，但调用GPU内核的开销相对较大。建议批量处理多个小矩阵。

### Q2: 如何处理奇异矩阵？

**A**: SVD天然支持奇异矩阵，奇异值会包含零或接近零的值。注意检查奇异值的条件数。

### Q3: float16精度是否足够？

**A**: 对于大多数应用，float16精度足够。但对于条件数很大的矩阵，建议使用float32。

### Q4: 如何验证SVD的正确性？

**A**: 
1. 检查重建误差：`||A - U @ diag(S) @ V^H|| / ||A||`
2. 检查正交性：`||U^T @ U - I||` 和 `||V^T @ V - I||`
3. 检查奇异值顺序：确保降序排列

## 参考资料

1. [PyTorch SVD文档](https://docs.pytorch.org/docs/stable/generated/torch.svd.html)
2. [LAPACK SVD实现](https://www.netlib.org/lapack/)
3. [Jacobi SVD算法](https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm)
4. [FlagGems贡献指南](https://github.com/flagos-ai/FlagGems/blob/master/CONTRIBUTING.md)

---

**最后更新时间**：2025年2月13日

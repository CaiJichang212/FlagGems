"""FlagGems 的 log10 点算子实现。

本文件实现了与 PyTorch 对齐的三个接口：
1. log10(Tensor self) -> Tensor
2. log10_(Tensor(a!) self) -> Tensor(a!)
3. log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

实现基于 Triton + pointwise_dynamic：
- Triton 负责逐元素数学计算；
- pointwise_dynamic 负责广播、输出分配、out 参数检查、inplace/out 绑定等通用逻辑。
"""

import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

# log10(x) = ln(x) * (1 / ln(10))
# 抽成常量便于维护，也避免在 Kernel 内重复写魔法数字。
LOG10E = 0.4342944819032518


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def log10_func(x):
    """Triton 标量函数：执行逐元素 log10 计算。

    Args:
        x: 输入元素（由 pointwise_dynamic 生成的 Kernel 在运行时传入）。

    Returns:
        逐元素的 log10 结果（在 fp32 中间精度计算后返回）。

    Notes:
        - 这里采用 fp32 中间精度计算，可提升 fp16/bf16 输入下的数值稳定性；
        - `pointwise_dynamic` 会在外围处理输出 dtype、广播和 out/inplace 行为。
    """
    # 先转 fp32 再计算 ln，最后乘常量得到 log10。
    return tl.log(x.to(tl.float32)) * LOG10E


def log10(A):
    """计算输入张量的常用对数（底数为 10），返回新张量。

    Args:
        A: 输入张量。

    Returns:
        与输入形状广播一致的新输出张量。
    """
    logger.debug("GEMS LOG10")
    return log10_func(A)


def log10_(A):
    """原位计算输入张量的常用对数（底数为 10）。

    Args:
        A: 输入张量，同时也是输出张量（原位写回）。

    Returns:
        原位更新后的输入张量 A。

    Notes:
        `out0=A` 是 pointwise_dynamic 的约定写法，表示将第 0 个输出绑定到 A。
    """
    logger.debug("GEMS LOG10_")
    return log10_func(A, out0=A)


# log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def log10_out(A, out):
    """将 log10 结果写入指定的 out 张量。

    Args:
        A: 输入张量。
        out: 由调用方提供的输出张量。

    Returns:
        写入结果后的 out 张量。

    Notes:
        `pointwise_dynamic` 负责校验 out 的形状合法性与写回路径。
    """
    logger.debug("GEMS LOG10_OUT")
    return log10_func(A, out0=out)

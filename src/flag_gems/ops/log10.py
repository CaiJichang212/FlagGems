import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_INV_LN10 = 0.4342944819032518  # 1 / ln(10)
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _check_log10_input(x, op_name):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{op_name} expects a torch.Tensor as input.")
    if x.is_complex():
        raise TypeError(f"{op_name} does not support complex dtypes.")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"{op_name} supports dtypes {_SUPPORTED_DTYPES}, but got {x.dtype}."
        )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
    ],
    key=["n_elements", "dtype_size"],
)
@triton.jit
def _log10_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    dtype_size,
    BLOCK_SIZE: tl.constexpr,
    USE_APPROX: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_fp32 = x.to(tl.float32)
    if USE_APPROX:
        pos_mask = x_fp32 > 0
        zero_mask = x_fp32 == 0
        ix = x_fp32.to(tl.int32, bitcast=True)
        exp = (ix >> 23) & 0xFF
        mant = (ix & 0x7FFFFF) | 0x3F800000
        m = mant.to(tl.float32, bitcast=True)
        k = exp.to(tl.int32) - 127
        t = (m - 1.0) / (m + 1.0)
        t2 = t * t
        log_m = 2.0 * (
            t + t2 * t * (1.0 / 3.0 + t2 * (1.0 / 5.0 + t2 * (1.0 / 7.0)))
        )
        log_val = log_m + k.to(tl.float32) * 0.6931471805599453
        nan_or_inf = tl.where(zero_mask, -float("inf"), float("nan"))
        y_fp32 = tl.where(pos_mask, log_val, nan_or_inf)
    else:
        y_fp32 = tl.log(x_fp32)
    y_fp32 = y_fp32 * _INV_LN10
    y = y_fp32.to(x.dtype)
    tl.store(out_ptr + offsets, y, mask=mask)


def _launch_log10(x, out):
    n_elements = out.numel()
    if n_elements == 0:
        return out
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    dtype_size = x.element_size()
    _log10_kernel[grid](
        x,
        out,
        n_elements,
        dtype_size,
        USE_APPROX=dtype_size == 2,
    )
    return out


def log10(A):
    logger.debug("GEMS LOG10")
    _check_log10_input(A, "log10")
    out = torch.empty_like(A)
    if A.is_contiguous() and out.is_contiguous():
        return _launch_log10(A, out)
    buf = A.contiguous()
    out_buf = torch.empty_like(buf)
    _launch_log10(buf, out_buf)
    out.copy_(out_buf)
    return out


def log10_(A):
    logger.debug("GEMS LOG10_")
    _check_log10_input(A, "log10_")
    if A.numel() == 0:
        return A
    if A.is_contiguous():
        _launch_log10(A, A)
        return A
    buf = A.contiguous()
    _launch_log10(buf, buf)
    A.copy_(buf)
    return A


# log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def log10_out(A, out):
    logger.debug("GEMS LOG10_OUT")
    _check_log10_input(A, "log10.out")
    if not isinstance(out, torch.Tensor):
        raise TypeError("log10.out expects 'out' to be a torch.Tensor.")
    if out.dtype != A.dtype:
        raise TypeError(
            f"log10.out expects out.dtype == input dtype ({A.dtype}), got {out.dtype}."
        )
    if out.shape != A.shape:
        raise RuntimeError(
            f"log10.out expects out.shape == input shape {A.shape}, got {out.shape}."
        )
    if A.numel() == 0:
        return out
    if A.is_contiguous() and out.is_contiguous():
        return _launch_log10(A, out)
    buf = A.contiguous()
    out_buf = torch.empty_like(buf)
    _launch_log10(buf, out_buf)
    out.copy_(out_buf)
    return out

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
    ],
    key=["n_elements", "DTYPE_ID"],
)
@triton.jit
def logaddexp_kernel(
    input_ptr,
    other_ptr,
    out_ptr,
    n_elements,
    OUT_DTYPE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load inputs and upcast to fp32 for numerical stability
    x = tl.load(input_ptr + offs, mask=mask, other=0).to(tl.float32)
    y = tl.load(other_ptr + offs, mask=mask, other=0).to(tl.float32)

    # logaddexp(x, y) = log(exp(x) + exp(y))
    # For numerical stability, use:
    # m = max(x, y)
    # logaddexp(x, y) = m + log(exp(x-m) + exp(y-m))
    # logaddexp(x, y) = m + log(1 + exp(-|x-y|))
    m = tl.maximum(x, y)
    diff = tl.abs(x - y)
    res = m + tl.log(1.0 + tl.exp(-diff))

    # Handle the cases where x == y (including +/-inf)
    # x - y is nan if x == y == inf or x == y == -inf
    res = tl.where(x == y, x + 0.6931471805599453, res)

    tl.store(out_ptr + offs, res.to(OUT_DTYPE), mask=mask)


def _to_triton_dtype(dtype):
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    return None


def _broadcast_and_check(input, other):
    if not isinstance(input, torch.Tensor):
        input = torch.as_tensor(input)
    if not isinstance(other, torch.Tensor):
        other = torch.as_tensor(other)
    return torch.broadcast_tensors(input, other)


def _choose_out_dtype(input: torch.Tensor, other: torch.Tensor, out: torch.Tensor = None):
    if out is not None:
        return out.dtype
    float_priority = [torch.float64, torch.float32, torch.bfloat16, torch.float16]
    for dt in float_priority:
        if input.dtype == dt or other.dtype == dt:
            return dt
    return torch.get_default_dtype()


def _launch_kernel(input_c, other_c, out_c, out_dtype):
    n_elements = out_c.numel()
    if n_elements == 0:
        return
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    triton_dtype = _to_triton_dtype(out_dtype)
    dtype_id = 0 if out_dtype == torch.float16 else 1 if out_dtype == torch.bfloat16 else 2
    logaddexp_kernel[grid](
        input_c,
        other_c,
        out_c,
        n_elements,
        OUT_DTYPE=triton_dtype,
        DTYPE_ID=dtype_id,
    )


def logaddexp(input: torch.Tensor, other: torch.Tensor):
    """
    Logarithm of the sum of exponentiations of the inputs.
    """
    b_input, b_other = _broadcast_and_check(input, other)
    if (
        b_input.device.type != "cuda"
        or b_other.device.type != "cuda"
        or b_input.device != b_other.device
        or b_input.is_complex()
        or b_other.is_complex()
    ):
        return torch.ops.aten.logaddexp(b_input, b_other)

    out_dtype = _choose_out_dtype(b_input, b_other)
    if _to_triton_dtype(out_dtype) is None:
        return torch.ops.aten.logaddexp(b_input, b_other)
    out = torch.empty(b_input.shape, device=b_input.device, dtype=out_dtype)

    input_c = b_input.contiguous().view(-1)
    other_c = b_other.contiguous().view(-1)
    out_c = out.contiguous().view(-1)
    _launch_kernel(input_c, other_c, out_c, out_dtype)
    return out


def logaddexp_out(input: torch.Tensor, other: torch.Tensor, out: torch.Tensor):
    """
    Logarithm of the sum of exponentiations of the inputs, with output tensor.
    """
    if out is None:
        raise ValueError("out tensor must be provided for logaddexp_out")

    b_input, b_other = _broadcast_and_check(input, other)
    if (
        out.device.type != "cuda"
        or b_input.device.type != "cuda"
        or b_other.device.type != "cuda"
        or not (b_input.device == b_other.device == out.device)
        or b_input.is_complex()
        or b_other.is_complex()
        or out.is_complex()
    ):
        return torch.ops.aten.logaddexp.out(b_input, b_other, out=out)
    if _to_triton_dtype(out.dtype) is None:
        return torch.ops.aten.logaddexp.out(b_input, b_other, out=out)

    if out.shape != b_input.shape:
        raise ValueError(
            f"out tensor has shape {out.shape}, expected {b_input.shape} from broadcast"
        )

    input_c = b_input.contiguous().view(-1)
    other_c = b_other.contiguous().view(-1)

    if out.is_contiguous():
        out_c = out.view(-1)
        _launch_kernel(input_c, other_c, out_c, out.dtype)
        return out
    else:
        tmp = torch.empty_like(out, memory_format=torch.contiguous_format)
        out_c = tmp.view(-1)
        _launch_kernel(input_c, other_c, out_c, out.dtype)
        out.copy_(tmp)
        return out

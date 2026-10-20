# Copyright 2026 FlagOS Contributors. Licensed under Apache-2.0.
"""Private BF16 decode O projection: eight disjoint FP32 K partials.

No atomics, persistent workspace, weight transform or retry after launch.
The caller guards the actual unquantized RowParallelLinear dispatch and phase.
"""
import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn


@triton.jit
def partial_kernel(X, W, P, M: tl.constexpr,
                   BM: tl.constexpr = 16, BN: tl.constexpr = 32,
                   BK: tl.constexpr = 32):
    r = tl.arange(0, BM)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    part = tl.program_id(1)
    k = part * 256 + tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(8):
        a = tl.load(X + r[:, None] * 2048 + k[None, :], r[:, None] < M, other=0.)
        b = tl.load(W + n[None, :] * 2048 + k[:, None])
        acc += tl.dot(a, b, allow_tf32=False)
        k += BK
    tl.store(P + (part * M + r[:, None]) * 2048 + n[None, :], acc, r[:, None] < M)


@triton.jit
def reduce_kernel(P, O, M: tl.constexpr, BLOCK: tl.constexpr = 256):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0., tl.float32)
    for part in range(8):
        value += tl.load(P + part * M * 2048 + i, i < M * 2048, other=0.)
    tl.store(O + i, value.to(tl.bfloat16), i < M * 2048)


def supported(x, weight):
    if type(x) is not torch.Tensor: return False
    if not (type(weight) is torch.Tensor or isinstance(weight, torch.nn.Parameter)): return False
    if torch.is_grad_enabled() or not torch.is_inference_mode_enabled() or torch.compiler.is_compiling(): return False
    if x.requires_grad or weight.requires_grad: return False
    if x.device.type != 'cuda' or weight.device != x.device: return False
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16: return False
    if x.ndim != 2 or not 1 <= x.shape[0] <= 4 or x.shape[1] != 2048: return False
    if tuple(weight.shape) != (2048, 2048): return False
    if not x.is_contiguous() or not weight.is_contiguous(): return False
    return True


def try_decode_o(x, weight, *, phase, layer):
    """Return None only before any submission; fresh outputs own all writes."""
    if phase != 'decode' or not supported(x, weight): return None
    m = x.shape[0]
    partial = torch.empty((8, m, 2048), device=x.device, dtype=torch.float32)
    out = torch.empty((m, 2048), device=x.device, dtype=x.dtype)
    with torch_device_fn.device(x.device):
        partial_kernel[(64, 8)](x, weight, partial, m, num_warps=4)
        reduce_kernel[(triton.cdiv(m * 2048, 256),)](partial, out, m, num_warps=4)
    return out

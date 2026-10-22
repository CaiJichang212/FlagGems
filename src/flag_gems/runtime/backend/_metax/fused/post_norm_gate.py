# SPDX-License-Identifier: Apache-2.0
"""Read-only M4 residual RMS + paired gate/up. Private MetaX experiment."""
import math
import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn
from .gate_up_silu import supported, _baseline_record


@triton.jit
def post_norm_gate_kernel(X, R, NW, W, O, NR, EPS: tl.constexpr,
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                         BLOCK_SIZE_K: tl.constexpr):
    rows = tl.arange(0, 4)
    cols = tl.arange(0, 2048)
    z = tl.load(X + rows[:, None]*2048 + cols[None, :]).to(tl.float32)
    z += tl.load(R + rows[:, None]*2048 + cols[None, :]).to(tl.float32)
    var = tl.sum(z*z/2048, axis=1)
    rrms = 1 / tl.sqrt(var + EPS)
    nw = tl.load(NW + cols)
    normalized = (z*rrms[:, None]*nw[None, :]).to(tl.bfloat16)
    if tl.program_id(1) == 0:
        tl.store(NR + rows[:, None]*2048 + cols[None, :], z)
    rm = tl.arange(0, BLOCK_SIZE_M)
    lane = tl.arange(0, BLOCK_SIZE_N)
    rn = tl.program_id(1)*(BLOCK_SIZE_N//2) + lane//2
    wn = rn + (lane%2)*6144
    rk = tl.arange(0, BLOCK_SIZE_K)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32)
    for k in range(0, tl.cdiv(2048, BLOCK_SIZE_K)):
        ix = tl.broadcast_to((rk + k*BLOCK_SIZE_K)[None, :], (4, BLOCK_SIZE_K))
        small = tl.gather(normalized, ix, axis=1)
        ir = tl.broadcast_to(tl.minimum(rm, 3)[:, None], (BLOCK_SIZE_M, BLOCK_SIZE_K))
        a = tl.gather(small, ir, axis=0)
        a = tl.where(rm[:, None] < 4, a, 0.).to(tl.bfloat16)
        w = tl.load(W+(rk[:, None]+k*BLOCK_SIZE_K)+wn[None, :]*2048,
                    (rk[:, None]+k*BLOCK_SIZE_K < 2048)&(rn[None, :]<6144), other=0.)
        acc += tl.dot(a, w, allow_tf32=False)
    rounded = acc.to(tl.bfloat16)
    gate, up = tl.split(tl.reshape(rounded, (BLOCK_SIZE_M, BLOCK_SIZE_N//2, 2)))
    gf = gate.to(tl.float32)
    silu = tl.fdiv(gf, 1.0+tl.exp(-gf))
    result = (silu*up.to(tl.float32)).to(tl.bfloat16)
    oc = tl.program_id(1)*(BLOCK_SIZE_N//2)+tl.arange(0, BLOCK_SIZE_N//2)
    tl.store(O+rm[:, None]*6144+oc[None, :], result, (rm[:, None]<4)&(oc[None, :]<6144))


def eligible(x, residual, norm_weight, weight, eps):
    if not supported(x, weight) or tuple(x.shape) != (4, 2048):
        return False
    if (type(residual) is not torch.Tensor or tuple(residual.shape) != (4, 2048)
            or not (type(norm_weight) is torch.Tensor or isinstance(norm_weight, torch.nn.Parameter))
            or tuple(norm_weight.shape) != (2048,) or type(eps) is not float
            or not math.isfinite(eps) or eps < 0):
        return False
    tensors = (x, residual, norm_weight, weight)
    if any(t.device != x.device or t.dtype != x.dtype or not t.is_contiguous()
           or t.data_ptr()%16 or (t.requires_grad and torch.is_grad_enabled()) for t in tensors):
        return False
    intervals = sorted((t.data_ptr(), t.data_ptr()+t.numel()*t.element_size()) for t in tensors)
    return all(a[1] <= b[0] for a,b in zip(intervals, intervals[1:]))


def try_post_norm_gate(x, residual, norm_weight, weight, eps):
    if not eligible(x, residual, norm_weight, weight, eps):
        return None
    # Query exactly the existing gate/up configuration; no independent tuning.
    meta = _baseline_record(x, weight)
    if meta is None or 2048 % meta['BLOCK_SIZE_K'] or meta['BLOCK_SIZE_M'] < 4:
        return None
    out = torch.empty((4,6144), device=x.device, dtype=x.dtype)
    new_residual = torch.empty_like(residual)
    grid = (1, triton.cdiv(6144,meta['BLOCK_SIZE_N']//2))
    with torch_device_fn.device(x.device):
        post_norm_gate_kernel[grid](x,residual,norm_weight,weight,out,new_residual,eps,**meta)
    # No retry after submission; exceptions propagate, inputs stay read-only.
    return out, new_residual

# Copyright 2026 FlagOS Contributors. Licensed under Apache-2.0.
"""Experimental paired-column gate/up GEMM; public linear/SiLU stay unchanged."""
import importlib

import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn

MAX_ROWS = 1024


@triton.jit
def paired_gate_up_kernel(X, W, O, M, K: tl.constexpr, H: tl.constexpr,
                          BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                          BLOCK_SIZE_K: tl.constexpr, FUSED: tl.constexpr):
    rm = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Adjacent logical lanes hold gate/up for one output. Total dot width unchanged.
    lane = tl.arange(0, BLOCK_SIZE_N)
    rn = tl.program_id(1) * (BLOCK_SIZE_N // 2) + lane // 2
    wn = rn + (lane % 2) * H
    rk = tl.arange(0, BLOCK_SIZE_K)
    xp = X + rm[:, None] * K + rk[None, :]
    wp = W + rk[:, None] + wn[None, :] * K
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(xp, (rm[:, None] < M) & (rk[None, :] < K-k*BLOCK_SIZE_K), other=0.)
        b = tl.load(wp, (rk[:, None] < K-k*BLOCK_SIZE_K) & (rn[None, :] < H), other=0.)
        acc += tl.dot(a, b, allow_tf32=False)
        xp += BLOCK_SIZE_K
        wp += BLOCK_SIZE_K
    rounded = acc.to(tl.bfloat16)
    if FUSED:
        gate, up = tl.split(tl.reshape(rounded, (BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2)))
        gf = gate.to(tl.float32)
        silu = tl.fdiv(gf, 1.0 + tl.exp(-gf))
        result = (silu * up.to(tl.float32)).to(tl.bfloat16)
        col = tl.program_id(1) * (BLOCK_SIZE_N // 2) + tl.arange(0, BLOCK_SIZE_N // 2)
        tl.store(O + rm[:, None] * H + col[None, :], result,
                 (rm[:, None] < M) & (col[None, :] < H))
    else:
        tl.store(O + rm[:, None] * (2*H) + wn[None, :], rounded,
                 (rm[:, None] < M) & (rn[None, :] < H))


def supported(x, weight):
    return (type(x) is torch.Tensor
            and (type(weight) is torch.Tensor or isinstance(weight, torch.nn.Parameter))
            and torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
            and not x.requires_grad and not weight.requires_grad
            and x.device.type == 'cuda' and x.device == weight.device
            and x.dtype == weight.dtype == torch.bfloat16
            and x.ndim == weight.ndim == 2 and tuple(weight.shape) == (12288, 2048)
            and 1 <= x.shape[0] <= MAX_ROWS and x.shape[1] == 2048
            and x.is_contiguous() and weight.is_contiguous()
            and x.data_ptr() % 16 == weight.data_ptr() % 16 == 0
            and not torch.compiler.is_compiling())


def _baseline_record(x, weight):
    """Use U's actual selected launch metadata, including warmed LibEntry identity.

    First call warms U (outside timing windows); no independent tile tuning occurs.
    Later calls reuse U's record, never input/weight values or pointers.
    """
    generic = importlib.import_module('flag_gems.ops.linear')
    metax = importlib.import_module('_metax.ops.linear')
    import flag_gems
    if flag_gems.linear is not metax.linear:
        return None
    entry = generic.linear_kernel
    if triton.__version__ != '3.0.0' or entry._has_flagtune_tuner:
        return None
    # Key creation does not depend on data values. Allocate the output only on cold
    # lookup to match U's pointer specialization and preserve compiled-cache changes.
    cache = metax._cache(x.device.index)
    metadata = metax._metadata(x, weight, None)
    if metadata is None:
        return None
    xs, ws, device, dtype, xstride, wstride, xa, wa = metadata
    signature = (entry, entry.jit_function, xs, ws, dtype, xstride, wstride, xa, wa)
    record = cache.get(signature)
    if record is None or entry.kernel_cache[x.device.index].get(record[0]) is not record[1]:
        metax.linear(x, weight)
        record = cache.get(signature)
    if record is None or record[1][4]:
        return None
    meta = record[1][1]
    required = ('BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'num_warps', 'num_stages')
    if any(k not in meta for k in required) or meta['BLOCK_SIZE_N'] < 2:
        return None
    return {k: meta[k] for k in required}


def try_gate_up_silu(x, weight):
    if not supported(x, weight):
        return None
    meta = _baseline_record(x, weight)
    if meta is None:
        return None
    out = torch.empty((x.shape[0], 6144), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(x.shape[0], meta['BLOCK_SIZE_M']),
            triton.cdiv(6144, meta['BLOCK_SIZE_N'] // 2))
    with torch_device_fn.device(x.device):
        paired_gate_up_kernel[grid](x, weight, out, x.shape[0], 2048, 6144,
                                   FUSED=True, **meta)
    return out


def paired_intermediate(x, weight):
    """Diagnostic control: paired GEMM writes original [M,12288] intermediate."""
    if not supported(x, weight):
        return None
    meta = _baseline_record(x, weight)
    if meta is None:
        return None
    out = torch.empty((x.shape[0], 12288), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(x.shape[0], meta['BLOCK_SIZE_M']),
            triton.cdiv(6144, meta['BLOCK_SIZE_N'] // 2))
    with torch_device_fn.device(x.device):
        paired_gate_up_kernel[grid](x, weight, out, x.shape[0], 2048, 6144,
                                   FUSED=False, **meta)
    return out

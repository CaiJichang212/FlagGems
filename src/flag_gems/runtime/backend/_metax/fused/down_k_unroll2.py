# Copyright 2026 FlagOS Contributors. Licensed under Apache-2.0.
"""M4 down projection: two ordered K steps per loop, original selected tiles."""
from collections import OrderedDict
import os
import threading

import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn
from . import gate_up_silu as launch


@triton.jit
def down_unroll2_kernel(X, W, O, M, K: tl.constexpr, N: tl.constexpr,
                        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                        BLOCK_SIZE_K: tl.constexpr):
    rm = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)
    xp = X + rm[:, None] * K + rk[None, :]
    wp = W + rk[:, None] + rn[None, :] * K
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32)
    for group in range(0, tl.cdiv(K, 2 * BLOCK_SIZE_K)):
        for inner in tl.static_range(0, 2):
            k = group * 2 + inner
            a = tl.load(xp, (rm[:, None] < M) & (rk[None, :] < K-k*BLOCK_SIZE_K), other=0.)
            b = tl.load(wp, (rk[:, None] < K-k*BLOCK_SIZE_K) & (rn[None, :] < N), other=0.)
            acc += tl.dot(a, b, allow_tf32=False)
            xp += BLOCK_SIZE_K
            wp += BLOCK_SIZE_K
    tl.store(O + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16),
             (rm[:, None] < M) & (rn[None, :] < N))


def supported(x, weight):
    return (type(x) is torch.Tensor
            and (type(weight) is torch.Tensor or isinstance(weight, torch.nn.Parameter))
            and torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
            and not x.requires_grad and not weight.requires_grad
            and x.device.type == 'cuda' and x.device == weight.device
            and x.dtype == weight.dtype == torch.bfloat16
            and tuple(x.shape) == (4, 6144) and tuple(weight.shape) == (2048, 6144)
            and x.is_contiguous() and weight.is_contiguous()
            and x.data_ptr() % 16 == weight.data_ptr() % 16 == 0
            and not torch.compiler.is_compiling())


_local = threading.local()


def _cache(device):
    pid = os.getpid()
    if getattr(_local, 'pid', None) != pid:
        _local.pid, _local.devices = pid, {}
    return _local.devices.setdefault(device, OrderedDict())


def clear_cache():
    _local.devices = {}


def try_down_unroll2(x, weight):
    if not supported(x, weight):
        return None
    device = x.device.index
    jit = down_unroll2_kernel
    stamp = launch._jit_stamp(jit)
    if stamp is None:
        return None
    cache = _cache(device)
    signature = launch._signature(x, weight)
    record = cache.get(signature)
    if record is not None:
        previous, dep, compiled_cache, key, kernel, runner = record
        if (stamp == previous and launch._valid_dependency(dep, device)
                and jit.cache[device] is compiled_cache and compiled_cache.get(key) is kernel):
            out = torch.empty((4, 2048), device=x.device, dtype=x.dtype)
            if out.data_ptr() % 16 == 0:
                with torch_device_fn.device(x.device):
                    runner(x, weight, out, 4)
                cache.move_to_end(signature)
                return out
        del cache[signature]
    meta = launch._baseline_record(x, weight)
    if meta is None or 6144 % (2 * meta['BLOCK_SIZE_K']):
        return None
    out = torch.empty((4, 2048), device=x.device, dtype=x.dtype)
    if out.data_ptr() % 16:
        return None
    grid = (triton.cdiv(4, meta['BLOCK_SIZE_M']), triton.cdiv(2048, meta['BLOCK_SIZE_N']), 1)
    # Once submitted, propagate errors; never execute a second fallback launch.
    with torch_device_fn.device(x.device):
        kernel = jit[grid](x, weight, out, 4, 6144, 2048, **meta)
    if kernel is None:
        return None  # Standard compile hook suppressed execution.
    stamp = launch._jit_stamp(jit)
    dep = launch._dependency(x, weight)
    if stamp is None or dep is None:
        return out
    _, specs, constants, _, excess = jit.binder(x, weight, out, 4, 6144, 2048,
                                               **dict(meta, debug=jit.debug))
    key = ''.join(specs) + str((constants, excess))
    compiled_cache = jit.cache[device]
    if compiled_cache.get(key) is kernel:
        cache[signature] = (stamp, dep, compiled_cache, key, kernel, kernel[grid])
        cache.move_to_end(signature)
        if len(cache) > 64:
            cache.popitem(last=False)
    return out

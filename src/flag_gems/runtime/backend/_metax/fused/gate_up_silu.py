# Copyright 2026 FlagOS Contributors. Licensed under Apache-2.0.
"""Experimental paired-column gate/up GEMM; public linear/SiLU stay unchanged."""
import importlib
import os
import threading
from collections import Counter, OrderedDict

from triton.runtime.jit import JITFunction
from triton import knobs

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


# Descriptors contain launch metadata only: never tensors, pointers or streams.
_local = threading.local()
_diagnostics = False


def diagnostics(enabled=True):
    global _diagnostics
    _diagnostics = enabled
    return dict(getattr(_local, 'counts', {}))


def _count(name):
    if _diagnostics:
        if not hasattr(_local, 'counts'):
            _local.counts = Counter()
        _local.counts[name] += 1


def _cache(device):
    pid = os.getpid()
    if getattr(_local, 'pid', None) != pid:
        _local.pid, _local.devices = pid, {}
    return _local.devices.setdefault(device, OrderedDict())


def clear_cache():
    _local.devices = {}


def _jit_stamp(jit):
    # JIT.run executes these before launch; never bypass dynamic callbacks.
    required = ('pre_run_hooks', 'used_global_vals', 'fn', 'src', 'debug',
                'noinline', 'cache_key', 'binder', 'cache', 'run')
    if any(not hasattr(jit, name) for name in required):
        return None
    run = getattr(jit.run, '__func__', None)
    if run is None:
        return None
    if (triton.__version__ != '3.0.0' or jit.pre_run_hooks
            or JITFunction.cache_hook is not None
            or knobs.runtime.jit_post_compile_hook is not None):
        return None
    # Mirror JIT.run's changed-global check, including changes after compilation.
    missing = object()
    for (name, _), (value, scope) in jit.used_global_vals.items():
        if scope.get(name, missing) != value:
            return None
    return (jit, jit.fn, jit.src, jit.debug, jit.noinline, jit.cache_key,
            jit.binder, jit.cache, run)


def _signature(x, weight):
    return (tuple(x.shape), tuple(weight.shape), x.dtype, weight.dtype,
            x.stride(), weight.stride(), x.data_ptr() % 16, weight.data_ptr() % 16)


def _dependency(x, weight):
    generic = importlib.import_module('flag_gems.ops.linear')
    metax = importlib.import_module('_metax.ops.linear')
    import flag_gems
    entry = generic.linear_kernel
    metadata = metax._metadata(x, weight, None)
    if metadata is None:
        return None
    xs, ws, device, dtype, xstride, wstride, xa, wa = metadata
    signature = (entry, entry.jit_function, xs, ws, dtype, xstride, wstride, xa, wa)
    record = metax._cache(x.device.index).get(signature)
    if record is None:
        return None
    key, original = record[:2]
    return (flag_gems, generic, metax, entry, entry.jit_function,
            entry.kernel_cache, key, original, tuple(original[1].items()))


def _valid_dependency(dep, device):
    flag_gems, generic, metax, entry, jit, cache, key, original, config = dep
    return (flag_gems.linear is metax.linear and generic.linear_kernel is entry
            and entry.jit_function is jit and not entry._has_flagtune_tuner
            and entry.kernel_cache is cache and cache[device].get(key) is original
            and not original[4] and tuple(original[1].items()) == config)


def try_gate_up_silu(x, weight, *, row32=False):
    if not supported(x, weight):
        _count('fallback')
        return None
    if row32 and tuple(x.shape) != (4, 2048):
        return None
    device = x.device.index
    cache = _cache(device)
    signature = (_signature(x, weight), row32)
    jit = paired_gate_up_kernel
    stamp = _jit_stamp(jit)
    record = cache.get(signature)
    if record is not None:
        previous, dep, compiled_cache, key, kernel, runner = record
        if (stamp is not None and stamp == previous
                and _valid_dependency(dep, device)
                and jit.cache[device] is compiled_cache
                and compiled_cache.get(key) is kernel):
            out = torch.empty((x.shape[0], 6144), device=x.device, dtype=x.dtype)
            if out.data_ptr() % 16 == 0:
                with torch_device_fn.device(x.device):
                    runner(x, weight, out, x.shape[0])
                cache.move_to_end(signature)
                _count('hit')
                return out
        del cache[signature]
        _count('invalidated')
    meta = _baseline_record(x, weight)
    if meta is None:
        _count('fallback')
        return None
    if row32:
        if meta['BLOCK_SIZE_M'] != 128:
            return None
        meta = dict(meta, BLOCK_SIZE_M=32)
    out = torch.empty((x.shape[0], 6144), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(x.shape[0], meta['BLOCK_SIZE_M']),
            triton.cdiv(6144, meta['BLOCK_SIZE_N'] // 2), 1)
    # Normal JIT path launches exactly once and returns that compiled kernel.
    with torch_device_fn.device(x.device):
        kernel = jit[grid](x, weight, out, x.shape[0], 2048, 6144,
                           FUSED=True, **meta)
    _count('cold')
    if kernel is None:
        # A compile hook may deliberately suppress execution. Use full FFN fallback.
        return None
    stamp = _jit_stamp(jit)
    dep = _dependency(x, weight)
    if stamp is None or dep is None or kernel is None or out.data_ptr() % 16:
        return out
    # One cold-only binding mirrors the pinned JIT.run cache key. No extra launch.
    kwargs = dict(meta, FUSED=True, debug=jit.debug)
    _, specs, constants, _, excess = jit.binder(
        x, weight, out, x.shape[0], 2048, 6144, **kwargs)
    key = ''.join(specs) + str((constants, excess))
    compiled_cache = jit.cache[device]
    if compiled_cache.get(key) is not kernel:
        return out
    cache[signature] = (stamp, dep, compiled_cache, key, kernel, kernel[grid])
    cache.move_to_end(signature)
    if len(cache) > 64:
        cache.popitem(last=False)
        _count('evicted')
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

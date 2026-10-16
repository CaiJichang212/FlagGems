# Copyright 2026 The FlagOS Contributors. Licensed under Apache-2.0.
"""Reuse only launch preparation for the unchanged generic in-place kernel."""
import importlib
import math
import os
import threading
from collections import Counter, OrderedDict

import torch
import triton
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.libentry import LibEntry

_local = threading.local()
_observe = False
_ENTRY_RUN = LibEntry.run
_JIT_RUN = triton.runtime.JITFunction.run


def diagnostics(enabled=False):
    global _observe
    _observe = enabled
    return dict(getattr(_local, 'counts', {}))


def _count(name):
    if _observe:
        if not hasattr(_local, 'counts'):
            _local.counts = Counter()
        _local.counts[name] += 1


def _cache(device):
    pid = os.getpid()
    if getattr(_local, 'pid', None) != pid:
        _local.pid, _local.devices, _local.counts = pid, {}, Counter()
    return _local.devices.setdefault(device, OrderedDict())


def _metadata(x, residual, normalized_shape, weight, eps):
    if (type(x) is not torch.Tensor or type(residual) is not torch.Tensor
            or not (type(weight) is torch.Tensor or isinstance(weight, torch.nn.Parameter))
            or type(eps) is not float or not math.isfinite(eps) or eps < 0
            or tuple(normalized_shape) != (2048,)):
        return None
    if (len(x.shape) != 2 or x.shape[1] != 2048 or not 1 <= x.shape[0] <= 1024
            or tuple(residual.shape) != tuple(x.shape) or tuple(weight.shape) != (2048,)
            or x.device.type != 'cuda' or x.device != residual.device or x.device != weight.device
            or any(t.dtype != torch.bfloat16 or not t.is_contiguous()
                   for t in (x, residual, weight))
            or x.requires_grad or residual.requires_grad
            or (weight.requires_grad and torch.is_grad_enabled())):
        return None
    # Contiguous supported tensors occupy one interval, including offset views.
    intervals = sorted((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
                       for t in (x, residual, weight))
    if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
        return None
    return (tuple(x.shape), x.device, tuple(t.data_ptr() % 16 == 0
                                         for t in (x, residual, weight)))


def _dynamic(entry):
    jit = entry.jit_function
    runtime = getattr(getattr(triton, 'knobs', None), 'runtime', None)
    return (triton.__version__ == '3.0.0' and type(entry) is LibEntry
            and entry.fn is jit and not entry._has_flagtune_tuner
            and getattr(entry.run, '__func__', None) is _ENTRY_RUN
            and type(jit) is triton.runtime.JITFunction
            and getattr(jit.run, '__func__', None) is _JIT_RUN
            and not jit.debug and not jit.pre_run_hooks
            and jit.launch_metadata is None and jit.cache_hook is None
            and runtime is not None and runtime.jit_cache_hook is None
            and runtime.jit_post_compile_hook is None)


def _globals_valid(jit):
    missing = object()
    return all(values.get(name, missing) == old
               for (name, _), (old, values) in jit.used_global_vals.items())


def _args(x, residual, weight, eps):
    return (x, residual, weight, 2048, 1, 2048, 1, 2048, eps, 2048)


def _entry_key(entry, args):
    return entry.key([v for i, v in enumerate(args) if i in entry.specialize_indices],
                     [v for i, v in enumerate(args) if i in entry.do_not_specialize_indices],
                     [v for i, v in enumerate(args) if i not in entry.specialize_indices
                      and i not in entry.do_not_specialize_indices])


def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
    generic = importlib.import_module('flag_gems.fused.fused_add_rms_norm')
    entry = generic.fused_add_rms_norm_kernel
    metadata = _metadata(x, residual, normalized_shape, weight, eps)
    if metadata is None or not _dynamic(entry):
        _count('fallback')
        return generic.fused_add_rms_norm(x, residual, normalized_shape, weight, eps)
    shape, device, alignment = metadata
    jit = entry.jit_function
    signature = (entry, jit, jit.fn, jit.src, jit.cache_key, entry.divisibility,
                 tuple(entry.specialize_indices), tuple(entry.do_not_specialize_indices),
                 shape, alignment)
    cache = _cache(device.index)
    record = cache.get(signature)
    if record is not None:
        key, compiled_cache, original, runner, meta, binder, compile_fn, jit_cache, jit_key = record
        if (entry.kernel_cache[device.index] is compiled_cache
                and compiled_cache.get(key) is original
                and original[1] == meta and not original[2] and not original[3] and not original[4]
                and jit.binder is binder and jit.compile is compile_fn
                and jit.cache.get(device.index) is jit_cache
                and jit_cache.get(jit_key) is original[0] and _globals_valid(jit)):
            # No exception handler after launch: retrying would add residual twice.
            with torch_device_fn.device(device):
                runner(x, residual, weight, 2048, 1, 2048, 1, 2048, eps)
            cache.move_to_end(signature)
            _count('hit')
            return x, residual
        del cache[signature]
        _count('invalidated')
    _count('cold')
    result = generic.fused_add_rms_norm(x, residual, normalized_shape, weight, eps)
    # The original call has ALREADY executed. Never call it again on capture failure.
    if not _dynamic(entry) or not _globals_valid(jit):
        return result
    key = _entry_key(entry, _args(x, residual, weight, eps))
    compiled_cache = entry.kernel_cache[device.index]
    original = compiled_cache.get(key)
    if (original is None or len(original) != 5 or any(original[i] for i in (2, 3, 4))
            or result[0] is not x or result[1] is not residual):
        _count('uncached')
        return result
    kernel = original[0]
    jit_cache = jit.cache.get(device.index, {})
    jit_key = next((k for k, v in jit_cache.items() if v is kernel), None)
    if jit_key is None:
        _count('uncached')
        return result
    cache[signature] = (key, compiled_cache, original, kernel[(shape[0], 1, 1)],
                        dict(original[1]), jit.binder, jit.compile, jit_cache, jit_key)
    cache.move_to_end(signature)
    if len(cache) > 64:
        cache.popitem(last=False)
        _count('evicted')
    return result

# Copyright 2026, The FlagOS Contributors. Licensed under Apache-2.0.
"""Reuse launch preparation for generic and fixed small-M dot kernels."""
import importlib
import os
import threading
from collections import Counter, OrderedDict

import torch
import triton
from flag_gems.runtime import torch_device_fn

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


def _supported(x, w, bias):
    return (bias is None and type(x) is torch.Tensor and (type(w) is torch.Tensor or isinstance(w, torch.nn.Parameter))
            and x.device.type == 'cuda' and x.device == w.device
            and x.ndim == w.ndim == 2 and x.shape[0] > 0
            and tuple(w.shape) in ((2048, 6144), (12288, 2048))
            and x.shape[1] == w.shape[1]
            and x.dtype == w.dtype == torch.bfloat16
            and x.is_contiguous() and w.is_contiguous()
            and not x.requires_grad and not w.requires_grad)


def _args(x, w, out):
    m, k = x.shape
    n = w.shape[0]
    return (x, w, w, out, m, n, k, k, 1, k, 1, n, 1, 0)


def _entry_key(entry, args):
    # Mirror this pinned LibEntry ABI only during a cold call.
    spec = [a for i, a in enumerate(args) if i in entry.specialize_indices]
    dns = [a for i, a in enumerate(args) if i in entry.do_not_specialize_indices]
    return entry.key(spec, dns, [False])


def linear(input, weight, bias=None):
    generic = importlib.import_module('flag_gems.ops.linear')
    supported = _supported(input, weight, bias)
    if (supported and 1 <= input.shape[0] <= 7
            and tuple(weight.shape) == (2048, 6144)
            and input.data_ptr() % 16 == 0 and weight.data_ptr() % 16 == 0):
        generic = importlib.import_module('_metax.ops.linear_smallm_dot')
    entry = generic.linear_kernel
    if (not supported or triton.__version__ != '3.0.0'
            or entry._has_flagtune_tuner):
        _count('fallback')
        return generic.linear(input, weight, bias)
    device = input.device.index
    signature = (entry, entry.jit_function, tuple(input.shape), tuple(weight.shape),
                 input.dtype, input.stride(), weight.stride(),
                 input.data_ptr() % 16 == 0, weight.data_ptr() % 16 == 0)
    cache = _cache(device)
    record = cache.get(signature)
    if record is not None:
        key, original, runner, tail = record
        if entry.kernel_cache[device].get(key) is original and not original[4]:
            out = torch.empty((input.shape[0], weight.shape[0]), device=input.device, dtype=input.dtype)
            # torch allocations are aligned; do not reuse a specialization otherwise.
            if out.data_ptr() % 16 == 0:
                with torch_device_fn.device(input.device):
                    runner(input, weight, weight, out, *tail)
                cache.move_to_end(signature)
                _count('hit')
                return out
        del cache[signature]
        _count('invalidated')
    _count('cold')
    out = generic.linear(input, weight, bias)
    args = _args(input, weight, out)
    key = _entry_key(entry, args)
    original = entry.kernel_cache[device].get(key)
    if original is None or len(original) != 5 or original[4] or out.data_ptr() % 16:
        _count('unsupported_entry')
        return out
    kernel, meta = original[:2]
    # The standard runner obtains the current stream and profiler hooks on EVERY call.
    grid = (triton.cdiv(input.shape[0], meta['BLOCK_SIZE_M']),
            triton.cdiv(weight.shape[0], meta['BLOCK_SIZE_N']), 1)
    cache[signature] = (key, original, kernel[grid], args[4:])
    cache.move_to_end(signature)
    if len(cache) > 64:
        cache.popitem(last=False)
        _count('evicted')
    return out

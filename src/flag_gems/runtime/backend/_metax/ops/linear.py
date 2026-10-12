# Copyright 2026, The FlagOS Contributors. Licensed under Apache-2.0.
"""Reuse generic linear launch preparation; never change its compiled kernel."""
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


def _metadata(x, w, bias):
    if (bias is not None or type(x) is not torch.Tensor
            or not (type(w) is torch.Tensor or isinstance(w, torch.nn.Parameter))):
        return None
    ws = tuple(w.shape)
    if ws not in ((2048, 6144), (12288, 2048)):
        return None
    xd, wd = x.device, w.device
    xs = tuple(x.shape)
    xt, wt = x.dtype, w.dtype
    if (xd.type != 'cuda' or xd != wd or len(xs) != 2 or len(ws) != 2
            or xs[0] <= 0 or ws not in ((2048, 6144), (12288, 2048))
            or xs[1] != ws[1] or xt != wt or xt != torch.bfloat16
            or not x.is_contiguous() or not w.is_contiguous()
            or x.requires_grad or w.requires_grad):
        return None
    return (xs, ws, xd, xt, x.stride(), w.stride(),
            x.data_ptr() % 16 == 0, w.data_ptr() % 16 == 0)


def _supported(x, w, bias):
    return _metadata(x, w, bias) is not None


def _args(x, w, out, metadata=None):
    m, k = x.shape if metadata is None else metadata[0]
    n = w.shape[0] if metadata is None else metadata[1][0]
    return (x, w, w, out, m, n, k, k, 1, k, 1, n, 1, 0)


def _entry_key(entry, args):
    # Mirror this pinned LibEntry ABI only during a cold call.
    spec = [a for i, a in enumerate(args) if i in entry.specialize_indices]
    dns = [a for i, a in enumerate(args) if i in entry.do_not_specialize_indices]
    return entry.key(spec, dns, [False])


def linear(input, weight, bias=None):
    generic = importlib.import_module('flag_gems.ops.linear')
    entry = generic.linear_kernel
    metadata = _metadata(input, weight, bias)
    if (metadata is None or triton.__version__ != '3.0.0'
            or entry._has_flagtune_tuner):
        _count('fallback')
        return generic.linear(input, weight, bias)
    xs, ws, xd, dtype, xstride, wstride, xalign, walign = metadata
    device = xd.index
    signature = (entry, entry.jit_function, xs, ws, dtype, xstride, wstride, xalign, walign)
    cache = _cache(device)
    record = cache.get(signature)
    if record is not None:
        key, original, runner, tail = record
        if entry.kernel_cache[device].get(key) is original and not original[4]:
            out = torch.empty((xs[0], ws[0]), device=xd, dtype=dtype)
            # torch allocations are aligned; do not reuse a specialization otherwise.
            if out.data_ptr() % 16 == 0:
                with torch_device_fn.device(xd):
                    runner(input, weight, weight, out, *tail)
                cache.move_to_end(signature)
                _count('hit')
                return out
        del cache[signature]
        _count('invalidated')
    _count('cold')
    out = generic.linear(input, weight, bias)
    args = _args(input, weight, out, metadata)
    key = _entry_key(entry, args)
    original = entry.kernel_cache[device].get(key)
    if original is None or len(original) != 5 or original[4] or out.data_ptr() % 16:
        _count('unsupported_entry')
        return out
    kernel, meta = original[:2]
    # The standard runner obtains the current stream and profiler hooks on EVERY call.
    grid = (triton.cdiv(xs[0], meta['BLOCK_SIZE_M']),
            triton.cdiv(ws[0], meta['BLOCK_SIZE_N']), 1)
    cache[signature] = (key, original, kernel[grid], args[4:])
    cache.move_to_end(signature)
    if len(cache) > 64:
        cache.popitem(last=False)
        _count('evicted')
    return out

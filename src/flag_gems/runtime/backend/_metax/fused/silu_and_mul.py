# Copyright 2026 FlagOS Contributors. Licensed under Apache-2.0.
"""Reuse preparation for the existing rank-two SiLU kernel on bounded sliced inputs."""
import importlib
import math
import os
import threading
from collections import Counter, OrderedDict

import torch
import triton
from flag_gems.runtime import torch_device_fn

_local = threading.local()
_diagnostics = False
_CONFIG = dict(max_tile_size=2048, max_grid_size=(65536, 65536, 65536),
               max_num_warps_per_cta=8, prefer_block_pointer=True, prefer_1d_tile=False)


def diagnostics(enabled=True):
    global _diagnostics
    _diagnostics = enabled
    return dict(getattr(_local, 'counts', {}))


def _count(name):
    if _diagnostics:
        if not hasattr(_local, 'counts'):
            _local.counts = Counter()
        _local.counts[name] += 1


def _cache():
    pid = os.getpid()
    if getattr(_local, 'pid', None) != pid:
        _local.pid, _local.records = pid, OrderedDict()
    return _local.records


def clear_cache():
    _cache().clear()


def _supported(x, y):
    return (type(x) is torch.Tensor and type(y) is torch.Tensor
            and torch.is_inference_mode_enabled() and not x.requires_grad and not y.requires_grad
            and x.device.type == 'cuda' and x.device == y.device
            and x.dtype == y.dtype == torch.bfloat16 and x.ndim == y.ndim == 2
            and x.shape == y.shape and 1 <= x.shape[0] <= 1024 and x.shape[1] > 0
            and x.stride() == y.stride() == (2*x.shape[1], 1)
            and x.data_ptr() % 16 == y.data_ptr() % 16 == 0)


def _stamp(pointwise):
    return (pointwise._scalar_fn, pointwise._scalar_fn_cache_key,
            repr(vars(pointwise.fx)), repr(vars(pointwise.complex_strategy)))


def _descriptor(pointwise, x, y, out):
    """Recover the original generated wrapper's layout and its cached binary, without a launch."""
    overload = pointwise.instantiate(2)
    info = pointwise.get_kernel_info(2)
    namespace = overload.__globals__
    entry = namespace[info.kernel_name]
    if entry._has_flagtune_tuner or entry.fn is not entry.jit_function:
        return None
    shape = tuple(x.shape)
    tile = namespace['heuristics_for_tile_size'](2048, *shape)
    num_tiles = math.prod(triton.cdiv(s, t) for s, t in zip(shape, tile))
    grid = (min(65536, num_tiles), 1, 1)
    tiles_per_cta = triton.cdiv(num_tiles, grid[0])
    args = [x, y, out]
    for tensor in (x, y, out):
        strides = tensor.stride()
        args.extend((*strides, *namespace['stride_order'](strides)))
    args.extend((*shape, math.prod(shape), tiles_per_cta, *tile, tiles_per_cta == 1))
    if len(args) != len(entry.jit_function.params):
        return None
    spec = [v for i, v in enumerate(args) if i in entry.specialize_indices]
    dns = [v for i, v in enumerate(args) if i in entry.do_not_specialize_indices]
    const = [v for i, v in enumerate(args) if i not in entry.specialize_indices and i not in entry.do_not_specialize_indices]
    key = entry.key(spec, dns, const)
    original = entry.kernel_cache[x.device.index].get(key)
    if original is None or len(original) != 5 or original[4]:
        return None
    runtime = [v for v, param in zip(args, entry.jit_function.params) if not param.is_constexpr]
    if len(runtime) != 7 or out.stride() != (shape[1], 1):
        return None
    return (overload, entry, entry.jit_function, key, original, original[0][grid],
            tuple(runtime[3:]), _stamp(pointwise))


def silu_and_mul(A, B):
    generic = importlib.import_module('flag_gems.fused.silu_and_mul')
    if not _supported(A, B) or triton.__version__ != '3.0.0':
        _count('fallback')
        return generic.silu_and_mul(A, B)
    pointwise = generic.silu_and_mul_kernel
    if vars(pointwise.config) != _CONFIG:
        _count('config_fallback')
        return generic.silu_and_mul(A, B)
    signature = (pointwise, A.device.index, tuple(A.shape), A.stride(), B.stride(), A.dtype)
    cache = _cache()
    record = cache.get(signature)
    if record is not None:
        overload, entry, jit, key, original, runner, tail, stamp = record
        if (pointwise.overloads.get('2_True') is overload
                and entry.fn is jit and entry.jit_function is jit and not entry._has_flagtune_tuner
                and overload.__globals__[pointwise.get_kernel_info(2).kernel_name] is entry
                and entry.kernel_cache[A.device.index].get(key) is original and not original[4]
                and _stamp(pointwise) == stamp):
            out = torch.empty(A.shape, device=A.device, dtype=A.dtype)
            if out.data_ptr() % 16 == 0:
                with torch_device_fn.device(A.device):
                    runner(A, B, out, *tail)
                cache.move_to_end(signature)
                _count('hit')
                return out
        del cache[signature]
        _count('invalidated')
        # A dynamic replacement must execute its own normal semantics this call.
        return generic.silu_and_mul(A, B)
    _count('cold')
    out = generic.silu_and_mul(A, B)
    record = _descriptor(pointwise, A, B, out)
    if record is not None:
        cache[signature] = record
        cache.move_to_end(signature)
        if len(cache) > 64:
            cache.popitem(last=False)
            _count('evicted')
    else:
        _count('unsupported_entry')
    return out

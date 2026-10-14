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


def _metadata(x, y):
    """Read each input metadata field once per supported public invocation."""
    if (type(x) is not torch.Tensor or type(y) is not torch.Tensor
            or not torch.is_inference_mode_enabled() or x.requires_grad or y.requires_grad):
        return None
    device, other_device = x.device, y.device
    dtype, other_dtype = x.dtype, y.dtype
    shape, other_shape = tuple(x.shape), tuple(y.shape)
    if (device.type != 'cuda' or device != other_device
            or dtype != torch.bfloat16 or dtype != other_dtype
            or len(shape) != 2 or len(other_shape) != 2 or shape != other_shape
            or not 1 <= shape[0] <= 1024 or shape[1] <= 0):
        return None
    xstride, ystride = x.stride(), y.stride()
    if (xstride != ystride or xstride != (2 * shape[1], 1)
            or x.data_ptr() % 16 != 0 or y.data_ptr() % 16 != 0):
        return None
    return shape, device, dtype, xstride, ystride, device.index


def _supported(x, y):
    return _metadata(x, y) is not None


def _stamp(pointwise):
    return (pointwise._scalar_fn, pointwise._scalar_fn_cache_key,
            repr(vars(pointwise.fx)), repr(vars(pointwise.complex_strategy)))


def _descriptor(pointwise, x, y, out, metadata=None):
    """Recover the original generated wrapper's layout and its cached binary, without a launch."""
    if metadata is None:
        metadata = _metadata(x, y)
    if metadata is None:
        return None
    shape, device, dtype, xstride, ystride, device_index = metadata
    overload = pointwise.instantiate(2)
    info = pointwise.get_kernel_info(2)
    namespace = overload.__globals__
    entry = namespace[info.kernel_name]
    if entry._has_flagtune_tuner or entry.fn is not entry.jit_function:
        return None
    tile = namespace['heuristics_for_tile_size'](2048, *shape)
    num_tiles = math.prod(triton.cdiv(s, t) for s, t in zip(shape, tile))
    grid = (min(65536, num_tiles), 1, 1)
    tiles_per_cta = triton.cdiv(num_tiles, grid[0])
    args = [x, y, out]
    output_stride = out.stride()
    for strides in (xstride, ystride, output_stride):
        args.extend((*strides, *namespace['stride_order'](strides)))
    args.extend((*shape, math.prod(shape), tiles_per_cta, *tile, tiles_per_cta == 1))
    if len(args) != len(entry.jit_function.params):
        return None
    spec = [v for i, v in enumerate(args) if i in entry.specialize_indices]
    dns = [v for i, v in enumerate(args) if i in entry.do_not_specialize_indices]
    const = [v for i, v in enumerate(args) if i not in entry.specialize_indices and i not in entry.do_not_specialize_indices]
    key = entry.key(spec, dns, const)
    original = entry.kernel_cache[device_index].get(key)
    if original is None or len(original) != 5 or original[4]:
        return None
    runtime = [v for v, param in zip(args, entry.jit_function.params) if not param.is_constexpr]
    if len(runtime) != 7 or output_stride != (shape[1], 1):
        return None
    return (overload, entry, entry.jit_function, key, original, original[0][grid],
            tuple(runtime[3:]), _stamp(pointwise))


def silu_and_mul(A, B):
    generic = importlib.import_module('flag_gems.fused.silu_and_mul')
    metadata = _metadata(A, B)
    if metadata is None or triton.__version__ != '3.0.0':
        _count('fallback')
        return generic.silu_and_mul(A, B)
    pointwise = generic.silu_and_mul_kernel
    if vars(pointwise.config) != _CONFIG:
        _count('config_fallback')
        return generic.silu_and_mul(A, B)
    shape, device, dtype, xstride, ystride, device_index = metadata
    signature = (pointwise, device_index, shape, xstride, ystride, dtype)
    cache = _cache()
    record = cache.get(signature)
    if record is not None:
        overload, entry, jit, key, original, runner, tail, stamp = record
        if (pointwise.overloads.get('2_True') is overload
                and entry.fn is jit and entry.jit_function is jit and not entry._has_flagtune_tuner
                and overload.__globals__[pointwise.get_kernel_info(2).kernel_name] is entry
                and entry.kernel_cache[device_index].get(key) is original and not original[4]
                and _stamp(pointwise) == stamp):
            out = torch.empty(shape, device=device, dtype=dtype)
            if out.data_ptr() % 16 == 0:
                with torch_device_fn.device(device):
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
    record = _descriptor(pointwise, A, B, out, metadata)
    if record is not None:
        cache[signature] = record
        cache.move_to_end(signature)
        if len(cache) > 64:
            cache.popitem(last=False)
            _count('evicted')
    else:
        _count('unsupported_entry')
    return out

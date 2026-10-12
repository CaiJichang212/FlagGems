# Copyright 2026 FlagOS Contributors. Licensed under Apache-2.0.
"""Reuse preparation for the existing rank-two SiLU kernel on sliced decode inputs."""
import importlib
import enum
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
            and x.shape == y.shape and 1 <= x.shape[0] <= 7 and x.shape[1] > 0
            and x.stride() == y.stride() == (2*x.shape[1], 1)
            and x.data_ptr() % 16 == y.data_ptr() % 16 == 0)


_FX_KEYS = frozenset(('_promotion_methods', '_num_inputs', '_is_tensor', '_dtypes',
                      '_num_outputs', '_num_input_tensors', '_num_non_tensor_inputs', '_input_id'))
_CS_KEYS = frozenset(('mode', 'cross_kernel', 'tensorize_scalars', 'fallback_target'))


def _stamp(pointwise):
    # All fields of the pinned schema are included by value. Unsupported dynamic
    # objects force invalidation instead of retaining a mutable object reference.
    fx, cs = vars(pointwise.fx), vars(pointwise.complex_strategy)
    if fx.keys() != _FX_KEYS or cs.keys() != _CS_KEYS:
        return (object(),)
    counts = (fx['_num_inputs'], fx['_num_outputs'], fx['_num_input_tensors'], fx['_num_non_tensor_inputs'])
    tensors, dtypes, ids, promotions = fx['_is_tensor'], fx['_dtypes'], fx['_input_id'], fx['_promotion_methods']
    if (any(type(v) is not int for v in counts)
            or any(type(v) is not list for v in (tensors, dtypes, ids, promotions))
            or any(type(v) is not bool for v in tensors)
            or any(v is not None for v in dtypes)
            or any(type(v) is not int for v in ids)
            or cs['cross_kernel'] is not None or cs['fallback_target'] is not None
            or type(cs['tensorize_scalars']) is not bool):
        return (object(),)
    packed = []
    for row in promotions:
        if type(row) is not tuple or not row or any(type(v) is not int for v in row[:-1]):
            return (object(),)
        kind = row[-1]
        value = getattr(kind, 'value', None)
        if (not isinstance(kind, enum.Enum) or type(kind.name) is not str
                or type(value) is not tuple or any(type(v) is not int for v in value)):
            return (object(),)
        packed.append((row[:-1], type(kind), kind.name, value))
    mode = cs['mode']
    if not isinstance(mode, enum.Enum) or type(mode.name) is not str or type(mode.value) is not int:
        return (object(),)
    return (pointwise._scalar_fn, pointwise._scalar_fn_cache_key,
            tuple(fx), counts, tuple(tensors), tuple(dtypes), tuple(ids), tuple(packed),
            tuple(cs), type(mode), mode.name, mode.value, None, cs['tensorize_scalars'], None)


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

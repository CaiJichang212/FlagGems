# Copyright 2026 FlagOS Contributors. Licensed under Apache-2.0.
"""Private small decode QKV dot + NeoX RoPE + page16 NHD insertion.

Caller owns valid positions and unique nonnegative slots; negative/short slots
are padding. No retry after submission. K/V projection values remain in QKV.
"""
import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn


@triton.jit
def qkv_rope_cache_kernel(X, W, O, POS, TABLE, CACHE, SLOTS,
                          M: tl.constexpr, XS: tl.constexpr, WS: tl.constexpr,
                          PS: tl.constexpr, TS: tl.constexpr, TD: tl.constexpr,
                          SS: tl.constexpr, NS: tl.constexpr, PAGES: tl.constexpr,
                          MAXPOS: tl.constexpr,
                          BLOCK_M: tl.constexpr = 128,
                          BLOCK_N: tl.constexpr = 32,
                          BLOCK_K: tl.constexpr = 32):
    rm = tl.arange(0, BLOCK_M)
    lane = tl.arange(0, BLOCK_N)
    pair = tl.program_id(0) * (BLOCK_N // 2) + lane // 2
    # Each adjacent pair owns NeoX dimensions d,d+64 of exactly one head.
    col = (pair // 64) * 128 + pair % 64 + (lane % 2) * 64
    rk = tl.arange(0, BLOCK_K)
    xp = X + rm[:, None] * XS + rk[None, :]
    wp = W + col[None, :] * WS + rk[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(0, 2048 // BLOCK_K):
        a = tl.load(xp, rm[:, None] < M, other=0.)
        b = tl.load(wp)
        acc += tl.dot(a, b, allow_tf32=False)
        xp += BLOCK_K
        wp += BLOCK_K
    rounded = acc.to(tl.bfloat16)
    lo, hi = tl.split(tl.reshape(rounded, (BLOCK_M, BLOCK_N // 2, 2)))
    pp = tl.program_id(0) * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
    c0 = (pp // 64) * 128 + pp % 64
    pos = tl.load(POS + rm * PS, rm < M, other=0).to(tl.int64)
    valid = (pos >= 0) & (pos < MAXPOS)
    tl.device_assert((rm >= M) | valid, 'RoPE position out of range')
    cos = tl.load(TABLE + pos[:, None] * TS + (pp % 64)[None, :] * TD,
                  (rm[:, None] < M) & valid[:, None], other=0).to(tl.float32)
    sin = tl.load(TABLE + pos[:, None] * TS + ((pp % 64) + 64)[None, :] * TD,
                  (rm[:, None] < M) & valid[:, None], other=0).to(tl.float32)
    # Exact standalone order: BF16 projection, FP32 table promotion, FMA enabled.
    rlo = lo.to(tl.float32) * cos + hi.to(tl.float32) * (-sin)
    rhi = hi.to(tl.float32) * cos + lo.to(tl.float32) * sin
    mask = (rm[:, None] < M) & valid[:, None]
    tl.store(O + rm[:, None] * 2560 + c0[None, :], tl.where(c0[None, :] < 2048, rlo, lo), mask)
    tl.store(O + rm[:, None] * 2560 + (c0 + 64)[None, :], tl.where(c0[None, :] < 2048, rhi, hi), mask)
    slot = tl.load(SLOTS + rm * SS, (rm < M) & (rm < NS), other=-1).to(tl.int64)
    tl.device_assert((rm >= M) | (slot < PAGES * 16), 'KV slot out of range')
    active = (slot >= 0) & (slot < PAGES * 16) & valid & (rm < M)
    safe = tl.where(active, slot, 0)
    # K/V blocks own disjoint cache halves; no cross-CTA read/write dependency.
    cache_col = c0 - 2048
    ptr = CACHE + (cache_col[None, :] // 256) * (PAGES * 4096) + safe[:, None] * 256 + (cache_col % 256)[None, :]
    cmask = active[:, None] & (c0[None, :] >= 2048)
    tl.store(ptr, tl.where(c0[None, :] < 2304, rlo, lo).to(tl.bfloat16), cmask)
    tl.store(ptr + 64, tl.where(c0[None, :] < 2304, rhi, hi).to(tl.bfloat16), cmask)


def supported(x, weight, positions, table, cache, slots):
    ts = (x, positions, table, cache, slots)
    if any(type(t) is not torch.Tensor for t in ts): return False
    if not (type(weight) is torch.Tensor or isinstance(weight, torch.nn.Parameter)): return False
    if torch.is_grad_enabled() or not torch.is_inference_mode_enabled() or torch.compiler.is_compiling(): return False
    if any(t.requires_grad for t in (*ts, weight)): return False
    if x.device.type != 'cuda' or any(t.device != x.device for t in (*ts, weight)): return False
    if x.dtype != weight.dtype or x.dtype != cache.dtype or x.dtype != torch.bfloat16: return False
    if x.ndim != 2 or not 1 <= x.shape[0] <= 4 or x.shape[1] != 2048: return False
    if tuple(weight.shape) != (2560, 2048) or not x.is_contiguous() or not weight.is_contiguous(): return False
    if cache.ndim != 5 or cache.shape[0] != 2 or tuple(cache.shape[2:]) != (16, 2, 128) or cache.shape[1] < 1 or not cache.is_contiguous(): return False
    if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] != 128 or table.dtype not in (torch.bfloat16, torch.float16, torch.float32): return False
    if table.stride(1) < 1 or table.stride(0) < 128 * table.stride(1): return False
    if positions.shape != (x.shape[0],) or slots.ndim != 1 or slots.numel() > x.shape[0]: return False
    if any(t.dtype not in (torch.int32, torch.int64) or t.stride(0) < 1 for t in (positions, slots)): return False
    # All reads/writes must use distinct storage except table/position read aliases.
    if cache.untyped_storage().data_ptr() in {t.untyped_storage().data_ptr() for t in (x,weight,positions,table,slots)}: return False
    return x.data_ptr() % 16 == weight.data_ptr() % 16 == 0


def _output_supported(out, x, weight, positions, table, cache, slots):
    # Private destination ABI: never permit overwriting any input or KV storage.
    # The paired attention destination is a disjoint view, not an input here.
    return (type(out) is torch.Tensor and out.device == x.device
            and out.dtype == x.dtype and tuple(out.shape) == (x.shape[0], 2560)
            and out.is_contiguous() and not out.requires_grad and out.data_ptr() % 16 == 0
            and out.untyped_storage().data_ptr() not in {
                t.untyped_storage().data_ptr() for t in (x, weight, positions, table, cache, slots)})


def try_qkv_rope_cache(x, weight, positions, table, cache, slots, *, out=None):
    """Return None only before launch; submitted exceptions propagate."""
    if not supported(x, weight, positions, table, cache, slots): return None
    if out is None:
        out = torch.empty((x.shape[0], 2560), dtype=x.dtype, device=x.device)
    elif not _output_supported(out, x, weight, positions, table, cache, slots):
        return None
    with torch_device_fn.device(x.device):
        kernel = qkv_rope_cache_kernel[(80,)](
            x, weight, out, positions, table, cache, slots,
            x.shape[0], x.stride(0), weight.stride(0), positions.stride(0),
            *table.stride(), slots.stride(0), slots.numel(), cache.shape[1], table.shape[0],
            num_warps=4, num_stages=4, enable_fp_fusion=True)
    if kernel is None: raise RuntimeError('QKV fusion launch suppressed; cache state is unknown')
    return out

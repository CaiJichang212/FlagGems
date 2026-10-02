# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Paged attention for batches whose requests each own zero or one query token.

The caller owns KV updates, output and persistent FP32 scratch. No allocations
or device-to-host metadata reads occur here. Feature/quantization dispatch is
the attention adapter's responsibility; this operator implements BF16 16Q/2KV,
head dimension 128 and full causal single-token attention only.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_single_token_split(
    Q, K, V, QS, SL, BT, A, M, L,
    scale,
    q0: tl.int64, q1: tl.int64, q2: tl.int64,
    k0: tl.int64, k1: tl.int64, k2: tl.int64, k3: tl.int64,
    v0: tl.int64, v1: tl.int64, v2: tl.int64, v3: tl.int64,
    bt0: tl.int64, bt1: tl.int64,
    qs0: tl.int64, sl0: tl.int64,
    a0: tl.int64, a1: tl.int64, a2: tl.int64, a3: tl.int64,
    m0: tl.int64, m1: tl.int64, m2: tl.int64,
    l0: tl.int64, l1: tl.int64, l2: tl.int64,
    PAGE: tl.constexpr, Q_CAPACITY: tl.constexpr,
    SPLITS: tl.constexpr, TILE: tl.constexpr,
):
    request = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    token = tl.load(QS + request * qs0).to(tl.int64)
    query_len = tl.load(QS + (request + 1) * qs0) - token
    # Zero-query slots can alias a following live query. Never write them.
    if (query_len != 1) | (token < 0) | (token >= Q_CAPACITY):
        return
    length = tl.load(SL + request * sl0)
    heads = kv_head * 8 + tl.arange(0, 16)
    real_heads = tl.arange(0, 16) < 8
    dims = tl.arange(0, 128)
    acc_ptr = A + token * a0 + heads[:, None] * a1 + split * a2 + dims[None, :] * a3
    max_ptr = M + token * m0 + heads * m1 + split * m2
    sum_ptr = L + token * l0 + heads * l1 + split * l2
    tiles_per_split = (tl.maximum(length, 0) + SPLITS * TILE - 1) // (SPLITS * TILE)
    begin = split * tiles_per_split * TILE
    end = tl.minimum(length, begin + tiles_per_split * TILE)
    # Every selected empty split is overwritten, including padding L=0.
    if begin >= end:
        tl.store(acc_ptr, tl.zeros((16, 128), tl.float32), mask=real_heads[:, None])
        tl.store(max_ptr, tl.full((16,), float("-inf"), tl.float32), mask=real_heads)
        tl.store(sum_ptr, tl.zeros((16,), tl.float32), mask=real_heads)
        return
    query = tl.load(Q + token * q0 + heads[:, None] * q1 + dims[None, :] * q2,
                    mask=real_heads[:, None], other=0.0)
    maximum = tl.full((16,), float("-inf"), tl.float32)
    denominator = tl.zeros((16,), tl.float32)
    accumulator = tl.zeros((16, 128), tl.float32)
    positions = tl.arange(0, TILE)
    for tile_begin in range(begin, end, TILE):
        logical = tile_begin + positions
        valid = logical < end
        # Per-token page lookup supports tiles spanning arbitrary physical pages.
        physical = tl.load(BT + request * bt0 + (logical // PAGE) * bt1,
                           mask=valid, other=0).to(tl.int64)
        slot = logical % PAGE
        keys = tl.load(K + physical[None, :] * k0 + slot[None, :] * k1
                       + kv_head * k2 + dims[:, None] * k3,
                       mask=valid[None, :], other=0.0)
        scores = tl.dot(query, keys, allow_tf32=False) * scale
        scores = tl.where(valid[None, :], scores, float("-inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        correction = tl.exp(maximum - next_maximum)
        probabilities = tl.exp(scores - next_maximum[:, None])
        denominator = denominator * correction + tl.sum(probabilities, 1)
        values = tl.load(V + physical[:, None] * v0 + slot[:, None] * v1
                         + kv_head * v2 + dims[None, :] * v3,
                         mask=valid[:, None], other=0.0)
        # Preserve the FP32 probability more accurately with two BF16 operands.
        # A single BF16 P cast can fail the independent FP32 numerical gate,
        # including shapes where the original kernel also exceeds tolerance.
        probability_hi = probabilities.to(tl.bfloat16)
        probability_lo = (probabilities - probability_hi.to(tl.float32)).to(tl.bfloat16)
        accumulator = accumulator * correction[:, None]
        accumulator += tl.dot(probability_hi, values, allow_tf32=False)
        accumulator += tl.dot(probability_lo, values, allow_tf32=False)
        maximum = next_maximum
    tl.store(acc_ptr, accumulator, mask=real_heads[:, None])
    tl.store(max_ptr, maximum, mask=real_heads)
    tl.store(sum_ptr, denominator, mask=real_heads)


@triton.jit
def _paged_single_token_merge(
    O, QS, A, M, L,
    qs0: tl.int64,
    o0: tl.int64, o1: tl.int64, o2: tl.int64,
    a0: tl.int64, a1: tl.int64, a2: tl.int64, a3: tl.int64,
    m0: tl.int64, m1: tl.int64, m2: tl.int64,
    l0: tl.int64, l1: tl.int64, l2: tl.int64,
    Q_CAPACITY: tl.constexpr, SPLITS: tl.constexpr,
):
    request = tl.program_id(0)
    head = tl.program_id(1)
    token = tl.load(QS + request * qs0).to(tl.int64)
    query_len = tl.load(QS + (request + 1) * qs0) - token
    if (query_len != 1) | (token < 0) | (token >= Q_CAPACITY):
        return
    split = tl.arange(0, SPLITS)
    dims = tl.arange(0, 128)
    maxima = tl.load(M + token * m0 + head * m1 + split * m2)
    sums = tl.load(L + token * l0 + head * l1 + split * l2)
    valid = sums > 0
    overall_max = tl.max(tl.where(valid, maxima, float("-inf")), 0)
    # Empty request/segments never evaluate -inf-(-inf), nor divide by zero.
    overall_max = tl.where(tl.sum(valid.to(tl.int32), 0) > 0, overall_max, 0.0)
    safe_maxima = tl.where(valid, maxima, overall_max)
    weights = tl.where(valid, tl.exp(safe_maxima - overall_max), 0.0)
    parts = tl.load(A + token * a0 + head * a1 + split[:, None] * a2 + dims[None, :] * a3)
    total = tl.sum(sums * weights, 0)
    merged = tl.sum(parts * weights[:, None], 0) / tl.where(total > 0, total, 1.0)
    tl.store(O + token * o0 + head * o1 + dims * o2, merged)


def paged_single_token_split_attention(
    q, k, v, out, cu_seqlens_q, seqused_k, block_table, softmax_scale,
    softmax_segm_output, softmax_segm_max, softmax_segm_expsum,
    *, splits=16, tile=128,
):
    """Write caller output using caller-owned scratch with its actual strides.

    Device metadata must describe query lengths 0 or 1, valid page entries and
    live sequence lengths within the table. Query length zero is ignored; a
    query of length one with zero KV length writes neutral scratch/output.
    The adapter validates all unsupported attention features before this call.
    """
    if type(splits) is not int or splits not in (1, 2, 4, 8, 16):
        raise ValueError("splits must be 1, 2, 4, 8 or 16")
    if type(tile) is not int or tile not in (64, 128, 256):
        raise ValueError("tile must be 64, 128 or 256")
    if type(softmax_scale) not in (int, float) or not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    if q.ndim != 3 or tuple(q.shape[1:]) != (16, 128) or out.shape != q.shape:
        raise ValueError("Q/output require matching [tokens,16,128] shapes")
    if k.ndim != 4 or k.shape != v.shape or tuple(k.shape[2:]) != (2, 128) or k.shape[1] <= 0:
        raise ValueError("K/V require matching [pages,page_size,2,128] shapes")
    if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v, out)):
        raise TypeError("Q/K/V/output must be BF16")
    batch = seqused_k.shape[0] if seqused_k.ndim == 1 else -1
    if (batch < 0 or cu_seqlens_q.ndim != 1 or cu_seqlens_q.shape[0] != batch + 1
            or block_table.ndim != 2 or block_table.shape[0] < batch):
        raise ValueError("Invalid query/length/page-table metadata shapes")
    if any(tensor.dtype not in (torch.int32, torch.int64) for tensor in (cu_seqlens_q, seqused_k, block_table)):
        raise TypeError("Metadata must be int32 or int64")
    scratch = (softmax_segm_output, softmax_segm_max, softmax_segm_expsum)
    if (softmax_segm_output.ndim != 4 or tuple(softmax_segm_output.shape[1:]) != (16, 16, 128)
            or any(tensor.ndim != 3 or tuple(tensor.shape[1:]) != (16, 16) for tensor in scratch[1:])
            or any(tensor.shape[0] < q.shape[0] for tensor in scratch)):
        raise ValueError("Persistent scratch requires capacity for Q and original 16 segments")
    if any(tensor.dtype != torch.float32 for tensor in scratch):
        raise TypeError("Persistent scratch must be FP32")
    tensors = (q, k, v, out, cu_seqlens_q, seqused_k, block_table) + scratch
    if q.device.type != "cuda" or any(tensor.device != q.device for tensor in tensors):
        raise ValueError("All tensors must be on the same Iluvatar CUDA device")
    if any(any(stride <= 0 for stride in tensor.stride()) for tensor in tensors):
        raise ValueError("Only positive tensor strides are supported")
    if batch == 0 or q.shape[0] == 0:
        return out
    _paged_single_token_split[(batch, 2, splits)](
        q, k, v, cu_seqlens_q, seqused_k, block_table, *scratch, softmax_scale,
        *q.stride(), *k.stride(), *v.stride(), *block_table.stride(),
        cu_seqlens_q.stride(0), seqused_k.stride(0),
        *softmax_segm_output.stride(), *softmax_segm_max.stride(), *softmax_segm_expsum.stride(),
        PAGE=k.shape[1], Q_CAPACITY=q.shape[0], SPLITS=splits, TILE=tile,
        num_warps=4, num_stages=1,
    )
    _paged_single_token_merge[(batch, 16)](
        out, cu_seqlens_q, *scratch, cu_seqlens_q.stride(0), *out.stride(),
        *softmax_segm_output.stride(), *softmax_segm_max.stride(), *softmax_segm_expsum.stride(),
        Q_CAPACITY=q.shape[0], SPLITS=splits, num_warps=4, num_stages=1,
    )
    return out

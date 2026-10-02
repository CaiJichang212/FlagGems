# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Iluvatar paged, causal BF16 prefill for 16 Q / 2 KV heads and head size 128.

The caller updates KV separately. This operator only reads Q/K/V and metadata,
then writes the supplied output. It does not allocate a GPU workspace, inspect
GPU scalar values on the host, or autotune. Tile arguments are for offline
verification; serving uses one fixed configuration.

Unlike the existing unified kernel's two-token / eight-head tile, each normal
program combines QUERY_TILE tokens with GROUP_SIZE query heads. Page-table
entries are loaded once per physical page and broadcast over its token lanes.
One shared KV loop applies the causal mask to every tile. Mixed-batch single-token
requests take a compact eight-head path inside the same launch.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _page_offsets(
    BlockTable,
    sequence,
    key_start,
    sequence_length,
    PAGE_SIZE: tl.constexpr,
    KV_TILE: tl.constexpr,
    BT_S0: tl.constexpr,
    BT_S1: tl.constexpr,
):
    # Supported page sizes divide every supported KV tile. The [pages, slots]
    # construction exposes one block-table load per page instead of per token.
    # The sequence index comes from the int32 program/sequence search. Keep its
    # table address in vector registers to avoid scalarizing page loads and
    # retaining their address/predicate state throughout the KV loop.
    sequence = tl.inline_asm_elementwise(
        "ml_mov_b32 $0, $1;", constraints="=v,v", args=[sequence],
        dtype=tl.int32, is_pure=True, pack=1,
    )
    pages = key_start // PAGE_SIZE + tl.arange(0, KV_TILE // PAGE_SIZE)
    valid_page = pages * PAGE_SIZE < sequence_length
    physical = tl.load(
        BlockTable + sequence.to(tl.int64) * BT_S0 + pages.to(tl.int64) * BT_S1,
        mask=valid_page,
        other=0,
    ).to(tl.int64)
    physical = tl.broadcast_to(physical[:, None], (KV_TILE // PAGE_SIZE, PAGE_SIZE))
    physical = tl.reshape(physical, (KV_TILE,))
    slots = tl.broadcast_to(
        tl.arange(0, PAGE_SIZE)[None, :], (KV_TILE // PAGE_SIZE, PAGE_SIZE)
    )
    slots = tl.reshape(slots, (KV_TILE,)).to(tl.int64)
    return physical, slots


@triton.jit
def _softmax_tile(
    K,
    V,
    BlockTable,
    q_tile,
    row_valid,
    query_positions,
    maximum,
    denominator,
    accumulator,
    sequence,
    kv_head,
    sequence_length,
    key_start,
    scale,
    PAGE_SIZE: tl.constexpr,
    KV_TILE: tl.constexpr,
    BT_S0: tl.constexpr,
    BT_S1: tl.constexpr,
    K_S0: tl.constexpr,
    K_S1: tl.constexpr,
    K_S2: tl.constexpr,
    K_S3: tl.constexpr,
    V_S0: tl.constexpr,
    V_S1: tl.constexpr,
    V_S2: tl.constexpr,
    V_S3: tl.constexpr,
):
    physical, slots = _page_offsets(
        BlockTable, sequence, key_start, sequence_length,
        PAGE_SIZE, KV_TILE, BT_S0, BT_S1,
    )
    keys = key_start + tl.arange(0, KV_TILE)
    dimensions = tl.arange(0, 128).to(tl.int64)
    # Promote before multiplying page number by stride: the cache can exceed
    # 2**31 elements even though every sequence length fits comfortably in int32.
    k_offset = (
        physical[:, None] * K_S0 + slots[:, None] * K_S1
        + kv_head.to(tl.int64) * K_S2 + dimensions[None, :] * K_S3
    )
    v_offset = (
        physical[:, None] * V_S0 + slots[:, None] * V_S1
        + kv_head.to(tl.int64) * V_S2 + dimensions[None, :] * V_S3
    )
    key_valid = keys < sequence_length
    k_tile = tl.load(K + k_offset, mask=key_valid[:, None], other=0)
    v_tile = tl.load(V + v_offset, mask=key_valid[:, None], other=0)
    scores = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32) * scale
    visible = (
        row_valid[:, None] & key_valid[None, :]
        & (keys[None, :] <= query_positions[:, None])
    )
    scores = tl.where(visible, scores, float("-inf"))
    next_max = tl.maximum(maximum, tl.max(scores, axis=1))
    correction = tl.exp(maximum - next_max)
    probability = tl.exp(scores - next_max[:, None])
    denominator = denominator * correction + tl.sum(probability, axis=1)
    accumulator = accumulator * correction[:, None]
    # Keep the FP32 softmax weights accurate enough near cancelling outputs.
    # BF16 high and residual parts use the existing BF16 tensor-core path;
    # both products accumulate in FP32 before final output conversion.
    probability_hi = probability.to(tl.bfloat16)
    probability_lo = (probability - probability_hi.to(tl.float32)).to(tl.bfloat16)
    accumulator += tl.dot(probability_hi, v_tile, out_dtype=tl.float32)
    accumulator += tl.dot(probability_lo, v_tile, out_dtype=tl.float32)
    return next_max, denominator, accumulator


@triton.jit
def _attention_tile(
    Q,
    K,
    V,
    Out,
    BlockTable,
    sequence,
    query_start,
    query_length,
    sequence_length,
    local_start,
    kv_head,
    subgroup,
    num_actual_tokens,
    scale,
    PAGE_SIZE: tl.constexpr,
    KV_TILE: tl.constexpr,
    TOKEN_TILE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    Q_S0: tl.constexpr,
    Q_S1: tl.constexpr,
    Q_S2: tl.constexpr,
    O_S0: tl.constexpr,
    O_S1: tl.constexpr,
    O_S2: tl.constexpr,
    K_S0: tl.constexpr,
    K_S1: tl.constexpr,
    K_S2: tl.constexpr,
    K_S3: tl.constexpr,
    V_S0: tl.constexpr,
    V_S1: tl.constexpr,
    V_S2: tl.constexpr,
    V_S3: tl.constexpr,
    BT_S0: tl.constexpr,
    BT_S1: tl.constexpr,
):
    rows = tl.arange(0, ROWS)
    # Row coordinates can be warp-uniform under the dot operand layout. Force
    # this identity into vector registers so their masks and address indices
    # do not consume scarce SGPRs across the entire KV recurrence.
    rows = tl.inline_asm_elementwise(
        "ml_mov_b32 $0, $1;", constraints="=v,v", args=[rows],
        dtype=tl.int32, is_pure=True, pack=1,
    )
    local_token = local_start + rows // GROUP_SIZE
    token = query_start.to(tl.int64) + local_token.to(tl.int64)
    head = kv_head * 8 + subgroup * GROUP_SIZE + rows % GROUP_SIZE
    # Compare packed bounds relative to the sequence start. Address arithmetic
    # stays int64, while int32 metadata does not need a per-row int64 token
    # comparison (and the associated scalar predicate pairs). int64 starts
    # retain int64 subtraction/comparison here without narrowing metadata.
    row_valid = (local_token < query_length) & (
        local_token < num_actual_tokens - query_start
    )
    dimensions = tl.arange(0, 128).to(tl.int64)
    q_offset = token[:, None] * Q_S0 + head[:, None].to(tl.int64) * Q_S1 + dimensions[None, :] * Q_S2
    q_tile = tl.load(Q + q_offset, mask=row_valid[:, None], other=0)
    history = sequence_length - query_length
    query_positions = history + local_token
    # Invalid rows start with finite max=0, so all-masked score rows never form
    # -inf - -inf. Their probability and accumulator stay exactly zero.
    maximum = tl.where(row_valid, float("-inf"), 0.0).to(tl.float32)
    denominator = tl.zeros((ROWS,), dtype=tl.float32)
    accumulator = tl.zeros((ROWS, 128), dtype=tl.float32)
    visible_end = history + tl.minimum(local_start + TOKEN_TILE, query_length)
    # A single recurrence avoids duplicating the large inlined QK/PV body and
    # carrying its state across two loop regions in the Iluvatar compiler.
    # The extra causal comparison is redundant on fully visible tiles, but it
    # preserves the same visible keys without a second copy of the loop body.
    for key_start in range(0, visible_end, KV_TILE):
        maximum, denominator, accumulator = _softmax_tile(
            K, V, BlockTable, q_tile, row_valid, query_positions,
            maximum, denominator, accumulator, sequence, kv_head,
            sequence_length, key_start, scale, PAGE_SIZE, KV_TILE,
            BT_S0, BT_S1, K_S0, K_S1, K_S2, K_S3,
            V_S0, V_S1, V_S2, V_S3,
        )
    result = accumulator / tl.where(row_valid, denominator, 1.0)[:, None]
    o_offset = token[:, None] * O_S0 + head[:, None].to(tl.int64) * O_S1 + dimensions[None, :] * O_S2
    tl.store(Out + o_offset, result.to(tl.bfloat16), mask=row_valid[:, None])


@triton.jit
def _paged_prefill_attention_kernel(
    Q,
    K,
    V,
    Out,
    QueryStarts,
    SequenceLengths,
    BlockTable,
    num_sequences,
    num_actual_tokens,
    scale,
    PAGE_SIZE: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    KV_TILE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    Q_S0: tl.constexpr,
    Q_S1: tl.constexpr,
    Q_S2: tl.constexpr,
    O_S0: tl.constexpr,
    O_S1: tl.constexpr,
    O_S2: tl.constexpr,
    K_S0: tl.constexpr,
    K_S1: tl.constexpr,
    K_S2: tl.constexpr,
    K_S3: tl.constexpr,
    V_S0: tl.constexpr,
    V_S1: tl.constexpr,
    V_S2: tl.constexpr,
    V_S3: tl.constexpr,
    BT_S0: tl.constexpr,
    BT_S1: tl.constexpr,
    QS_S0: tl.constexpr,
    SL_S0: tl.constexpr,
):
    program = tl.program_id(0)
    group = tl.program_id(1)
    kv_head = group // (8 // GROUP_SIZE)
    subgroup = group % (8 // GROUP_SIZE)
    # boundary(s)=query_start[s]//QUERY_TILE+s is strictly increasing even for
    # zero-length requests. It gives an allocation-free upper-bound tile grid.
    left = tl.full((), 0, tl.int32)
    right = num_sequences
    while left < right:
        mid = (left + right) // 2
        start = tl.load(QueryStarts + mid.to(tl.int64) * QS_S0)
        boundary = start // QUERY_TILE + mid
        if boundary <= program:
            left = mid + 1
        else:
            right = mid
    sequence = left - 1
    query_start = tl.load(QueryStarts + sequence.to(tl.int64) * QS_S0)
    query_stop = tl.load(QueryStarts + (sequence + 1).to(tl.int64) * QS_S0)
    query_length = query_stop - query_start
    sequence_length = tl.load(SequenceLengths + sequence.to(tl.int64) * SL_S0)
    # Model runner padding can have qlen=1 but S=0. Do not read any Q/KV/page
    # pointers or write output before excluding it.
    if query_length <= 0:
        return
    if sequence_length <= 0:
        return
    # Device metadata is an upstream contract; debug builds diagnose violations
    # without requiring host-side .item()/D2H in production.
    tl.device_assert(sequence_length >= query_length, "active sequence shorter than query")
    if sequence_length < query_length:
        return
    local_tile = program - (query_start // QUERY_TILE + sequence)
    local_start = local_tile * QUERY_TILE
    if local_start >= query_length:
        return
    if query_start + local_start >= num_actual_tokens:
        return
    if query_length == 1:
        # One compact program per KV head, not one large mostly-masked Q tile
        # per subgroup. Large-path resource ceilings must still be profiled.
        if subgroup != 0:
            return
        _attention_tile(
            Q, K, V, Out, BlockTable, sequence, query_start, query_length,
            sequence_length, local_start, kv_head, 0, num_actual_tokens, scale,
            PAGE_SIZE, KV_TILE, 1, 8, 16,
            Q_S0, Q_S1, Q_S2, O_S0, O_S1, O_S2,
            K_S0, K_S1, K_S2, K_S3, V_S0, V_S1, V_S2, V_S3, BT_S0, BT_S1,
        )
    else:
        _attention_tile(
            Q, K, V, Out, BlockTable, sequence, query_start, query_length,
            sequence_length, local_start, kv_head, subgroup, num_actual_tokens, scale,
            PAGE_SIZE, KV_TILE, QUERY_TILE, GROUP_SIZE, QUERY_TILE * GROUP_SIZE,
            Q_S0, Q_S1, Q_S2, O_S0, O_S1, O_S2,
            K_S0, K_S1, K_S2, K_S3, V_S0, V_S1, V_S2, V_S3, BT_S0, BT_S1,
        )


def paged_prefill_attention(
    q,
    k_cache,
    v_cache,
    out,
    query_start_loc,
    seq_lens,
    block_table,
    max_query_len,
    softmax_scale,
    *,
    num_actual_tokens,
    query_tile=16,
    kv_tile=64,
    group_size=8,
):
    """Write paged causal attention into ``out`` and return that same tensor.

    Metadata must describe nondecreasing packed query starts, active S >= qlen,
    and valid physical page IDs for each active sequence. Zero-S requests are
    padding even when qlen > 0. Positive, nonoverlapping tensor strides are
    supported; writable output must not alias inputs. The adapter excludes
    noncausal/bias/quantized/windowed attention before calling this operator.
    ``max_query_len`` and ``num_actual_tokens`` are CPU metadata, never tensors.
    """
    if query_tile not in (16, 32, 64) or kv_tile not in (64, 128, 256) or group_size not in (1, 2, 4, 8):
        raise ValueError("Unsupported offline attention tile configuration")
    if type(max_query_len) is not int or max_query_len < 0:
        raise ValueError("max_query_len must be a nonnegative CPU integer")
    if type(num_actual_tokens) is not int or num_actual_tokens < 0:
        raise ValueError("num_actual_tokens must be a nonnegative CPU integer")
    if not isinstance(softmax_scale, (int, float)) or not math.isfinite(softmax_scale):
        raise ValueError("softmax_scale must be a finite CPU number")
    if q.ndim != 3 or q.shape[1:] != (16, 128) or out.shape != q.shape:
        raise ValueError("Expected Q and output [tokens, 16, 128]")
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape or k_cache.shape[2:] != (2, 128):
        raise ValueError("Expected paged K/V [pages, page_size, 2, 128]")
    page_size = k_cache.shape[1]
    if page_size not in (16, 32, 64):
        raise ValueError("Supported page sizes are 16, 32 and 64")
    if query_start_loc.ndim != 1 or seq_lens.ndim != 1 or block_table.ndim != 2:
        raise ValueError("Expected 1-D starts/lengths and 2-D block table")
    num_sequences = seq_lens.shape[0]
    if query_start_loc.shape[0] != num_sequences + 1 or block_table.shape[0] < num_sequences:
        raise ValueError("Metadata sequence counts disagree")
    if num_actual_tokens > q.shape[0]:
        raise ValueError("Actual token count exceeds Q/output capacity")
    tensors = (q, k_cache, v_cache, out, query_start_loc, seq_lens, block_table)
    if q.device.type != "cuda" or any(t.device != q.device for t in tensors):
        raise ValueError("All arguments must use the same Iluvatar CUDA device")
    if any(t.dtype != torch.bfloat16 for t in (q, k_cache, v_cache, out)):
        raise ValueError("Q/K/V/output must be BF16")
    if any(t.dtype not in (torch.int32, torch.int64) for t in (query_start_loc, seq_lens, block_table)):
        raise ValueError("Metadata must use int32 or int64")
    if any(any(s <= 0 for s in tensor.stride()) for tensor in tensors):
        raise ValueError("Only positive tensor strides are supported")
    if num_actual_tokens == 0 or num_sequences == 0:
        return out
    if max_query_len == 0 or k_cache.shape[0] == 0 or block_table.shape[1] == 0:
        raise ValueError("Nonempty attention requires nonempty metadata and KV")
    grid = (num_actual_tokens // query_tile + num_sequences, 2 * (8 // group_size))
    _paged_prefill_attention_kernel[grid](
        q, k_cache, v_cache, out, query_start_loc, seq_lens, block_table,
        num_sequences, num_actual_tokens, float(softmax_scale),
        page_size, query_tile, kv_tile, group_size,
        *q.stride(), *out.stride(), *k_cache.stride(), *v_cache.stride(),
        *block_table.stride(), query_start_loc.stride(0), seq_lens.stride(0),
        # Eight warps reduce per-warp row/layout predicate expansion, which
        # otherwise spills SGPR masks even when the vector payload fits.
        num_warps=8,
        num_stages=1,
    )
    return out

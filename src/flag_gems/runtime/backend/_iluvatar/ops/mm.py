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

import json
import logging
import math
import os

import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]


def get_higher_dtype(a, b):
    if a is b:
        return a
    assert a in _ordered_datatypes
    assert b in _ordered_datatypes
    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a
    raise AssertionError("unreachable")


def _to_tl_type(ty):
    return getattr(tl, str(ty).split(".")[-1])


def _requires_masked_load(a, b, M, N):
    """Keep unsupported SME transfers on explicit elementwise masked loads.

    BI-V150's unmasked dot-load lowering is unsafe for sub-16 output tiles
    with pipelining, and for input base/pitch not aligned to 128 bytes. Check
    the actual tensors rather than logical shapes: views may have padding or
    nonzero storage offsets. A mask also keeps these layouts out of that
    lowering without allocating a contiguous copy.
    """
    if M < 16 or N < 16:
        return True
    for tensor in (a, b):
        if tensor.data_ptr() % 128:
            return True
        strides = tensor.stride()
        if 1 not in strides or max(strides) * tensor.element_size() % 128:
            return True
    return False


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K"],
    prune_configs_by={
        "early_config_prune": early_config_prune,
        "perf_model": estimate_matmul_time,
        "top_k": 15,
    },
    warmup=5,
    rep=10,
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
        "UPGRADE": lambda args: math.ceil(
            (args["M"] * args["N"]) / (args["BLOCK_M"] * args["BLOCK_N"])
        ).bit_length()
        > 31,
        "UPGRADE_A_OFFS": lambda args: math.ceil(args["M"] * args["K"]).bit_length()
        > 31,
        "UPGRADE_B_OFFS": lambda args: math.ceil(args["K"] * args["N"]).bit_length()
        > 31,
        "UPGRADE_C_OFFS": lambda args: math.ceil(args["M"] * args["N"]).bit_length()
        > 31,
    }
)
@triton.jit
def mm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    acc_dtype: tl.constexpr,
    input_precision: tl.constexpr,
    fp8_fast_accum: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    AB_DTYPE: tl.constexpr,
    UPGRADE: tl.constexpr,
    UPGRADE_A_OFFS: tl.constexpr,
    UPGRADE_B_OFFS: tl.constexpr,
    UPGRADE_C_OFFS: tl.constexpr,
    MASKED_LOAD: tl.constexpr,
):
    # matrix multiplication
    if UPGRADE:
        pid = tle.program_id(0)
        pid_z = tle.program_id(1)
    else:
        pid = tl.program_id(0)
        pid_z = tl.program_id(1)
    # grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # # re-order program ID for better L2 performance
    # width = GROUP_M * grid_n
    # group_id = pid // width
    # group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    # pid_m = group_id * GROUP_M + (pid % group_size)
    # pid_n = (pid % width) // (group_size)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    # do matrix multiplication
    if UPGRADE_A_OFFS:
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        ram = (tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)).to(tl.int64)
    else:
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    if UPGRADE_B_OFFS:
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        rbn = (tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)).to(tl.int64)
    else:
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    if EVEN_K:
        for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
            if MASKED_LOAD:
                # Keep both dimensions in the predicate. The modulo indices
                # preserve the existing tail semantics; explicit masks avoid
                # unsafe SME dot-load lowering for small/unaligned layouts.
                a = tl.load(A, mask=(ram[:, None] < M) & (rk[None, :] < K), other=0)
                b = tl.load(B, mask=(rk[:, None] < K) & (rbn[None, :] < N), other=0)
            else:
                a = tl.load(A)
                b = tl.load(B)
            if AB_DTYPE is not None:
                a = a.to(AB_DTYPE)
                b = b.to(AB_DTYPE)
            if fp8_fast_accum:
                acc = tl.dot(
                    a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
                )
            else:
                acc += tl.dot(
                    a, b, out_dtype=acc_dtype, input_precision=input_precision
                )
            A += BLOCK_K * SPLIT_K * stride_ak
            B += BLOCK_K * SPLIT_K * stride_bk
    else:
        loop_num = tl.cdiv(K, BLOCK_K * SPLIT_K) - 1
        for k in range(0, loop_num):
            if MASKED_LOAD:
                # Keep both dimensions in the predicate. The modulo indices
                # preserve the existing tail semantics; explicit masks avoid
                # unsafe SME dot-load lowering for small/unaligned layouts.
                a = tl.load(A, mask=(ram[:, None] < M) & (rk[None, :] < K), other=0)
                b = tl.load(B, mask=(rk[:, None] < K) & (rbn[None, :] < N), other=0)
            else:
                a = tl.load(A)
                b = tl.load(B)
            if AB_DTYPE is not None:
                a = a.to(AB_DTYPE)
                b = b.to(AB_DTYPE)
            if fp8_fast_accum:
                acc = tl.dot(
                    a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
                )
            else:
                acc += tl.dot(
                    a, b, out_dtype=acc_dtype, input_precision=input_precision
                )
            A += BLOCK_K * SPLIT_K * stride_ak
            B += BLOCK_K * SPLIT_K * stride_bk

        _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
        k_remaining = K - loop_num * (BLOCK_K * SPLIT_K)
        a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
        b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if fp8_fast_accum:
            acc = tl.dot(
                a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
            )
        else:
            acc += tl.dot(a, b, out_dtype=acc_dtype, input_precision=input_precision)

    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    if UPGRADE_C_OFFS:
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn).to(tl.int64)
    else:
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


# Independent candidate keeps fixed configuration experiments out of the tuner.
@triton.jit
def grouped_mm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    acc_dtype: tl.constexpr,
    input_precision: tl.constexpr,
    fp8_fast_accum: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    AB_DTYPE: tl.constexpr,
    UPGRADE: tl.constexpr,
    UPGRADE_A_OFFS: tl.constexpr,
    UPGRADE_B_OFFS: tl.constexpr,
    UPGRADE_C_OFFS: tl.constexpr,
    MASKED_LOAD: tl.constexpr,
    GROUPED_M: tl.constexpr = False,
):
    # matrix multiplication
    if UPGRADE:
        pid = tle.program_id(0)
        pid_z = tle.program_id(1)
    else:
        pid = tl.program_id(0)
        pid_z = tl.program_id(1)
    grid_n = tl.cdiv(N, BLOCK_N)
    if GROUPED_M:
        grid_m = tl.cdiv(M, BLOCK_M)
        width = GROUP_M * grid_n
        group_id = pid // width
        first_m = group_id * GROUP_M
        group_size = tl.minimum(grid_m - first_m, GROUP_M)
        # Use the group-local ID, including when the final group is partial.
        local_pid = pid % width
        pid_m = first_m + local_pid % group_size
        pid_n = local_pid // group_size
    else:
        pid_m = pid // grid_n
        pid_n = pid % grid_n
    # do matrix multiplication
    if UPGRADE_A_OFFS:
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        ram = (tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)).to(tl.int64)
    else:
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    if UPGRADE_B_OFFS:
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        rbn = (tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)).to(tl.int64)
    else:
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    if EVEN_K:
        for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
            if MASKED_LOAD:
                # Keep both dimensions in the predicate. The modulo indices
                # preserve the existing tail semantics; explicit masks avoid
                # unsafe SME dot-load lowering for small/unaligned layouts.
                a = tl.load(A, mask=(ram[:, None] < M) & (rk[None, :] < K), other=0)
                b = tl.load(B, mask=(rk[:, None] < K) & (rbn[None, :] < N), other=0)
            else:
                a = tl.load(A)
                b = tl.load(B)
            if AB_DTYPE is not None:
                a = a.to(AB_DTYPE)
                b = b.to(AB_DTYPE)
            if fp8_fast_accum:
                acc = tl.dot(
                    a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
                )
            else:
                acc += tl.dot(
                    a, b, out_dtype=acc_dtype, input_precision=input_precision
                )
            A += BLOCK_K * SPLIT_K * stride_ak
            B += BLOCK_K * SPLIT_K * stride_bk
    else:
        loop_num = tl.cdiv(K, BLOCK_K * SPLIT_K) - 1
        for k in range(0, loop_num):
            if MASKED_LOAD:
                # Keep both dimensions in the predicate. The modulo indices
                # preserve the existing tail semantics; explicit masks avoid
                # unsafe SME dot-load lowering for small/unaligned layouts.
                a = tl.load(A, mask=(ram[:, None] < M) & (rk[None, :] < K), other=0)
                b = tl.load(B, mask=(rk[:, None] < K) & (rbn[None, :] < N), other=0)
            else:
                a = tl.load(A)
                b = tl.load(B)
            if AB_DTYPE is not None:
                a = a.to(AB_DTYPE)
                b = b.to(AB_DTYPE)
            if fp8_fast_accum:
                acc = tl.dot(
                    a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
                )
            else:
                acc += tl.dot(
                    a, b, out_dtype=acc_dtype, input_precision=input_precision
                )
            A += BLOCK_K * SPLIT_K * stride_ak
            B += BLOCK_K * SPLIT_K * stride_bk

        _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
        k_remaining = K - loop_num * (BLOCK_K * SPLIT_K)
        a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
        b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if fp8_fast_accum:
            acc = tl.dot(
                a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
            )
        else:
            acc += tl.dot(a, b, out_dtype=acc_dtype, input_precision=input_precision)

    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    if UPGRADE_C_OFFS:
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn).to(tl.int64)
    else:
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


# Only independently validated projection shapes may eventually be enabled here.
# An empty default preserves the contest-baseline dispatch, including small M.
_MM_CANDIDATE_CONFIGS = {}
_MM_PROJECTION_NK = {(2560, 2048), (2048, 2048), (12288, 2048), (2048, 6144)}


def _validate_fixed_config(config):
    required = {"block_m", "block_n", "block_k", "num_warps", "num_stages", "group_m"}
    if set(config) != required:
        raise ValueError("fixed GEMM configuration must contain " + str(sorted(required)))
    for name, value in config.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("block_m", "block_n", "block_k", "num_warps"):
        value = config[name]
        if value & (value - 1):
            raise ValueError(f"{name} must be a power of two")
    if min(config["block_m"], config["block_n"], config["block_k"]) < 16:
        raise ValueError("BF16 dot tile dimensions must be at least 16")


def configure_mm_candidate(configs):
    """Explicit experiment opt-in; call once before warmup or Graph capture.

    ``configs`` maps (M, N, K) to fixed launch dictionaries. Passing {} restores
    baseline dispatch. This is intentionally not an autotuning interface.
    """
    validated = {}
    for shape, config in configs.items():
        if len(shape) != 3 or shape[0] != 2048 or tuple(shape[1:]) not in _MM_PROJECTION_NK:
            raise ValueError(
                "candidate dispatch is restricted to observed M=2048 projections"
            )
        _validate_fixed_config(config)
        validated[tuple(shape)] = dict(config)
    global _MM_CANDIDATE_CONFIGS
    _MM_CANDIDATE_CONFIGS = validated


def _load_mm_candidate_environment():
    """Load a frozen experiment whitelist once, before any model execution."""
    name = "FLAGGEMS_ILUVATAR_MM_CONFIG"
    value = os.environ.get(name)
    if value is None:
        return
    try:
        raw = json.loads(value)
        if not isinstance(raw, dict):
            raise ValueError("expected a JSON object mapping M,N,K to fixed configs")
        configs = {}
        for key, config in raw.items():
            shape = tuple(int(part) for part in key.split(","))
            if key != ",".join(map(str, shape)):
                raise ValueError("shape keys must use canonical M,N,K decimal notation")
            configs[shape] = config
        configure_mm_candidate(configs)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"Invalid {name}: {exc}") from exc


_load_mm_candidate_environment()


def launch_mm_fixed(
    a,
    b,
    out,
    *,
    block_m,
    block_n,
    block_k,
    num_warps,
    num_stages,
    group_m=1,
    grouped=False,
):
    """Launch the baseline arithmetic with an explicit tile and return its kernel.

    ``grouped=False`` is a same-configuration row-major control. The candidate
    changes only program-ID ordering when ``grouped=True``. Neither path tunes
    or uses atomic split-K. Returned metadata/IR identifies GROUPED_M and tiles.
    """
    config = dict(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        group_m=group_m,
    )
    _validate_fixed_config(config)
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise ValueError("fixed GEMM requires matrices")
    M, K = a.shape
    if b.shape[0] != K or out.shape != (M, b.shape[1]):
        raise ValueError("incompatible fixed GEMM dimensions")
    N = b.shape[1]
    if min(M, N, K) == 0:
        raise ValueError("fixed GEMM requires nonempty matrices")
    if not (a.dtype == b.dtype == out.dtype == torch.bfloat16):
        raise ValueError("fixed GEMM requires BF16 inputs and output")
    if not (a.device == b.device == out.device):
        raise ValueError("fixed GEMM tensors must share a device")
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n), 1)
    # LibEntry exposes its underlying JIT function: bypass both its launch cache
    # and the tuner so fixed controls never search, even during Graph capture.
    with torch_device_fn.device(a.device):
        kernel = grouped_mm_kernel if grouped else mm_kernel.jit_function
        extra = {"GROUPED_M": True} if grouped else {}
        return kernel[grid](
            a,
            b,
            out,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            acc_dtype=tl.float32,
            input_precision=None,
            fp8_fast_accum=True,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=group_m,
            SPLIT_K=1,
            EVEN_K=K % block_k == 0,
            MASKED_LOAD=_requires_masked_load(a, b, M, N),
            AB_DTYPE=tl.bfloat16,
            UPGRADE=math.ceil(M * N / (block_m * block_n)).bit_length() > 31,
            UPGRADE_A_OFFS=(M * K).bit_length() > 31,
            UPGRADE_B_OFFS=(K * N).bit_length() > 31,
            UPGRADE_C_OFFS=(M * N).bit_length() > 31,
            num_warps=num_warps,
            num_stages=num_stages,
            **extra,
        )


def _candidate_config(a, b, c, M, N, K):
    if not _MM_CANDIDATE_CONFIGS:
        return None
    if not (a.dtype == b.dtype == c.dtype == torch.bfloat16):
        return None
    if a.stride() != (K, 1) or b.stride() != (1, K) or c.stride() != (N, 1):
        return None
    return _MM_CANDIDATE_CONFIGS.get((M, N, K))


def _launch_mm(a, b, c, M, N, K):
    """Launch Triton matmul _kernel; c must be pre-allocated."""
    config = _candidate_config(a, b, c, M, N, K)
    if config is not None:
        launch_mm_fixed(a, b, c, grouped=True, **config)
        return c
    ab_dtype = get_higher_dtype(a.dtype, b.dtype)
    acc_dtype_tl = tl.float32
    ab_dtype_tl = _to_tl_type(ab_dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )

    with torch_device_fn.device(a.device):
        mm_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            acc_dtype=acc_dtype_tl,
            input_precision=None,
            fp8_fast_accum=True,
            GROUP_M=8,
            AB_DTYPE=ab_dtype_tl,
            # Explicit constexpr is part of LibEntry's dispatch key. Pointer
            # divisibility alone does not distinguish 16- vs 128-byte alignment.
            MASKED_LOAD=_requires_masked_load(a, b, M, N),
        )
    return c


def mm(a, b):
    logger.debug("GEMS_ILUVATAR MM")
    device = a.device
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    return _launch_mm(a, b, c, M, N, K)


def mm_out(a, b, *, out):
    logger.debug("GEMS_ILUVATAR MM_OUT")
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    return _launch_mm(a, b, out, M, N, K)

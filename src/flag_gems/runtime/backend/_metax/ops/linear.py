# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Small-M BF16 forward linear with weight reuse across input rows.

Mechanism reference: TileOPs-Metax gemm_maca.py at 4df32b28 (MIT),
output-column blocking and FP32 reduction. This is an independent Triton
implementation; it does not require TileLang or change weight layout.
"""
import torch
import triton
import triton.language as tl

from flag_gems.ops.linear import linear as _generic_linear

# Filled only by the fixed independent confirmation, never runtime tuning.
SELECTED_CONFIGS = {(2048, 6144): (4, 512, 4), (12288, 2048): (4, 512, 4)}
ENABLED_SHAPES = ((1, 2048, 6144), (1, 12288, 2048), (2, 2048, 6144), (4, 2048, 6144))


@triton.jit
def small_m_linear_kernel(X, W, Y, M: tl.constexpr, N: tl.constexpr,
                          K: tl.constexpr, BM: tl.constexpr,
                          BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.arange(0, BM)
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    accum = tl.zeros((BM, BN, BK), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + ks
        x = tl.load(X + rows[:, None] * K + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K), 0).to(tl.float32)
        w = tl.load(W + cols[:, None] * K + k[None, :],
                    (cols[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
        accum += x[:, None, :] * w[None, :, :]
    value = tl.sum(accum, 2)
    tl.store(Y + rows[:, None] * N + cols[None, :], value,
             (rows[:, None] < M) & (cols[None, :] < N))


def dispatch_config(input, weight, bias=None):
    # Reject prefill and unsupported ranks/shapes before dtype/device checks.
    if bias is not None or input.ndim != 2 or weight.ndim != 2:
        return None
    m, k = input.shape
    if m not in (1, 2, 4, 7):
        return None
    n, wk = weight.shape
    if k != wk or (m, n, k) not in ENABLED_SHAPES:
        return None
    if (input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or input.device.type != "cuda" or weight.device != input.device
            or not input.is_contiguous() or not weight.is_contiguous()
            or input.requires_grad or weight.requires_grad):
        return None
    return SELECTED_CONFIGS.get((n, k))


def eligible(input, weight, bias=None):
    return dispatch_config(input, weight, bias) is not None


def candidate_linear(input, weight, config):
    """Explicit diagnostic entry; caller validates the frozen supported domain."""
    m, k = input.shape
    n = weight.shape[0]
    bn, bk, warps = config
    out = torch.empty((m, n), device=input.device, dtype=input.dtype)
    small_m_linear_kernel[(triton.cdiv(n, bn),)](
        input, weight, out, m, n, k, triton.next_power_of_2(m), bn, bk,
        num_warps=warps, num_stages=1, enable_fp_fusion=False)
    return out


def linear(input, weight, bias=None):
    config = dispatch_config(input, weight, bias)
    if config is not None:
        return candidate_linear(input, weight, config)
    return _generic_linear(input, weight, bias)

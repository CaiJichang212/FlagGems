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
SELECTED_CONFIGS = {}
ENABLED_SHAPES = ()


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


def eligible(input, weight, bias=None):
    return (bias is None and input.ndim == 2 and weight.ndim == 2
            and input.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
            and input.device.type == "cuda" and weight.device == input.device
            and input.is_contiguous() and weight.is_contiguous()
            and not input.requires_grad and not weight.requires_grad
            and input.shape[1] == weight.shape[1]
            and (input.shape[0], *weight.shape) in ENABLED_SHAPES)


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
    if eligible(input, weight, bias):
        return candidate_linear(input, weight, SELECTED_CONFIGS[tuple(weight.shape)])
    return _generic_linear(input, weight, bias)

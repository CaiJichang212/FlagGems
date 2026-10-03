# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Public BI-V150 GEMM regressions with an independent CPU FP32 oracle.

Run with the grouped candidate environment unset. These tests exercise the
default implementation and do not assume how a backend repairs its kernels.
"""

import os

import pytest
import torch

import flag_gems

pytestmark = [
    pytest.mark.mm,
    pytest.mark.skipif(flag_gems.vendor_name != "iluvatar", reason="Iluvatar only"),
]


@pytest.fixture(autouse=True)
def default_gemm():
    assert not os.environ.get("FLAGGEMS_ILUVATAR_MM_CONFIG"), (
        "Run default-path regressions without FLAGGEMS_ILUVATAR_MM_CONFIG"
    )


def _inputs(m, n, k, seed):
    # Generate on CPU so the oracle does not depend on any GPU matmul dispatch.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn((m, k), generator=generator, dtype=torch.bfloat16)
    b = torch.randn((n, k), generator=generator, dtype=torch.bfloat16).t()
    reference = a.float() @ b.float()
    return a.to(flag_gems.device), b.to(flag_gems.device), reference


def _check(result, reference, k):
    actual = result.detach().cpu()
    assert bool(torch.isfinite(actual).all()), "Non-finite or unwritten output"
    # Same BF16/K-scaled standard as scripts/flagos_gemm.py; no relaxed gate.
    torch.testing.assert_close(
        actual, reference.to(torch.bfloat16), atol=1e-4 * k, rtol=0.016
    )


def _guarded(shape):
    rows, columns = shape
    backing = torch.full(
        (rows + 2, columns + 7), 73.0,
        device=flag_gems.device, dtype=torch.bfloat16,
    )
    view = backing[1:rows + 1, 3:columns + 3]
    assert view.storage_offset() > 0
    return backing, view


def _check_guards(backing, rows, columns):
    assert bool((backing[0] == 73).all())
    assert bool((backing[-1] == 73).all())
    assert bool((backing[1:rows + 1, :3] == 73).all())
    assert bool((backing[1:rows + 1, columns + 3:] == 73).all())


@pytest.mark.parametrize("m", [1, 2, 3, 4, 7, 8, 15, 16])
def test_small_projection_repeated_public_calls(m):
    n, k = 2560, 2048
    backing, out = _guarded((m, n))
    for seed in (42, 43, 44):
        a, b, reference = _inputs(m, n, k, seed)
        _check(flag_gems.mm(a, b), reference, k)
        out.fill_(float("nan"))
        pointer = out.data_ptr()
        returned = flag_gems.mm_out(a, b, out=out)
        assert returned is out and out.data_ptr() == pointer
        _check(out, reference, k)
        _check_guards(backing, m, n)
    # A final zero input catches accumulation or stale values across calls.
    a.zero_()
    out.fill_(float("nan"))
    flag_gems.mm_out(a, b, out=out)
    torch.testing.assert_close(out.cpu(), torch.zeros_like(reference).bfloat16(), atol=0, rtol=0)
    _check_guards(backing, m, n)


def _offset_operand(value, padding, offset):
    rows, columns = value.shape
    backing = torch.full(
        (offset + rows * (columns + padding) + 64,), 73.0,
        device=value.device, dtype=value.dtype,
    )
    view = backing.as_strided((rows, columns), (columns + padding, 1), offset)
    view.copy_(value)
    assert view.storage_offset() == offset
    return backing, view


PADDED_CASES = [
    (7, 129, 64, layout, padding, offset)
    for layout in ("a_padded", "b_padded", "both_padded")
    for padding in (7, 64)
    for offset in (1, 8, 64)
] + [
    (2048, 2560, 2048, "a_padded", padding, 64)
    for padding in (7, 64)
]


@pytest.mark.parametrize("m,n,k,layout,padding,offset", PADDED_CASES)
def test_padded_operands_offsets_and_output_sentinels(m, n, k, layout, padding, offset):
    a, b, reference = _inputs(m, n, k, 71)
    inputs = []
    if layout in ("a_padded", "both_padded"):
        a_backing, a_view = _offset_operand(a, padding, offset)
        a = a_view
        inputs.append((a_backing, a_backing.clone()))
    if layout in ("b_padded", "both_padded"):
        b_backing, b_view = _offset_operand(b.t(), padding, offset)
        b = b_view.t()
        inputs.append((b_backing, b_backing.clone()))
    backing, out = _guarded((m, n))
    for scale in (1.0, -0.5):
        if scale != 1.0:
            a.mul_(scale)
            inputs = [(original, original.clone()) for original, _ in inputs]
        _check(flag_gems.mm(a, b), reference * scale, k)
        out.fill_(float("nan"))
        returned = flag_gems.mm_out(a, b, out=out)
        assert returned is out
        _check(out, reference * scale, k)
        _check_guards(backing, m, n)
        for original, snapshot in inputs:
            assert torch.equal(original, snapshot), "Input or its padding was written"


def test_contiguous_large_projection_repeat_and_out_agree():
    m, n, k = 2048, 2560, 2048
    a, b, reference = _inputs(m, n, k, 97)
    assert a.stride() == (k, 1) and b.stride() == (1, k)
    baseline = flag_gems.mm(a, b)
    _check(baseline, reference, k)
    out = torch.empty_like(baseline)
    for _ in range(3):
        out.fill_(float("nan"))
        assert flag_gems.mm_out(a, b, out=out) is out
        assert torch.equal(out, baseline), "Repeated public calls changed arithmetic"
        assert torch.equal(flag_gems.mm(a, b), baseline)


def test_same_shape_stride_cache_distinguishes_pointer_alignment():
    """BF16 offset 8 is 16-byte aligned but breaks 128-byte SME alignment."""
    m, n, k = 32, 256, 256
    a, b, reference = _inputs(m, n, k, 113)
    assert a.data_ptr() % 128 == b.data_ptr() % 128 == 0
    a_backing, offset_a = _offset_operand(a, padding=0, offset=8)
    b_backing, offset_bt = _offset_operand(b.t(), padding=0, offset=8)
    offset_b = offset_bt.t()
    assert offset_a.stride() == a.stride()
    assert offset_b.stride() == b.stride()
    assert offset_a.data_ptr() % 128 == offset_b.data_ptr() % 128 == 16
    # All pointers have the same ordinary 16-byte specialization. Switching
    # only the tensor base address must still select the correct load policy.
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    snapshots = [(a_backing, a_backing.clone()), (b_backing, b_backing.clone())]
    for unaligned in ((offset_a, b), (a, offset_b), (offset_a, offset_b)):
        for aa, bb in ((a, b), unaligned, (a, b)):
            _check(flag_gems.mm(aa, bb), reference, k)
            out.fill_(float("nan"))
            assert flag_gems.mm_out(aa, bb, out=out) is out
            _check(out, reference, k)
    for backing, snapshot in snapshots:
        assert torch.equal(backing, snapshot)

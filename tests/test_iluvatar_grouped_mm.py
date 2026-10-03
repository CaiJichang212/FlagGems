# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit, opt-in checks for the BI-V150 grouped projection candidate."""

import importlib

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .test_mm import _mm_atol_base

pytestmark = [
    pytest.mark.mm,
    pytest.mark.skipif(flag_gems.vendor_name != "iluvatar", reason="Iluvatar only"),
]
CONFIG = dict(
    block_m=64, block_n=64, block_k=32, num_warps=8, num_stages=1, group_m=8
)
PROJECTIONS = [(2560, 2048), (2048, 2048), (12288, 2048), (2048, 6144)]


@pytest.fixture
def impl():
    module = importlib.import_module(flag_gems.mm.__module__)
    assert module.__dict__ is flag_gems.mm.__globals__
    assert flag_gems.mm_out.__globals__ is module.__dict__
    saved = module._MM_CANDIDATE_CONFIGS
    module.configure_mm_candidate({})
    yield module
    module.configure_mm_candidate(saved)


def _inputs(M, N, K):
    a = torch.randn(M, K, dtype=torch.bfloat16, device=flag_gems.device)
    b = torch.randn(N, K, dtype=torch.bfloat16, device=flag_gems.device).t()
    return a, b


def _assert_reference(result, a, b):
    # Use precisely the upstream mm reduction-scaled tolerance.
    reference = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))
    utils.gems_assert_close(
        result, reference, a.dtype, reduce_dim=a.shape[1], atol=_mm_atol_base()
    )


@pytest.mark.parametrize("N,K", PROJECTIONS)
def test_grouped_projection_exact(impl, N, K):
    a, b = _inputs(2048, N, K)
    baseline = torch.empty(2048, N, device=a.device, dtype=a.dtype)
    candidate = torch.full_like(baseline, float("nan"))
    impl.launch_mm_fixed(a, b, baseline, grouped=False, **CONFIG)
    compiled = impl.launch_mm_fixed(a, b, candidate, grouped=True, **CONFIG)
    assert "grouped_mm_kernel" in compiled.name
    assert torch.equal(candidate, baseline)
    _assert_reference(candidate, a, b)


@pytest.mark.parametrize("M,N,K", [(1, 33, 31), (65, 129, 33), (513, 131, 65)])
def test_grouped_tail_exact(impl, M, N, K):
    a, b = _inputs(M, N, K)
    control = torch.empty(M, N, device=a.device, dtype=a.dtype)
    result = torch.full_like(control, float("nan"))
    impl.launch_mm_fixed(a, b, control, grouped=False, **CONFIG)
    impl.launch_mm_fixed(a, b, result, grouped=True, **CONFIG)
    assert torch.equal(result, control)
    _assert_reference(result, a, b)


def test_grouped_public_dispatch_and_fallback(impl, monkeypatch):
    a, b = _inputs(2048, 2560, 2048)
    calls = []
    original = impl.launch_mm_fixed

    def tracked(*args, **kwargs):
        calls.append(kwargs["grouped"])
        return original(*args, **kwargs)

    monkeypatch.setattr(impl, "launch_mm_fixed", tracked)
    impl.configure_mm_candidate({(2048, 2560, 2048): CONFIG})
    result = impl.mm(a, b)
    out = torch.full_like(result, float("nan"))
    pointer = out.data_ptr()
    returned = impl.mm_out(a, b, out=out)
    assert returned is out and out.data_ptr() == pointer
    assert calls == [True, True]
    assert torch.equal(result, out)
    _assert_reference(out, a, b)
    # Small M, row-major weights and strided out must retain the original path.
    for aa, bb, cc in [
        (a[:1], b, torch.empty_like(out[:1])),
        (a, b.contiguous(), torch.empty_like(out)),
        (a, b, torch.empty(2560, 2048, device=a.device, dtype=a.dtype).t()),
    ]:
        calls.clear()
        impl.mm_out(aa, bb, out=cc)
        assert calls == []
        _assert_reference(cc, aa, bb)
    impl.configure_mm_candidate({})
    calls.clear()
    impl.mm_out(a, b, out=out)
    assert calls == []


@pytest.mark.parametrize("M,N,K", [(65, 129, 33), (2048, 2560, 2048)])
def test_grouped_graph_repeated_input_and_full_write(impl, M, N, K):
    a, b = _inputs(M, N, K)
    out = torch.empty(M, N, device=a.device, dtype=a.dtype)
    pointer = out.data_ptr()
    # Compile and warm on a side stream before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            impl.launch_mm_fixed(a, b, out, grouped=True, **CONFIG)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(2):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            impl.launch_mm_fixed(a, b, out, grouped=True, **CONFIG)
        for _ in range(3):
            a.normal_()
            out.fill_(float("nan"))
            graph.replay()
            assert out.data_ptr() == pointer
            expected = torch.empty_like(out)
            impl.launch_mm_fixed(a, b, expected, grouped=False, **CONFIG)
            assert torch.equal(out, expected)
            _assert_reference(out, a, b)

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.logaddexp import logaddexp as gems_logaddexp
from flag_gems.experimental_ops.logaddexp import logaddexp_out as gems_logaddexp_out

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close
except ImportError:
    TO_CPU = False

    def gems_assert_close(res, ref, dtype, **kwargs):
        torch.testing.assert_close(res, ref, equal_nan=True, **kwargs)


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    if TO_CPU:
        ref_inp = inp.to("cpu")
    else:
        ref_inp = inp.clone()
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.logaddexp
@pytest.mark.parametrize(
    "shape", [(2, 3), (128, 256), (512, 512), (1024, 1024), (4096, 4096)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_logaddexp_tensor(shape, dtype):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = to_reference(input)
    ref_other = to_reference(other)
    ref_out = torch.ops.aten.logaddexp(ref_input, ref_other)

    with flag_gems.use_gems():
        act_out = gems_logaddexp(input, other)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.logaddexp
@pytest.mark.parametrize("shape", [(128, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_logaddexp_out(shape, dtype):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = to_reference(input)
    ref_other = to_reference(other)

    ref_out_buf = torch.empty(shape, dtype=dtype, device=ref_input.device)
    ref_out = torch.ops.aten.logaddexp.out(ref_input, ref_other, out=ref_out_buf)

    act_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        act_out = gems_logaddexp_out(input, other, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.logaddexp
@pytest.mark.parametrize(
    "shape_input, shape_other",
    [
        ((1, 128), (512, 128)),
        ((512, 1), (512, 128)),
        ((128,), (512, 128)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_logaddexp_broadcast(shape_input, shape_other, dtype):
    input = torch.randn(shape_input, dtype=dtype, device=flag_gems.device)
    other = torch.randn(shape_other, dtype=dtype, device=flag_gems.device)

    ref_input = to_reference(input)
    ref_other = to_reference(other)
    ref_out = torch.ops.aten.logaddexp(ref_input, ref_other)

    with flag_gems.use_gems():
        act_out = gems_logaddexp(input, other)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.logaddexp
@pytest.mark.parametrize("dtype", [torch.float32])
def test_logaddexp_special_values(dtype):
    input = torch.tensor(
        [1.0, float("inf"), float("-inf"), float("nan"), 0.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    other = torch.tensor(
        [float("inf"), 1.0, float("-inf"), 0.0, float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )

    ref_input = to_reference(input)
    ref_other = to_reference(other)
    ref_out = torch.ops.aten.logaddexp(ref_input, ref_other)

    with flag_gems.use_gems():
        act_out = gems_logaddexp(input, other)

    gems_assert_close(act_out, ref_out, dtype=dtype)

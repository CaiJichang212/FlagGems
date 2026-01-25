import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    POINTWISE_SHAPES,
    QUICK_MODE,
    gems_assert_close,
    to_reference,
)


@pytest.mark.log10
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_log10(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(inp, True)
    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_log10_size_coverage(dtype):
    shapes = [(1, 1), (8, 8), (64, 64), (256, 256), (1024, 1024)]
    if QUICK_MODE:
        shapes = [(1, 1), (8, 8)]
    for shape in shapes:
        inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = to_reference(inp, True)
        ref_out = torch.log10(ref_inp)
        with flag_gems.use_gems():
            res_out = torch.log10(inp)
        gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_log10_edge_values(dtype):
    values = torch.tensor(
        [0.0, 1.0, 10.0, -1.0, 100.0], dtype=dtype, device=flag_gems.device
    )
    ref_inp = to_reference(values, True)
    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(values)
    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_log10_empty(dtype):
    inp = torch.empty((0,), dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)
    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_log10_noncontiguous(dtype):
    base = torch.rand((4, 5, 6), dtype=dtype, device=flag_gems.device)
    inp = base.transpose(0, 2)
    ref_inp = to_reference(inp, True)
    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.log10_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_log10_(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.clone(), True)

    ref_out = torch.log10_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10_(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_log10_out(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.log10(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.empty_like(inp)
        torch.log10(inp, out=res_out)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10
def test_log10_invalid_dtype():
    inp = torch.randint(0, 10, (4,), dtype=torch.int32, device=flag_gems.device)
    with flag_gems.use_gems():
        with pytest.raises(TypeError):
            torch.log10(inp)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_log10_out_invalid_shape(dtype):
    inp = torch.rand((4, 4), dtype=dtype, device=flag_gems.device)
    out = torch.empty((4, 5), dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        with pytest.raises(RuntimeError):
            torch.log10(inp, out=out)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_log10_out_invalid_dtype(dtype):
    inp = torch.rand((4, 4), dtype=dtype, device=flag_gems.device)
    out = torch.empty((4, 4), dtype=torch.float32, device=flag_gems.device)
    if out.dtype == inp.dtype:
        out = torch.empty((4, 4), dtype=torch.float16, device=flag_gems.device)
    with flag_gems.use_gems():
        with pytest.raises(TypeError):
            torch.log10(inp, out=out)

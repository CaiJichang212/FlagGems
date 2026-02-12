import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    gems_assert_close,
    to_reference,
)
from .conftest import QUICK_MODE

SVD_SHAPES = (
    [(8, 8), (16, 32), (32, 16)]
    if QUICK_MODE
    else [
        (8, 8),
        (16, 32),
        (32, 16),
        (64, 64),
        (128, 256),
        (256, 128),
        (512, 512),
        (1024, 1024),
    ]
)

BATCH_SVD_SHAPES = (
    [(2, 8, 8), (4, 16, 32)]
    if QUICK_MODE
    else [
        (2, 8, 8),
        (4, 16, 32),
        (8, 32, 16),
        (16, 64, 64),
        (8, 128, 256),
        (4, 256, 128),
    ]
)

FLOAT_DTYPES = [torch.float32] if QUICK_MODE else FLOAT_DTYPES


@pytest.mark.svd
@pytest.mark.parametrize("shape", SVD_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("some", [True, False])
def test_accuracy_svd_basic(shape, dtype, some):
    M, N = shape
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = to_reference(A, True)
    
    ref_U, ref_S, ref_V = torch.svd(ref_A, some=some, compute_uv=True)
    
    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(A, some=some, compute_uv=True)
    
    K = min(M, N)
    if some:
        assert res_U.shape == (M, K)
        assert res_S.shape == (K,)
        assert res_V.shape == (N, K)
    else:
        assert res_U.shape == (M, M)
        assert res_S.shape == (K,)
        assert res_V.shape == (N, N)
    
    gems_assert_close(res_S, ref_S, dtype, atol=1e-2)
    
    if some:
        reconstructed = torch.matmul(res_U, torch.matmul(torch.diag(res_S), res_V.t()))
    else:
        reconstructed = torch.matmul(
            res_U[:, :K], torch.matmul(torch.diag(res_S), res_V[:, :K].t())
        )
    
    ref_A_compute = ref_A.to(torch.float32) if dtype in [torch.float16, torch.bfloat16] else ref_A.float()
    reconstructed_compute = reconstructed.to(torch.float32) if dtype in [torch.float16, torch.bfloat16] else reconstructed.float()
    
    torch.testing.assert_close(reconstructed_compute, ref_A_compute, rtol=5e-2, atol=5e-2)


@pytest.mark.svd
@pytest.mark.parametrize("shape", BATCH_SVD_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("some", [True, False])
def test_accuracy_svd_batch(shape, dtype, some):
    batch_size, M, N = shape
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = to_reference(A, True)
    
    ref_U, ref_S, ref_V = torch.svd(ref_A, some=some, compute_uv=True)
    
    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(A, some=some, compute_uv=True)
    
    K = min(M, N)
    if some:
        assert res_U.shape == (batch_size, M, K)
        assert res_S.shape == (batch_size, K)
        assert res_V.shape == (batch_size, N, K)
    else:
        assert res_U.shape == (batch_size, M, M)
        assert res_S.shape == (batch_size, K)
        assert res_V.shape == (batch_size, N, N)
    
    gems_assert_close(res_S, ref_S, dtype, atol=1e-2)


@pytest.mark.svd
@pytest.mark.parametrize("shape", SVD_SHAPES[:2])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_svd_no_uv(shape, dtype):
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = to_reference(A, True)
    
    ref_U, ref_S, ref_V = torch.svd(ref_A, some=True, compute_uv=False)
    
    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(A, some=True, compute_uv=False)
    
    M, N = shape
    K = min(M, N)
    
    assert res_S.shape == (K,)
    gems_assert_close(res_S, ref_S, dtype, atol=1e-2)
    
    assert res_U.shape == ref_U.shape
    assert res_V.shape == ref_V.shape
    assert torch.allclose(res_U, torch.zeros_like(res_U))
    assert torch.allclose(res_V, torch.zeros_like(res_V))


@pytest.mark.svd
@pytest.mark.parametrize("shape", [(8, 8), (16, 32)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_svd_singular_values_order(shape, dtype):
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = to_reference(A, True)
    
    _, ref_S, _ = torch.svd(ref_A, some=True, compute_uv=True)
    
    with flag_gems.use_gems():
        _, res_S, _ = torch.svd(A, some=True, compute_uv=True)
    
    for i in range(len(res_S) - 1):
        assert res_S[i] >= res_S[i + 1], f"Singular values not in descending order at index {i}"


@pytest.mark.svd
@pytest.mark.parametrize("shape", [(8, 8), (16, 32)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_svd_orthogonality(shape, dtype):
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    
    with flag_gems.use_gems():
        U, S, V = torch.svd(A, some=False, compute_uv=True)
    
    M, N = shape
    K = min(M, N)
    
    U_orth = torch.matmul(U.t(), U)
    eye_M = torch.eye(M, dtype=dtype, device=flag_gems.device)
    gems_assert_close(U_orth, eye_M, dtype, atol=1e-3)
    
    V_orth = torch.matmul(V.t(), V)
    eye_N = torch.eye(N, dtype=dtype, device=flag_gems.device)
    gems_assert_close(V_orth, eye_N, dtype, atol=1e-3)

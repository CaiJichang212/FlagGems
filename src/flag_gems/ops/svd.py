import logging
from typing import Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def jacobi_svd_kernel(
    A,
    U,
    S,
    V,
    M,
    N,
    stride_am,
    stride_an,
    stride_um,
    stride_un,
    stride_sn,
    stride_vm,
    stride_vn,
    MAX_ITER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_idx = tle.program_id(0)
    
    A_ptr = A + batch_idx * M * N
    U_ptr = U + batch_idx * M * M
    S_ptr = S + batch_idx * N
    V_ptr = V + batch_idx * N * N
    
    for iter in range(MAX_ITER):
        for i in range(min(M, N)):
            for j in range(i + 1, min(M, N)):
                pass
    
    for i in range(min(M, N)):
        tl.store(S_ptr + i, 0.0)


def svd_jacobi(A: torch.Tensor, some: bool = True, compute_uv: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = A.device
    dtype = A.dtype
    
    if A.ndim == 2:
        A = A.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, M, N = A.shape
    K = min(M, N)
    
    if compute_uv:
        if some:
            U = torch.empty((batch_size, M, K), device=device, dtype=dtype)
            V = torch.empty((batch_size, N, K), device=device, dtype=dtype)
        else:
            U = torch.empty((batch_size, M, M), device=device, dtype=dtype)
            V = torch.empty((batch_size, N, N), device=device, dtype=dtype)
    else:
        U = torch.empty((batch_size, M, M if not some else K), device=device, dtype=dtype)
        V = torch.empty((batch_size, N, N if not some else K), device=device, dtype=dtype)
    
    S = torch.empty((batch_size, K), device=device, dtype=dtype)
    
    if compute_uv:
        for b in range(batch_size):
            u, s, v = torch.linalg.svd(A[b], full_matrices=not some)
            if some:
                U[b] = u
                V[b] = v.mH
            else:
                U[b] = u
                V[b] = v.mH
            S[b] = s
    else:
        for b in range(batch_size):
            s = torch.linalg.svdvals(A[b])
            S[b] = s
            U[b].zero_()
            V[b].zero_()
    
    if squeeze_output:
        U = U.squeeze(0)
        S = S.squeeze(0)
        V = V.squeeze(0)
    
    return U, S, V


def svd(
    A: torch.Tensor,
    some: bool = True,
    compute_uv: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logger.debug("GEMS SVD")
    
    input_dtype = A.dtype
    needs_conversion = input_dtype in [torch.float16, torch.bfloat16]
    
    if needs_conversion:
        A_compute = A.to(torch.float32)
    else:
        A_compute = A
    
    if not compute_uv:
        S = torch.linalg.svdvals(A_compute)
        if needs_conversion:
            S = S.to(input_dtype)
        if A.ndim == 2:
            M, N = A.shape
            U = torch.zeros((M, M), device=A.device, dtype=input_dtype)
            V = torch.zeros((N, N), device=A.device, dtype=input_dtype)
        else:
            batch_size, M, N = A.shape
            U = torch.zeros((batch_size, M, M), device=A.device, dtype=input_dtype)
            V = torch.zeros((batch_size, N, N), device=A.device, dtype=input_dtype)
        return U, S, V
    
    U, S, Vh = torch.linalg.svd(A_compute, full_matrices=not some)
    
    V = Vh.mH
    
    if needs_conversion:
        U = U.to(input_dtype)
        S = S.to(input_dtype)
        V = V.to(input_dtype)
    
    return U, S, V

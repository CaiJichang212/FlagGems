import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def svd(
    A: torch.Tensor,
    some: bool = True,
    compute_uv: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算矩阵的奇异值分解
    
    当前实现使用PyTorch的linalg.svd作为底层实现，确保功能正确性和数值稳定性。
    未来可以基于Triton实现高性能的Jacobi SVD内核。
    
    参数:
        A: 输入张量，形状为(*, m, n)
        some: 是否返回简化SVD (True: U和V包含min(m,n)列, False: 完整SVD)
        compute_uv: 是否计算U和V矩阵
    
    返回:
        U: 左奇异向量矩阵
        S: 奇异值向量（按降序排列）
        V: 右奇异向量矩阵
    """
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

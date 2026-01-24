import logging

import torch
import triton
import triton.language as tl

from flag_gems.config import use_c_extension
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# enum Fp8KVCacheDataType
FP8_KV_CACHE_DATA_TYPE_AUTO = tl.constexpr(0)
FP8_KV_CACHE_DATA_TYPE_FP8E4M3 = tl.constexpr(1)
FP8_KV_CACHE_DATA_TYPE_FP8E5M2 = tl.constexpr(2)


@libentry()
@triton.jit
def reshape_and_cache_flash_kernel(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    block_stride,
    key_stride,
    value_stride,
    num_heads,
    head_size,
    block_size,
    k_scale_ptr,
    v_scale_ptr,
    kv_dtype: tl.constexpr,
    n: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping + token_idx)
    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    i = tl.arange(0, triton.next_power_of_2(n))
    mask = i < n

    src_key_idx = token_idx * key_stride + i
    src_value_idx = token_idx * value_stride + i
    head_idx = i // head_size
    head_offset = i % head_size
    tgt_key_value_idx = (
        block_idx * block_stride
        + block_offset * num_heads * head_size
        + head_idx * head_size
        + head_offset
    )

    tgt_key = tl.load(key + src_key_idx, mask=mask)
    tgt_value = tl.load(value + src_value_idx, mask=mask)

    if kv_dtype != FP8_KV_CACHE_DATA_TYPE_AUTO:
        key_scale = tl.load(k_scale_ptr + head_idx, mask=mask, other=1.0)
        value_scale = tl.load(v_scale_ptr + head_idx, mask=mask, other=1.0)
        if kv_dtype == FP8_KV_CACHE_DATA_TYPE_FP8E4M3:
            tgt_key = (tgt_key / key_scale).to(tl.float8e4nv)
            tgt_value = (tgt_value / value_scale).to(tl.float8e4nv)
        elif kv_dtype == FP8_KV_CACHE_DATA_TYPE_FP8E5M2:
            tgt_key = (tgt_key / key_scale).to(tl.float8e5)
            tgt_value = (tgt_value / value_scale).to(tl.float8e5)
        tgt_key = tgt_key.to(tl.uint8, bitcast=True)
        tgt_value = tgt_value.to(tl.uint8, bitcast=True)
    tl.store(key_cache + tgt_key_value_idx, tgt_key, mask=mask)
    tl.store(value_cache + tgt_key_value_idx, tgt_value, mask=mask)


def reshape_and_cache_flash(
    key,  # [num_tokens, num_heads, head_size]
    value,  # [num_tokens, num_heads, head_size]
    key_cache,  # [num_blocks, block_size, num_heads, head_size]
    value_cache,  # [num_blocks, block_size, num_heads, head_size]
    slot_mapping,  # [num_tokens]
    kv_cache_dtype,
    k_scale,
    v_scale,
):
    kv_dtype_map = {
        "auto": FP8_KV_CACHE_DATA_TYPE_AUTO,
        "fp8": FP8_KV_CACHE_DATA_TYPE_FP8E4M3,
        "fp8e4m3": FP8_KV_CACHE_DATA_TYPE_FP8E4M3,
        "fp8e5m2": FP8_KV_CACHE_DATA_TYPE_FP8E5M2,
    }
    kv_dtype = kv_dtype_map.get(kv_cache_dtype)
    if kv_dtype is None:
        raise ValueError(f"Unsupported kv_cache_dtype: {kv_cache_dtype}")

    if kv_cache_dtype != "auto":
        if key_cache.dtype != torch.uint8 or value_cache.dtype != torch.uint8:
            raise ValueError("For FP8 kv_cache must be uint8 dtype")

    if use_c_extension:
        logger.debug("GEMS RESHAPE_AND_CACHE_FLASH(C EXTENSION)")
        torch.ops.flag_gems.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
    else:
        logger.debug("GEMS RESHAPE_AND_CACHE_FLASH")
        num_tokens = slot_mapping.size(0)
        num_heads = key.size(1)
        head_size = key.size(2)
        block_size = key_cache.size(1)

        if kv_cache_dtype != "auto":
            if key_cache.dtype != torch.uint8 or value_cache.dtype != torch.uint8:
                raise ValueError("For FP8 kv_cache must be uint8 dtype")

            # Normalize scales to per-head tensors on device
            if not isinstance(k_scale, torch.Tensor):
                k_scale = torch.tensor(k_scale, dtype=torch.float32, device=key.device)
            if not isinstance(v_scale, torch.Tensor):
                v_scale = torch.tensor(v_scale, dtype=torch.float32, device=key.device)
            k_scale = k_scale.to(device=key.device, dtype=torch.float32).flatten()
            v_scale = v_scale.to(device=key.device, dtype=torch.float32).flatten()
            if k_scale.numel() == 1:
                k_scale = k_scale.expand(num_heads)
            if v_scale.numel() == 1:
                v_scale = v_scale.expand(num_heads)
            if k_scale.numel() != num_heads or v_scale.numel() != num_heads:
                raise ValueError(
                    "k_scale and v_scale must be scalar or length num_heads for FP8 kv cache"
                )
        else:
            # Dummy placeholders for kernel signature.
            k_scale = torch.empty((num_heads,), dtype=torch.float32, device=key.device)
            v_scale = torch.empty((num_heads,), dtype=torch.float32, device=key.device)

        key_stride = key.stride(0)
        value_stride = value.stride(0)
        block_stride = key_cache.stride(0)
        assert key_cache.stride(0) == value_cache.stride(0)

        grid = (num_tokens,)
        with torch_device_fn.device(key.device):
            reshape_and_cache_flash_kernel[grid](
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                block_stride,
                key_stride,
                value_stride,
                num_heads,
                head_size,
                block_size,
                k_scale,
                v_scale,
                num_heads * head_size,
                kv_dtype=int(kv_dtype),
            )

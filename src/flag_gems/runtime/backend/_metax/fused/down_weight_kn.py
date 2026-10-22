# SPDX-License-Identifier: Apache-2.0
"""Private persistent KxN down weight layout; arithmetic and K order unchanged."""
import importlib
import torch
import triton
from flag_gems.runtime import torch_device_fn
from .gate_up_silu import _baseline_record


def tensor_stamp(weight):
    # Inference tensors without a version counter cannot safely be cached.
    if not (type(weight) is torch.Tensor or isinstance(weight,torch.nn.Parameter)):
        return None
    try:
        version=weight._version
    except RuntimeError:
        return None
    return (id(weight),weight.data_ptr(),version,tuple(weight.shape),weight.stride(),weight.dtype,weight.device)


def pack_down_weight(weight):
    stamp=tensor_stamp(weight)
    if (stamp is None or weight.device.type!='cuda' or weight.dtype!=torch.bfloat16
            or tuple(weight.shape)!=(2048,6144) or not weight.is_contiguous()
            or weight.requires_grad):
        return None
    with torch.inference_mode(False),torch.no_grad():
        packed=weight.detach().T.contiguous()
    return [packed,stamp,tensor_stamp(packed),True]


def try_down_kn(x,weight,record):
    if record is None or record[3] is not True:
        return None
    packed,old,packed_old,_=record
    if tensor_stamp(weight)!=old or tensor_stamp(packed)!=packed_old:
        return None
    if (type(x) is not torch.Tensor or tuple(x.shape)!=(4,6144) or x.device!=weight.device
            or x.dtype!=torch.bfloat16 or not x.is_contiguous() or x.data_ptr()%16
            or x.requires_grad or not torch.is_inference_mode_enabled() or torch.is_grad_enabled()
            or torch.compiler.is_compiling()):
        return None
    meta=_baseline_record(x,weight)
    if meta is None:
        return None
    # Invoke the exact existing generic JIT with its selected tile. Only W strides
    # change: original W[N,K] vs prepacked W[K,N]. No public linear replacement.
    generic=importlib.import_module('flag_gems.ops.linear')
    out=torch.empty((4,2048),device=x.device,dtype=x.dtype)
    grid=(triton.cdiv(4,meta['BLOCK_SIZE_M']),triton.cdiv(2048,meta['BLOCK_SIZE_N']))
    with torch_device_fn.device(x.device):
        generic.linear_kernel.jit_function[grid](x,packed,packed,out,4,2048,6144,
            6144,1,1,2048,2048,1,0,BIAS=False,**meta)
    return out

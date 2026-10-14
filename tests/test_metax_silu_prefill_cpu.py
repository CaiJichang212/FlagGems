"""CPU lifecycle contracts for the isolated MetaX activation candidate."""
import ast
from collections import Counter,OrderedDict
import contextlib
import os
from pathlib import Path
import threading
import types
import unittest

SOURCE=Path(__file__).resolve().parents[1]/'src/flag_gems/runtime/backend/_metax/fused/silu_and_mul.py'
class Tensor:
    nextptr=256
    def __init__(self,shape=(4,6144),stride=None,ptr=None,grad=False,dtype='bf16'):
        self.shape=shape;self.ndim=len(shape);self._stride=stride or (2*shape[-1],1)
        self.device=types.SimpleNamespace(type='cuda',index=0);self.dtype=dtype;self.requires_grad=grad
        Tensor.nextptr+=256;self.ptr=ptr if ptr is not None else Tensor.nextptr
    def stride(self):return self._stride
    def data_ptr(self):return self.ptr

class SiluLifecycle(unittest.TestCase):
    def setUp(self):
        tree=ast.parse(SOURCE.read_text());tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
        self.launches=[];self.cold=[];self.stream=1;self.inference=True
        class Pointwise:pass
        self.pw=Pointwise();self.pw.config=types.SimpleNamespace(max_tile_size=2048,max_grid_size=(65536,)*3,max_num_warps_per_cta=8,prefer_block_pointer=True,prefer_1d_tile=False)
        self.pw._scalar_fn=object();self.pw._scalar_fn_cache_key='scalar';self.pw.fx=types.SimpleNamespace(schema='original');self.pw.complex_strategy=types.SimpleNamespace(mode='none')
        self.overload=lambda:None
        self.pw.overloads={'2_True':self.overload};self.pw.get_kernel_info=lambda n:types.SimpleNamespace(kernel_name='test_silu_entry')
        self.entry=types.SimpleNamespace(fn=object(),kernel_cache=({},),_has_flagtune_tuner=False)
        self.entry.jit_function=self.entry.fn;self.overload.__globals__['test_silu_entry']=self.entry
        self.record=(object(),{},{},{},())
        def allocate(shape,device,dtype):return Tensor(tuple(shape),stride=(shape[-1],1),dtype=dtype)
        def generic(x,y):
            out=allocate(x.shape,x.device,x.dtype);self.cold.append((x,y,out));self.entry.kernel_cache[0]['key']=self.record;return out
        self.generic=types.SimpleNamespace(silu_and_mul=generic,silu_and_mul_kernel=self.pw)
        self.ns=dict(torch=types.SimpleNamespace(Tensor=Tensor,bfloat16='bf16',empty=allocate,is_inference_mode_enabled=lambda:self.inference),
            triton=types.SimpleNamespace(__version__='3.0.0'),torch_device_fn=types.SimpleNamespace(device=lambda d:contextlib.nullcontext()),
            importlib=types.SimpleNamespace(import_module=lambda n:self.generic),os=os,threading=threading,Counter=Counter,OrderedDict=OrderedDict)
        exec(compile(tree,str(SOURCE),'exec'),self.ns)
        def descriptor(p,x,y,out,metadata=None):
            if self.entry._has_flagtune_tuner:return None
            def runner(*args):self.launches.append((args,self.stream))
            return (self.overload,self.entry,self.entry.jit_function,'key',self.record,runner,(4,6144,24576,1),self.ns['_stamp'](p))
        self.ns['_descriptor']=descriptor;self.fn=self.ns['silu_and_mul'];self.ns['diagnostics'](True)
    def inputs(self,**kw):return Tensor(**kw),Tensor(**kw)
    def test_cold_once_fresh_inputs_output_stream(self):
        x,y=self.inputs();a=self.fn(x,y);self.stream=2;x2,y2=self.inputs();b=self.fn(x2,y2)
        self.assertEqual(len(self.cold),1);self.assertEqual(len(self.launches),1)
        self.assertIsNot(a,b);self.assertIs(self.launches[0][0][0],x2);self.assertEqual(self.launches[0][1],2)
        record=next(iter(self.ns['_cache']().values()));self.assertFalse(any(isinstance(v,Tensor) for v in record))
    def test_dynamic_changes_never_use_stale_runner(self):
        for mutate in (lambda:self.entry.kernel_cache[0].clear(),lambda:self.pw.overloads.clear(),lambda:setattr(self.entry,'_has_flagtune_tuner',True),lambda:setattr(self.pw.config,'max_tile_size',1024),lambda:setattr(self.pw.fx,'schema','changed'),lambda:setattr(self.entry,'fn',object())):
            self.setUp();x,y=self.inputs();self.fn(x,y);mutate();self.fn(x,y)
            self.assertEqual(len(self.launches),0);self.assertEqual(len(self.cold),2)
    def test_fallbacks(self):
        for kw in ({'shape':(1025,6144)},{'stride':(6144,1)},{'ptr':258},{'grad':True},{'dtype':'fp32'}):self.fn(*self.inputs(**kw))
        self.inference=False;self.fn(*self.inputs());self.assertEqual(len(self.cold),6);self.assertFalse(self.launches)
    def test_thread_pid_eviction(self):
        self.fn(*self.inputs());t=threading.Thread(target=lambda:self.fn(*self.inputs()));t.start();t.join()
        self.assertEqual(len(self.cold),2);self.ns['_local'].pid=-1;self.fn(*self.inputs());self.assertEqual(len(self.cold),3)
        for n in range(32,32+65):self.fn(*self.inputs(shape=(4,n)))
        self.assertEqual(len(self.ns['_cache']()),64)

    def test_generic_row_boundaries_and_non_model_tail_widths(self):
        for m in (1,7,8,14,25,35,1023,1024):
            for n in (1,17,6143,6144,6145):
                self.setUp();x,y=self.inputs(shape=(m,n))
                a=self.fn(x,y);b=self.fn(x,y)
                self.assertEqual(len(self.cold),1)
                self.assertEqual(len(self.launches),1)
                self.assertIsNot(a,b)
                self.assertIs(self.launches[0][0][0],x)
                self.assertTrue(self.ns['_supported'](x,y))
    def test_unsupported_row_bounds_rank_and_layout(self):
        for kw in ({'shape':(0,6144)},{'shape':(1025,6144)},{'shape':(14,0)},
                   {'shape':(14,17),'stride':(35,1)},{'shape':(8,17),'stride':(0,1)},
                   {'shape':(8,17),'ptr':258},{'shape':(14,17),'grad':True}):
            self.setUp();self.fn(*self.inputs(**kw));self.fn(*self.inputs(**kw))
            self.assertEqual(len(self.cold),2);self.assertFalse(self.launches)
        self.setUp();x,y=self.inputs();y.shape=(8,6144)
        self.assertFalse(self.ns['_supported'](x,y))
    def test_large_row_current_stream_and_dynamic_invalidation(self):
        for m in (8,1024):
            self.setUp();x,y=self.inputs(shape=(m,6144));self.fn(x,y);self.stream=9
            self.fn(x,y);self.assertEqual(self.launches[0][1],9)
            self.entry.kernel_cache[0].clear();self.fn(x,y)
            self.assertEqual(len(self.launches),1);self.assertEqual(len(self.cold),2)

if __name__=='__main__':unittest.main()

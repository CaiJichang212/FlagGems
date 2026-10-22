"""CPU contracts; fake runner mutates twice detectably, never initializes GPU."""
import ast
from collections import Counter, OrderedDict
from contextlib import nullcontext
import math
import os
from pathlib import Path
import threading
import types
import unittest

import json
SOURCE=Path(__file__).resolve().parents[1]/'src/flag_gems/runtime/backend/_metax/fused/fused_add_rms_norm.py'

class Device:
    type='cuda'
    def __init__(self,index=0):self.index=index
    def __eq__(self,o):return self.index==o.index
class Tensor:
    seq=0
    def __init__(self,shape=(4,2048),device=0):
        Tensor.seq+=1;self.ptr=Tensor.seq*2**24;self.shape=shape;self.device=Device(device);self.dtype='bf16';self.requires_grad=False;self.contiguous=True;self.updates=0
    def data_ptr(self):return self.ptr
    def numel(self):return math.prod(self.shape)
    def element_size(self):return 2
    def is_contiguous(self):return self.contiguous
class Parameter(Tensor):pass
class JITFunction:
    def run(self):pass
class LibEntry:
    def run(self):pass

class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.cold=[];self.hot=[];self.stream=1;self.grad_enabled=True
        jit=JITFunction();jit.fn=object();jit.src='kernel';jit.cache_key='hash';jit.debug=False;jit.pre_run_hooks=[];jit.launch_metadata=None;jit.cache_hook=None;jit.used_global_vals={};jit.binder=object();jit.compile=object();jit.cache={0:{},1:{}}
        self.jit=jit;self.entry=e=LibEntry();e.fn=e.jit_function=jit;e._has_flagtune_tuner=False;e.divisibility=16;e.specialize_indices=list(range(8));e.do_not_specialize_indices=[8];e.kernel_cache=({}, {})
        e.key=lambda a,b,c:tuple((v.dtype,v.data_ptr()%16==0) if isinstance(v,Tensor) else v for v in a)+tuple(type(v) for v in b)+tuple(c)
        outer=self
        class Kernel:
            def __getitem__(self,grid):
                def run(*args):
                    outer.hot.append((args,outer.stream));args[0].updates+=1;args[1].updates+=1
                    if getattr(outer,'launch_failure',False):raise RuntimeError('after launch')
                return run
        self.record=(Kernel(),{},{},{},())
        def original(x,r,n,w,eps):
            self.cold.append((x,r,w,eps));x.updates+=1;r.updates+=1
            if not getattr(self,'missing',False):e.kernel_cache[x.device.index][self.ns['_entry_key'](e,self.ns['_args'](x,r,w,eps))]=self.record
            jit.cache.setdefault(x.device.index,{})['jit-key']=self.record[0]
            return x,r
        self.generic=types.SimpleNamespace(fused_add_rms_norm_kernel=e,fused_add_rms_norm=original)
        triton=types.SimpleNamespace(__version__='3.0.0',runtime=types.SimpleNamespace(JITFunction=JITFunction),knobs=types.SimpleNamespace(runtime=types.SimpleNamespace(jit_cache_hook=None,jit_post_compile_hook=None)))
        self.ns=dict(torch=types.SimpleNamespace(Tensor=Tensor,nn=types.SimpleNamespace(Parameter=Parameter),bfloat16='bf16',is_grad_enabled=lambda:self.grad_enabled),triton=triton,LibEntry=LibEntry,math=math,os=os,threading=threading,Counter=Counter,OrderedDict=OrderedDict,importlib=types.SimpleNamespace(import_module=lambda n:self.generic),torch_device_fn=types.SimpleNamespace(device=lambda d:nullcontext()))
        tree=ast.parse(SOURCE.read_text());tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))];exec(compile(tree,str(SOURCE),'exec'),self.ns);self.fn=self.ns['fused_add_rms_norm']
    def inputs(self,m=4,device=0):return Tensor((m,2048),device),Tensor((m,2048),device),Parameter((2048,),device)
    def call(self,t=None,eps=1e-5):
        x,r,w=t or self.inputs();return self.fn(x,r,[2048],w,eps)
    def test_once_return_current_epsilon_weight_stream_no_tensor_cache(self):
        t=self.inputs();out=self.call(t);self.assertIs(out[0],t[0]);self.assertIs(out[1],t[1]);self.assertEqual([t[i].updates for i in range(3)],[1,1,0])
        t2=self.inputs();self.stream=7;self.call(t2,.2)
        self.assertEqual(len(self.cold),1);self.assertEqual(len(self.hot),1);self.assertIs(self.hot[0][0][2],t2[2]);self.assertEqual(self.hot[0][0][-1],.2);self.assertEqual(self.hot[0][1],7)
        def tensors(v):return isinstance(v,Tensor) or (isinstance(v,(tuple,list,dict)) and any(tensors(x) for x in (v.values() if isinstance(v,dict) else v)))
        self.assertFalse(tensors(self.ns['_cache'](0)))
    def test_invalidation(self):
        for change in (lambda:self.entry.kernel_cache[0].clear(),lambda:self.jit.cache[0].clear(),lambda:setattr(self.jit,'cache',{}),lambda:setattr(self.jit,'binder',object()),lambda:setattr(self.jit,'compile',object()),lambda:self.record[1].update(num_warps=8),lambda:setattr(self.jit,'src','new'),lambda:setattr(self.jit,'fn',object()),lambda:setattr(self.entry,'divisibility',32)):
            self.setUp();self.call();change();self.call();self.assertEqual(len(self.cold),2);self.assertFalse(self.hot)
    def test_hooks_tuning_fallback(self):
        for change in (lambda:self.jit.pre_run_hooks.append(object()),lambda:setattr(self.jit,'debug',True),lambda:setattr(self.jit,'cache_hook',object()),lambda:setattr(self.jit,'launch_metadata',object()),lambda:setattr(self.entry,'_has_flagtune_tuner',True),lambda:setattr(self.ns['triton'].knobs.runtime,'jit_cache_hook',object()),lambda:setattr(self.entry,'run',lambda:None)):
            self.setUp();self.call();change();self.call();self.assertEqual(len(self.cold),2);self.assertFalse(self.hot)
    def test_layout_dtype_gradient_alias_size_fallback(self):
        for kind in ('empty','1025','stride','dtype','gradient','alias','weight-alias','wrong-shape','epsilon'):
            self.setUp();x,r,w=self.inputs()
            if kind=='empty':x,r,w=self.inputs(0)
            elif kind=='1025':x,r,w=self.inputs(1025)
            elif kind=='stride':x.contiguous=False
            elif kind=='dtype':w.dtype='fp32'
            elif kind=='gradient':w.requires_grad=True
            elif kind=='alias':r.ptr=x.ptr+2
            elif kind=='weight-alias':w.ptr=x.ptr
            elif kind=='wrong-shape':w.shape=(1024,)
            self.call((x,r,w),float('nan') if kind=='epsilon' else 1e-5);self.assertEqual(len(self.cold),1);self.assertFalse(self.ns['_cache'](0))
    def test_lru_threads_devices_pid(self):
        self.call();self.call(self.inputs(device=1));self.assertEqual(len(self.cold),2)
        thread=threading.Thread(target=self.call);thread.start();thread.join();self.assertEqual(len(self.cold),3)
        self.ns['_local'].pid=-1;self.call();self.assertEqual(len(self.cold),4)
        for m in range(1,66):self.call(self.inputs(m))
        self.assertEqual(len(self.ns['_cache'](0)),64)
    def test_launch_error_never_retries_original(self):
        self.call();self.launch_failure=True;t=self.inputs()
        with self.assertRaisesRegex(RuntimeError,'after launch'):self.call(t)
        self.assertEqual(len(self.cold),1);self.assertEqual(t[0].updates,1);self.assertEqual(t[1].updates,1)
    def test_real_parameter_flag_allowed_only_with_grad_disabled(self):
        t=self.inputs();t[2].requires_grad=True;self.grad_enabled=False
        self.call(t);self.call(t);self.assertEqual(len(self.cold),1);self.assertEqual(len(self.hot),1)
        self.grad_enabled=True;self.call(t);self.assertEqual(len(self.cold),2);self.assertEqual(len(self.hot),1)
    def test_missing_description_never_repeats_cold(self):
        self.missing=True;t=self.inputs();self.call(t);self.assertEqual(len(self.cold),1);self.assertEqual(t[0].updates,1);self.assertFalse(self.hot)
    def test_changed_global_invalidates(self):
        values={'K':1};self.jit.used_global_vals={('K',0):(1,values)};self.call();values['K']=2;self.call();self.assertEqual(len(self.cold),2);self.assertFalse(self.hot)

class MetadataParity(Lifecycle):
    def test_metadata_acceptance_errors_and_reads(self):
        import subprocess
        baseline=subprocess.check_output(['git','-C',str(SOURCE.parents[6]),'show','c99fe08f88d7504ab50c9f0524203f5bfc475a09:src/flag_gems/runtime/backend/_metax/fused/fused_add_rms_norm.py'],text=True)
        fn=next(n for n in ast.parse(baseline).body if isinstance(n,ast.FunctionDef) and n.name=='_metadata')
        ns=dict(self.ns);exec(compile(ast.Module(body=[fn],type_ignores=[]),'<baseline>','exec'),ns)
        for kind in ('valid','empty','1025','stride','offset','overlap','adjacent','weight-alias','parameter','grad','inference','bad-eps','bad-shape','none-shape'):
            with self.subTest(kind=kind):
                x,r,w=self.inputs();shape=[2048];eps=1e-5;self.grad_enabled=True
                if kind=='empty':x,r,w=self.inputs(0)
                if kind=='1025':x,r,w=self.inputs(1025)
                if kind=='stride':x.contiguous=False
                if kind=='offset':x.ptr+=2
                if kind=='overlap':r.ptr=x.ptr+2
                if kind=='adjacent':r.ptr=x.ptr+x.numel()*2
                if kind=='weight-alias':w.ptr=x.ptr
                if kind=='parameter':
                    class RealParameter(Parameter):pass
                    w=RealParameter((2048,))
                if kind in ('grad','inference'):w.requires_grad=True;self.grad_enabled=kind=='grad'
                if kind=='bad-eps':eps='1e-5'
                if kind=='bad-shape':shape=[1024]
                if kind=='none-shape':shape=None
                def result(fn):
                    try:return ('ok',fn(x,r,shape,w,eps))
                    except Exception as exc:return (type(exc).__name__,str(exc))
                self.assertEqual(result(ns['_metadata']),result(self.ns['_metadata']))
        from unittest.mock import patch
        t=self.inputs();reads=[];original=Tensor.data_ptr
        with patch.object(Tensor,'data_ptr',lambda obj:(reads.append(id(obj)),original(obj))[1]):
            self.ns['_metadata'](t[0],t[1],[2048],t[2],1e-5)
        self.assertEqual(Counter(reads),Counter({id(v):1 for v in t}))

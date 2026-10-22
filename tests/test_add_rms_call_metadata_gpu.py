"""Frozen A/B metadata-only validation. Runs only in the owned GPU worker."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import time
import unittest


class AddRMSMetadataGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch
        cls.protocol=Path(os.environ['C500_FROZEN_PROTOCOL'])
        cls.out=Path(os.environ['C500_NUMERICAL_CAPTURE_PATH']).parent
        baseline=cls.protocol.parent/'A/gems/src/flag_gems/runtime/backend/_metax/fused/fused_add_rms_norm.py'
        spec=importlib.util.spec_from_file_location('baseline_add_rms',baseline)
        cls.a=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.a)
        cls.b=importlib.import_module('_metax.fused.fused_add_rms_norm')
        cls.generic=importlib.import_module('flag_gems.fused.fused_add_rms_norm')
        cls.rows=[]

    def inputs(self,m,kind):
        t=self.torch;t.manual_seed(20261022)
        size=m*2048
        backing=t.randn(2*size+8192,device='cuda',dtype=t.bfloat16)
        offset=1 if kind=='offset' else 16
        x=backing[offset:offset+size].view(m,2048)
        r=backing[offset+size:offset+2*size].view(m,2048)
        w=t.nn.Parameter(t.randn(2048,device='cuda',dtype=t.bfloat16),requires_grad=False)
        if kind=='overlap':r=backing[offset+1:offset+size+1].view(m,2048)
        if kind=='weight-alias':w=x[0]
        if kind=='stride':
            x=t.randn(m,4096,device='cuda',dtype=t.bfloat16)[:,::2]
        if kind=='dtype':x=x.float();r=r.float();w=w.float()
        if kind=='parameter':
            from vllm.model_executor.parameter import ModelWeightParameter
            w=ModelWeightParameter(data=w.detach(),input_dim=0,output_dim=0,weight_loader=lambda *a:None)
        if kind=='gradient':x.requires_grad_()
        if kind=='weight-grad':w.requires_grad_()
        return x,r,w,backing

    def test_gpu_parity_reference_and_boundaries(self):
        t=self.torch
        cases=[(m,'valid') for m in (1,4,1024)]+[(4,k) for k in ('adjacent','offset','overlap','weight-alias','stride','dtype','parameter','gradient','weight-grad','dynamic-hook','alternate-stream')]+[(0,'empty'),(1025,'outside'),(4,'bad-eps'),(4,'bad-shape')]
        for m,kind in cases:
            with self.subTest(m=m,kind=kind):
                outputs={};errors={};metadata={};reference=None
                hook=(lambda *a,**kw:None) if kind=='dynamic-hook' else None
                if hook:self.generic.fused_add_rms_norm_kernel.jit_function.pre_run_hooks.append(hook)
                try:
                    for label,module in [('A',self.a),('B-cold',self.b),('B-hot',self.b)]:
                        x,r,w,backing=self.inputs(m,kind)
                        before=[v.detach().cpu().clone() for v in (x,r,w,backing)]
                        eps='bad' if kind=='bad-eps' else 1e-5
                        shape=[1024] if kind=='bad-shape' else [2048]
                        if label!='B-hot':module._cache(x.device.index).clear()
                        metadata[label]=module._metadata(x,r,shape,w,eps)
                        stream=t.cuda.Stream() if kind=='alternate-stream' else t.cuda.current_stream()
                        stream.wait_stream(t.cuda.current_stream())
                        try:
                            with t.cuda.stream(stream):ret=module.fused_add_rms_norm(x,r,shape,w,eps)
                            stream.synchronize()
                        except Exception as exc:
                            errors[label]=(type(exc).__name__,str(exc));continue
                        outputs[label]=[v.detach().cpu().clone() for v in (*ret,w,backing)]
                        if x.is_contiguous() and r.is_contiguous():self.assertTrue(ret[0] is x and ret[1] is r)
                        if metadata[label] is not None:
                            self.assertTrue(t.equal(before[2],outputs[label][2]))
                            # Entire allocation outside two legal tensor intervals is unchanged.
                            mask=t.ones(len(before[3]),dtype=t.bool);off=1 if kind=='offset' else 16
                            mask[off:off+2*m*2048]=False
                            self.assertTrue(t.equal(before[3][mask],outputs[label][3][mask]))
                            xx,rr,ww=[v.double() for v in before[:3]];summed=xx+rr
                            target=(summed/((summed*summed/2048).sum(-1,keepdim=True)+1e-5).sqrt()*ww).to(x.dtype).double()
                            refs=[target,summed.to(r.dtype).double()];violations=[]
                            for index,ref in enumerate(refs):
                                actual=outputs[label][index].double();bad=(actual-ref).abs()>1e-4+.016*ref.abs()
                                if bad.any():violations.append(dict(output=index,coordinates=bad.nonzero(),actual=actual[bad],reference=ref[bad],unrounded_sum=summed[bad]))
                            if violations:
                                artifact=self.out/f'fp64-failure-{m}-{kind}-{label}.pt'
                                t.save(dict(inputs=before,outputs=outputs[label],violations=violations),artifact)
                                self.fail('independent FP64 violations saved: '+str(artifact))
                    self.assertEqual(metadata['A'],metadata['B-cold']);self.assertEqual(metadata['A'],metadata['B-hot'])
                    if errors:
                        self.assertEqual(set(errors),{'A','B-cold','B-hot'});self.assertEqual(errors['A'],errors['B-cold']);self.assertEqual(errors['A'],errors['B-hot'])
                    else:
                        for label in ('B-cold','B-hot'):
                            self.assertTrue(all(t.equal(a,b) for a,b in zip(outputs['A'],outputs[label])),(m,kind,label))
                    self.rows.append(dict(M=m,kind=kind,exact=True,reference=metadata['A'] is not None,errors=errors))
                    (self.out/'add-rms-gpu-checks.json').write_text(json.dumps(self.rows,indent=2))
                finally:
                    if hook:self.generic.fused_add_rms_norm_kernel.jit_function.pre_run_hooks.remove(hook)

    def test_dynamic_invalidation_and_public_cost(self):
        t=self.torch;timings=[]
        with t.inference_mode():
            for m in (1,4,1024):
                for name,module in [('A',self.a),('B',self.b)]:
                    x,r,w,_=self.inputs(m,'valid');module._cache(0).clear();module.fused_add_rms_norm(x,r,[2048],w)
                    # Mutating a descriptor invalidates reuse; original entry recaptures safely.
                    entry=self.generic.fused_add_rms_norm_kernel
                    entry.kernel_cache[0].clear()
                    module.fused_add_rms_norm(x,r,[2048],w);t.cuda.synchronize()
                    for repeat in range(5):
                        start=time.perf_counter_ns()
                        for _ in range(20):module.fused_add_rms_norm(x,r,[2048],w)
                        submitted=time.perf_counter_ns();t.cuda.synchronize();end=time.perf_counter_ns()
                        timings.append(dict(M=m,arm=name,repeat=repeat,calls=20,submit_ns=submitted-start,complete_ns=end-start))
        (self.out/'public-call-advisory.json').write_text(json.dumps(timings,indent=2))

    def test_post_submission_exception_never_repeats_add(self):
        t=self.torch
        with t.inference_mode():
            x,r,w,_=self.inputs(4,'valid');self.b._cache(0).clear()
            self.b.fused_add_rms_norm(x,r,[2048],w);t.cuda.synchronize()
            expected=(x.double()+r.double()).to(r.dtype).cpu()
            cache=self.b._cache(0);key,record=next(iter(cache.items()));original=record[3]
            calls=[]
            def fail_after_launch(*args):
                calls.append(1);original(*args);raise RuntimeError('injected after submission')
            cache[key]=(*record[:3],fail_after_launch,*record[4:])
            try:
                with self.assertRaisesRegex(RuntimeError,'injected after submission'):self.b.fused_add_rms_norm(x,r,[2048],w)
                t.cuda.synchronize();self.assertEqual(len(calls),1);self.assertTrue(t.equal(r.cpu(),expected))
            finally:cache.clear()

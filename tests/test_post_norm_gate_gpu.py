# SPDX-License-Identifier: Apache-2.0
import importlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

class DeviceTests(unittest.TestCase):
    def test_exact_cold_hot_stream_and_inputs(self):
        import torch as t
        import flag_gems
        m=importlib.import_module('flag_gems.runtime.backend._metax.fused.post_norm_gate')
        gate=importlib.import_module('flag_gems.runtime.backend._metax.fused.gate_up_silu')
        norm=importlib.import_module('flag_gems.fused.fused_add_rms_norm')
        out=Path(os.environ['C500_NUMERICAL_CAPTURE_PATH']).parent
        with t.inference_mode():
            t.manual_seed(20261022)
            x=t.randn(4,2048,device='cuda',dtype=t.bfloat16)
            r=t.randn_like(x);nw=t.randn(2048,device='cuda',dtype=t.bfloat16)
            w=t.randn(12288,2048,device='cuda',dtype=t.bfloat16)*.02
            saved=[v.clone() for v in (x,r,nw,w)]
            xx,rr=x.clone(),r.clone()
            norm.fused_add_rms_norm(xx,rr,[2048],nw,1e-5)
            ref=gate.try_gate_up_silu(xx,w)
            self.assertIsNotNone(ref)
            meta=gate._baseline_record(x,w)
            (out/'selected-tile.json').write_text(json.dumps(meta,indent=2))
            for name in ('cold','hot','alternate-stream'):
                stream=t.cuda.Stream() if name=='alternate-stream' else t.cuda.current_stream()
                stream.wait_stream(t.cuda.current_stream())
                with t.cuda.stream(stream):result=m.try_post_norm_gate(x,r,nw,w,1e-5)
                stream.synchronize()
                self.assertIsNotNone(result)
                y,nr=result
                bad=y!=ref
                if bad.any():
                    t.save(dict(inputs=saved,norm=xx,reference=ref,actual=y,coordinates=bad.nonzero(),residual=nr),out/('exact-failure-'+name+'.pt'))
                self.assertTrue(t.equal(y,ref),'all exact violations retained')
                self.assertTrue(t.equal(nr,rr))
                self.assertNotEqual(nr.data_ptr(),r.data_ptr())
                self.assertTrue(all(t.equal(a,b) for a,b in zip(saved,(x,r,nw,w))))
            # Independent FP64 norm reference uses the unrounded residual sum.
            z=x.double()+r.double()
            expected=z/((z*z/2048).sum(-1,keepdim=True)+1e-5).sqrt()*nw.double()
            bad=(xx.double()-expected).abs()>1e-4+.016*expected.abs()
            if bad.any():t.save(dict(coordinates=bad.nonzero(),actual=xx[bad],reference=expected[bad]),out/'norm-reference-failure.pt')
            self.assertFalse(bool(bad.any()))
            # Dot rounding checked against FP64 using exactly the BF16 normalized input.
            dot=xx.double().cpu()@w.double().cpu().T
            paired=gate.paired_intermediate(xx,w).double().cpu()
            error=(paired-dot).abs()
            # One BF16 output rounding plus bounded FP32 accumulation, scaled by absolute dot sum.
            magnitude=xx.double().abs().cpu()@w.double().abs().cpu().T
            bound=dot.abs()/256+2e-6*magnitude+1e-6
            bad=error>bound
            if bad.any():t.save(dict(coordinates=bad.nonzero(),actual=paired[bad],reference=dot[bad],bound=bound[bad]),out/'projection-reference-failure.pt')
            self.assertFalse(bool(bad.any()))

    def test_rejected_domains_and_submission_exception(self):
        import torch as t
        m=importlib.import_module('flag_gems.runtime.backend._metax.fused.post_norm_gate')
        with t.inference_mode():
            x=t.zeros(4,2048,device='cuda',dtype=t.bfloat16);r=t.ones_like(x)
            nw=t.ones(2048,device='cuda',dtype=t.bfloat16);w=t.zeros(12288,2048,device='cuda',dtype=t.bfloat16)
            for args in ((x[:3],r[:3],nw,w,1e-5),(x,x,nw,w,1e-5),(x,r,x[0],w,1e-5),(x,r,nw,w,float('nan')),(x[:,::2],r,nw,w,1e-5)):
                self.assertIsNone(m.try_post_norm_gate(*args))
            class Broken:
                def __getitem__(self,grid):
                    def launch(*a,**kw):raise RuntimeError('submitted')
                    return launch
            with patch.object(m,'post_norm_gate_kernel',Broken()):
                with self.assertRaisesRegex(RuntimeError,'submitted'):m.try_post_norm_gate(x,r,nw,w,1e-5)
            self.assertTrue(bool((x==0).all()));self.assertTrue(bool((r==1).all()))

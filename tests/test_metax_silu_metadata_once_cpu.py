"""CPU getter counts complement inherited stream/cache/config lifecycle cases."""
from collections import Counter
import test_metax_silu_prefill_cpu as baseline
Tensor = baseline.Tensor


class MetadataOnce(baseline.SiluLifecycle):
    def test_hot_supported_inputs_read_metadata_once(self):
        x,y=self.inputs();self.fn(x,y)
        reads=Counter();original=Tensor.__getattribute__;identities={id(x),id(y)}
        tracked={'shape','device','dtype','stride','data_ptr','requires_grad','ndim'}
        def getter(t,name):
            if id(t) in identities and name in tracked:reads[(id(t),name)]+=1
            return original(t,name)
        Tensor.__getattribute__=getter
        try:self.fn(x,y)
        finally:Tensor.__getattribute__=original
        for t in (x,y):
            for name in tracked-{'ndim'}:self.assertEqual(reads[(id(t),name)],1,name)
            self.assertEqual(reads[(id(t),'ndim')],0)
        self.assertEqual(len(self.launches),1)
    def test_cold_descriptor_receives_admission_metadata(self):
        descriptor=self.ns['_descriptor'];seen=[]
        def spy(*args,**kwargs):
            seen.append(args[4]);return descriptor(*args,**kwargs)
        self.ns['_descriptor']=spy;x,y=self.inputs();self.fn(x,y)
        self.assertEqual(len(seen),1)
        self.assertEqual(seen[0][0],(4,6144));self.assertEqual(seen[0][3:5],((12288,1),(12288,1)))
    def test_changed_input_metadata_cannot_use_prior_signature(self):
        x,y=self.inputs();self.fn(x,y)
        x.shape=y.shape=(4,6143);x._stride=y._stride=(12286,1)
        self.fn(x,y)
        self.assertEqual(len(self.cold),2);self.assertEqual(len(self.launches),0)

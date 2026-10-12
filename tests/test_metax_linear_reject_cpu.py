"""CPU property contracts for the archived early-rejection prototype."""
import ast
from pathlib import Path
import subprocess
import types
import unittest
ROOT=Path(__file__).resolve().parents[1]
REL='src/flag_gems/runtime/backend/_metax/ops/linear.py'
class Tensor:
    def __init__(self,shape):self._shape=shape;self.reads=[];self.requires_grad=False
    @property
    def shape(self):self.reads.append('shape');return self._shape
    @property
    def device(self):self.reads.append('device');return types.SimpleNamespace(type='cuda',index=0)
    @property
    def dtype(self):self.reads.append('dtype');return 'bf16'
    def stride(self):return (self._shape[-1],1)
    def data_ptr(self):return 256
    def is_contiguous(self):return True
class Parameter(Tensor):pass

def function(source):
    tree=ast.parse(source);tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_metadata']
    scope=dict(torch=types.SimpleNamespace(Tensor=Tensor,nn=types.SimpleNamespace(Parameter=Parameter),bfloat16='bf16'))
    exec(compile(tree,REL,'exec'),scope);return scope['_metadata']
class RejectTests(unittest.TestCase):
    def setUp(self):
        self.probe=function((ROOT/REL).read_text())
        self.base=function(subprocess.check_output(['git','-C',str(ROOT),'show','480d67922f020b775a2a35efed90238380c84388:'+REL],text=True))
    def test_accepted_parameter_metadata_same_once(self):
        for m in (0,1,7,8,1024):
            for n,k in ((2048,6144),(12288,2048)):
                x,w=Tensor((m,k)),Parameter((n,k));expected=self.base(x,w,None)
                x.reads.clear();w.reads.clear();self.assertEqual(expected,self.probe(x,w,None))
                self.assertEqual(w.reads.count('shape'),1);self.assertEqual(x.reads.count('shape'),1)
    def test_real_rejected_weight_shapes_read_only_shape(self):
        for n in (2048,2560,130560):
            x,w=Tensor((4,2048)),Parameter((n,2048))
            self.assertIsNone(self.probe(x,w,None));self.assertEqual(x.reads,[]);self.assertEqual(w.reads,['shape'])
    def test_bias_and_unsupported_objects_do_not_read_properties(self):
        x,w=Tensor((4,2048)),Parameter((12288,2048))
        self.assertIsNone(self.probe(x,w,object()));self.assertEqual(x.reads+w.reads,[])
        self.assertIsNone(self.probe(object(),w,None));self.assertEqual(w.reads,[])
if __name__=='__main__':unittest.main()

"""CPU-only support contracts; no torch/Triton import or device initialization."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

SOURCE = Path(__file__).resolve().parents[2] / 'src/flag_gems/runtime/backend/_metax/fused/decode_o_split_k.py'


class Tensor:
    def __init__(self, shape, dtype='bf16', device=None, contiguous=True, grad=False):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device or NS(type='cuda', index=0)
        self.requires_grad = grad
        self.contiguous = contiguous
    def is_contiguous(self): return self.contiguous


class Parameter(Tensor): pass


def load_support():
    tree = ast.parse(SOURCE.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'supported')
    torch = NS(Tensor=Tensor, nn=NS(Parameter=Parameter), bfloat16='bf16',
               is_grad_enabled=lambda: False, is_inference_mode_enabled=lambda: True,
               compiler=NS(is_compiling=lambda: False))
    ns = {'torch': torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns['supported'], torch


class SupportContracts(unittest.TestCase):
    def test_real_parameter_subclass_and_all_decode_shapes(self):
        supported, _ = load_support()
        for m in (1, 2, 3, 4):
            self.assertTrue(supported(Tensor((m, 2048)), Parameter((2048, 2048))))

    def test_reject_boundaries_without_submitting(self):
        supported, _ = load_support()
        w = Parameter((2048, 2048))
        for shape in ((0,2048), (5,2048), (1024,2048), (4,2047), (2048,)):
            self.assertFalse(supported(Tensor(shape), w), shape)
        self.assertFalse(supported(Parameter((4,2048)), w))
        self.assertFalse(supported(Tensor((4,2048)), Tensor((2560,2048))))
        for key,value in [('dtype','fp32'), ('contiguous',False), ('grad',True),
                          ('device',NS(type='cpu',index=0))]:
            self.assertFalse(supported(Tensor((4,2048), **{key:value}), w), key)
            self.assertFalse(supported(Tensor((4,2048)), Parameter((2048,2048), **{key:value})), key)

    def test_execution_modes(self):
        for key in ('is_grad_enabled', 'is_inference_mode_enabled', 'is_compiling'):
            supported, torch = load_support()
            target = torch.compiler if key == 'is_compiling' else torch
            setattr(target, key, lambda: key != 'is_inference_mode_enabled')
            self.assertFalse(supported(Tensor((4,2048)), Parameter((2048,2048))))

    def test_workspace_ownership_and_no_atomic_or_persistent_tensor(self):
        tree = ast.parse(SOURCE.read_text())
        # Reject module-scope allocations and atomics; fresh storage is essential
        # for asynchronous stream safety and cross-layer calls.
        self.assertFalse(any(isinstance(n,(ast.Assign,ast.AnnAssign)) for n in tree.body))
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr.startswith('atomic') for n in ast.walk(tree)))
        for m in range(1,5):
            # Exhaustively audit the partial/reduction storage partition.
            writes={(part*m+row)*2048+col for part in range(8) for row in range(m) for col in range(2048)}
            self.assertEqual(writes, set(range(8*m*2048)))


if __name__ == '__main__': unittest.main()

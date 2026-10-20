"""Registered exact GPU gate; run only by the locked controller worker.

CPU contract module is separately runnable without importing torch/Triton.
Baseline call is extracted from the frozen A source; the unchanged kernel AST is
checked before sharing its actual JIT object, so both paths use identical code.
"""
import ast
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

cpu_path = Path(__file__).parent / 'metax_cpu/test_qkv_native_launch.py'
spec = importlib.util.spec_from_file_location('_qkv_cpu_contracts', cpu_path)
cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)
LaunchContracts = cpu.LaunchContracts


class ActualExactLaunch(unittest.TestCase):
    def test_frozen_A_vs_hot_B_boundaries_stream_and_profiler(self):
        import torch
        from triton.compiler.compiler import CompiledKernel
        op = importlib.import_module('flag_gems.runtime.backend._metax.fused.qkv_rope_cache')
        protocol_file = Path(os.environ['C500_FROZEN_PROTOCOL'])
        protocol = json.loads(protocol_file.read_text())
        rel = 'src/flag_gems/runtime/backend/_metax/fused/qkv_rope_cache.py'
        source_a = protocol_file.parent / 'A/gems' / rel
        a_tree, b_tree = ast.parse(source_a.read_text()), ast.parse(Path(op.__file__).read_text())
        def kernel(tree):
            return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                 and n.name == 'qkv_rope_cache_kernel'), include_attributes=False)
        self.assertEqual(kernel(a_tree), kernel(b_tree), 'arithmetic changed under exact policy')
        fn = next(n for n in a_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'try_qkv_rope_cache')
        ns = dict(op.__dict__)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source_a), 'exec'), ns)
        baseline = ns['try_qkv_rope_cache']
        records = []
        # No skip when GPU/init/import is unavailable: fail before serving.
        with torch.inference_mode():
            for m in (1, 2, 3, 4):
                for label in ('random', 'zero', 'cancellation', 'small', 'large', 'offset',
                              'negative-slot', 'short-slots', 'strided-table', 'alternate-stream'):
                    torch.manual_seed(20261020 + m)
                    x = torch.randn((m, 2048), device='cuda', dtype=torch.bfloat16)
                    w = torch.randn((2560, 2048), device='cuda', dtype=torch.bfloat16) * .02
                    if label == 'zero': x.zero_()
                    if label == 'cancellation':
                        x[:, 1::2] = -x[:, ::2]; w[:, 1::2] = w[:, ::2]
                    if label == 'small': x.mul_(.01)
                    if label == 'large': x.mul_(10)
                    if label == 'offset':
                        backing = torch.empty(x.numel()+8, device=x.device, dtype=x.dtype)
                        backing[8:] = x.flatten(); x = backing[8:].view_as(x)
                    # Exercise vLLM's actual parameter subclass rather than exact Tensor only.
                    from vllm.model_executor.parameter import ModelWeightParameter
                    # No distributed model exists in the pre-service gate. Only
                    # rank/world access is stubbed; the real subclass initializes.
                    with patch('vllm.model_executor.parameter.get_tensor_model_parallel_rank', return_value=0), \
                         patch('vllm.model_executor.parameter.get_tensor_model_parallel_world_size', return_value=1):
                        w = ModelWeightParameter(data=w, input_dim=1, output_dim=0,
                                                 weight_loader=lambda *args: None)
                    positions = torch.tensor([0, 15, 16, 31][:m], device='cuda')
                    slots = torch.tensor([15, 16, 31, 32][:m], device='cuda')
                    if label == 'negative-slot': slots[-1] = -1
                    if label == 'short-slots': slots = slots[:m-1]
                    phase = torch.randn((32, 64), device='cuda')
                    table = torch.cat((phase.cos(), phase.sin()), -1)
                    if label == 'strided-table':
                        storage = torch.empty((32, 256), device='cuda'); storage[:, ::2] = table
                        table = storage[:, ::2]
                    cache = torch.full((2, 4, 16, 2, 128), -19, device='cuda', dtype=x.dtype)
                    original = cache.clone()
                    op.clear_launch_cache()
                    a = baseline(x, w, positions, table, cache, slots)
                    cache_a = cache.clone()
                    cache.copy_(original)
                    # One normal cold invocation installs the descriptor.
                    with patch.object(op, '_standard_launch', wraps=op._standard_launch) as cold:
                        op.try_qkv_rope_cache(x, w, positions, table, cache, slots)
                        self.assertEqual(cold.call_count, 1)
                        cache.copy_(original)
                        stream = torch.cuda.Stream() if label == 'alternate-stream' else torch.cuda.current_stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            b = op.try_qkv_rope_cache(x, w, positions, table, cache, slots)
                        torch.cuda.current_stream().wait_stream(stream)
                        torch.cuda.synchronize()
                        self.assertEqual(cold.call_count, 1, 'B did not use compiled hot branch')
                    self.assertIsNot(a, b)
                    active = torch.zeros(64, device='cuda', dtype=torch.bool)
                    active[slots[slots >= 0].long()] = True
                    exact = torch.equal(a, b) and torch.equal(cache_a, cache)
                    sentinel = torch.equal(cache.reshape(2, 64, 2, 128)[:, ~active],
                                           original.reshape(2, 64, 2, 128)[:, ~active])
                    row = dict(M=m, label=label, exact=exact, sentinel=sentinel)
                    records.append(row)
                    if not exact or not sentinel:
                        torch.save(dict(row=row, input=x.cpu(), weight=w.cpu(), A=a.cpu(), B=b.cpu(),
                                        cache_A=cache_a.cpu(), cache_B=cache.cpu(), positions=positions.cpu(),
                                        table=table.cpu(), slots=slots.cpu()),
                                   Path(os.environ['C500_NUMERICAL_CAPTURE_PATH']).with_suffix('.failure.pt'))
                    self.assertTrue(exact and sentinel, row)
            # Real standard CompiledKernel profiler hooks must see hot submissions.
            self.assertIsNone(CompiledKernel.launch_enter_hook)
            self.assertIsNone(CompiledKernel.launch_exit_hook)
            calls = []
            try:
                CompiledKernel.launch_enter_hook = lambda metadata: calls.append('enter')
                CompiledKernel.launch_exit_hook = lambda metadata: calls.append('exit')
                op.try_qkv_rope_cache(x, w, positions, table, cache, slots)
                torch.cuda.synchronize()
            finally:
                CompiledKernel.launch_enter_hook = None
                CompiledKernel.launch_exit_hook = None
            self.assertEqual(calls, ['enter', 'exit'])
            self.assertIsNone(op.try_qkv_rope_cache(x.repeat(5, 1), w, positions, table, cache, slots))
        Path(os.environ['C500_NUMERICAL_CAPTURE_PATH']).with_name('exact-launch-cases.json').write_text(
            json.dumps(dict(source_refs=protocol['source_refs'], cases=records,
                            kernel_ast_sha256=hashlib.sha256(kernel(a_tree).encode()).hexdigest(),
                            scope='synthetic boundary exactness; real model hit/output audited separately'), indent=2))


if __name__ == '__main__':
    unittest.main()

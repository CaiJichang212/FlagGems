"""One allocation, unchanged QKV kernel. GPU test runs only under frozen worker."""
import ast
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

GEMS = Path(__file__).resolve().parents[2]
PLUGIN = GEMS.parent / ('plugin' if (GEMS.parent / 'plugin').exists() else 'vllm-plugin-FL')
SOURCE = GEMS / 'src/flag_gems/runtime/backend/_metax/fused/qkv_rope_cache.py'
PLUGIN_SOURCE = PLUGIN / 'vllm_fl/models/minicpm5_rope_kv.py'
BASE_KERNEL_AST = '8fbc11c9b22eecd14a39b98ef3685f59a72b63509698f140092eac1a4f603290'


def allocator(torch):
    tree = ast.parse(PLUGIN_SOURCE.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_paired_qkv_attention_outputs')
    ns = {'torch': torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(PLUGIN_SOURCE), 'exec'), ns)
    return ns[fn.name]


class GeometryTests(unittest.TestCase):
    def test_kernel_ast_unchanged(self):
        tree = ast.parse(SOURCE.read_text())
        kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'qkv_rope_cache_kernel')
        self.assertEqual(hashlib.sha256(ast.dump(kernel).encode()).hexdigest(), BASE_KERNEL_AST)

    def test_one_allocation_disjoint_views_and_retained_results(self):
        import torch
        allocate = allocator(torch)
        for m in (1, 2, 3, 4):
            x = torch.zeros((m, 2048), dtype=torch.bfloat16)
            with patch.object(torch, 'empty', wraps=torch.empty) as empty:
                q, a = allocate(x)
                self.assertEqual(empty.call_count, 1)
            self.assertEqual(q.untyped_storage().data_ptr(), a.untyped_storage().data_ptr())
            self.assertEqual(a.data_ptr() - q.data_ptr(), m * 2560 * 2)
            self.assertEqual(tuple(q.shape), (m, 2560))
            self.assertEqual(tuple(a.shape), (m, 16, 128))
            self.assertTrue(q.is_contiguous() and a.is_contiguous())
            q.fill_(3); a.fill_(7)
            q2, a2 = allocate(x); q2.zero_(); a2.zero_()
            self.assertTrue(bool((q == 3).all()) and bool((a == 7).all()))
            self.assertNotEqual(q.data_ptr(), q2.data_ptr())
            # Losing one view must not invalidate the other view's owning storage.
            del q
            self.assertTrue(bool((a == 7).all()))


class RealWeightGPU(unittest.TestCase):
    def test_actual_weight_subclass_all_shapes_reference_and_boundaries(self):
        import importlib
        import torch
        import flag_gems
        from safetensors import safe_open
        import vllm.model_executor.parameter as parameter
        from c500_qkv_rope_worker import references
        p = json.loads(Path(os.environ['C500_FROZEN_PROTOCOL']).read_text())
        model = Path(p['model'])
        artifact = Path(os.environ['C500_NUMERICAL_CAPTURE_PATH']).parent
        op = importlib.import_module('flag_gems.runtime.backend._metax.fused.qkv_rope_cache')
        self.assertTrue(Path(op.__file__).resolve().is_relative_to(GEMS))
        allocate = allocator(torch)
        index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
        rows = []
        # Construct the real vLLM Parameter class from original model weights.
        # Only TP metadata queries are fixtures (rank0/world1); served-model hit
        # validation separately checks actual vLLM-loaded objects and dispatch.
        with torch.inference_mode():
            torch.manual_seed(20261022)
            for layer in range(42):
                tensors = []
                for proj in ('q', 'k', 'v'):
                    key = f'model.layers.{layer}.self_attn.{proj}_proj.weight'
                    with safe_open(str(model / index[key]), framework='pt', device='cpu') as f:
                        tensors.append(f.get_tensor(key))
                data = torch.cat(tensors).to(device='cuda', dtype=torch.bfloat16)
                with patch.object(parameter, 'get_tensor_model_parallel_rank', return_value=0), patch.object(parameter, 'get_tensor_model_parallel_world_size', return_value=1):
                    weight = parameter.ModelWeightParameter(data=data, input_dim=1, output_dim=0, weight_loader=lambda *a: None)
                self.assertEqual(tuple(weight.shape), (2560, 2048))
                for m in (1, 2, 3, 4):
                    x = torch.randn((m, 2048), device='cuda', dtype=torch.bfloat16)
                    original_x = x.clone()
                    positions = torch.tensor([0, 15, 16, 31][:m], device='cuda')
                    slots = torch.tensor([15, 16, 31, 32][:m], device='cuda')
                    angles = torch.arange(32 * 64, device='cuda', dtype=torch.float32).reshape(32, 64) / 137
                    table = torch.cat((angles.cos(), angles.sin()), dim=-1)
                    cache = torch.full((2, 4, 16, 2, 128), -19., device='cuda', dtype=torch.bfloat16)
                    initial = cache.clone()
                    with patch.object(torch, 'empty', wraps=torch.empty) as empty:
                        separate_attention = torch.empty((m, 16, 128), device=x.device, dtype=x.dtype)
                        a = op.try_qkv_rope_cache(x, weight, positions, table, cache, slots)
                        self.assertEqual(empty.call_count, 2)
                    ca = cache.clone(); cache.copy_(initial)
                    with patch.object(torch, 'empty', wraps=torch.empty) as empty:
                        dst, attention = allocate(x); attention.fill_(-17.)
                        b = op.try_qkv_rope_cache(x, weight, positions, table, cache, slots, out=dst)
                        self.assertEqual(empty.call_count, 1)
                    self.assertIs(b, dst)
                    exact = torch.equal(a, b) and torch.equal(ca, cache)
                    sentinel = bool((attention == -17).all()) and torch.equal(x, original_x)
                    # Existing independent FP64 all-coordinate projection/rotation
                    # oracle, with raw (unrotated) projection as its input.
                    pre = flag_gems.linear(x, weight)
                    ref, detail = references(x, weight, pre, positions, table, b[:, :2048].view(m, 16, 128), cache, slots)
                    row = dict(layer=layer, M=m, exact=bool(exact), sentinel=bool(sentinel), reference=ref,
                               weight_type=type(weight).__module__ + '.' + type(weight).__name__, allocations_A=2, allocations_B=1)
                    if layer == 0:
                        # Exercise the actual FA2 destination ABI with the shared
                        # allocation, including its nonzero storage offset.
                        from flash_attn import flash_attn_interface as fa
                        decode = importlib.import_module('vllm_fl.dispatch.backends.vendor.metax.impl.attention.ops.decode_output')
                        q = b[:, :2048].view(m, 1, 16, 128)
                        kc, vc = cache.unbind(0)
                        bt = torch.arange(4, device='cuda', dtype=torch.int32).repeat(m, 1)
                        lens = torch.tensor([16, 17, 32, 33][:m], device='cuda', dtype=torch.int32)
                        scale = 128 ** -.5
                        expected_attention = fa.flash_attn_with_kvcache(q, kc, vc, cache_seqlens=lens,
                            block_table=bt, softmax_scale=scale, causal=True).view(m, 16, 128)
                        self.assertTrue(decode.try_decode_into(fa.flash_attn_with_kvcache, q, kc, vc,
                            attention, bt, lens, scale, (-1, -1), None, 0., None))
                        self.assertTrue(torch.equal(attention, expected_attention))
                        self.assertTrue(torch.equal(b, a))
                        qc = q[:, 0].cpu().double()
                        keys = kc.reshape(-1, 2, 128).cpu().double().repeat_interleave(8, dim=1)
                        values = vc.reshape(-1, 2, 128).cpu().double().repeat_interleave(8, dim=1)
                        independent = []
                        for i, length in enumerate(lens.cpu().tolist()):
                            logits = torch.einsum('hd,lhd->hl', qc[i], keys[:length]) * scale
                            independent.append(torch.einsum('hl,lhd->hd', logits.softmax(-1), values[:length]))
                        independent = torch.stack(independent)
                        difference = (attention.cpu().double() - independent).abs()
                        violations = (difference > (.02 + .02 * independent.abs())).nonzero().tolist()
                        row['attention_destination'] = dict(exact=True, fp64_passed=not violations,
                            violations=violations, max_abs=float(difference.max()), storage_offset=attention.storage_offset())
                        if violations:
                            torch.save(dict(actual=attention.cpu(), reference=independent, violations=violations), artifact/'attention-allocation-failure.pt')
                            self.fail('independent attention output reference failed')
                    rows.append(row)
                    if not exact or not sentinel or not ref['passed']:
                        torch.save(dict(row=row, x=x.cpu(), weight=weight.cpu(), A=a.cpu(), B=b.cpu(), cache=cache.cpu(), **detail), artifact/'allocation-failure.pt')
                        (artifact/'allocation-failure.json').write_text(json.dumps(row, indent=2))
                        self.fail('actual-weight exact/reference/sentinel gate failed')
                del weight, data, tensors
            # Last layer weights/data remain referenced by the last actual weight
            # argument only if deliberately retained; build a fresh Parameter for
            # boundary cases without saving any buffer inside the implementation.
            weight = torch.nn.Parameter(torch.randn((2560, 2048), device='cuda', dtype=torch.bfloat16) * .02, requires_grad=False)
            m = 4; x = torch.randn((m, 2048), device='cuda', dtype=torch.bfloat16)
            positions = torch.tensor([0, 15, 16, 31], device='cuda')
            slots = torch.tensor([15, 16, 31, 32], device='cuda')
            cache.fill_(-19); original = cache.clone()
            bads = [torch.empty((m, 2561), device='cuda', dtype=x.dtype),
                    torch.empty((m, 2560), device='cuda', dtype=torch.float32),
                    torch.empty((2560, m), device='cuda', dtype=x.dtype).T,
                    torch.empty(m*2560+1, device='cuda', dtype=x.dtype)[1:].view(m, 2560),
                    weight.view(-1)[:m*2560].view(m, 2560),
                    cache.view(-1)[:m*2560].view(m, 2560)]
            for bad in bads:
                self.assertIsNone(op.try_qkv_rope_cache(x, weight, positions, table, cache, slots, out=bad))
                self.assertTrue(torch.equal(cache, original))
            # Misaligned/unsupported input still falls back before any launch.
            self.assertIsNone(op.try_qkv_rope_cache(x[:1, :1024], weight, positions[:1], table, cache, slots[:1]))
            boundaries = []
            for label, use_slots in [('short-slots', slots[:3]), ('negative-slot', torch.tensor([15,16,31,-1], device='cuda'))]:
                cache.copy_(original)
                a = op.try_qkv_rope_cache(x, weight, positions, table, cache, use_slots); ca = cache.clone()
                cache.copy_(original)
                dst, attention = allocate(x); attention.fill_(11)
                b = op.try_qkv_rope_cache(x, weight, positions, table, cache, use_slots, out=dst)
                self.assertTrue(torch.equal(a, b) and torch.equal(ca, cache) and bool((attention == 11).all()))
                boundaries.append(label)
            backing = torch.full((m*2560+16,), 23., device='cuda', dtype=x.dtype)
            offset = backing[8:-8].view(m, 2560)
            cache.copy_(original)
            a = op.try_qkv_rope_cache(x, weight, positions, table, cache, slots); ca=cache.clone()
            cache.copy_(original)
            stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                b = op.try_qkv_rope_cache(x, weight, positions, table, cache, slots, out=offset)
            torch.cuda.current_stream().wait_stream(stream)
            self.assertTrue(torch.equal(a,b) and torch.equal(ca,cache))
            self.assertTrue(bool((backing[:8]==23).all()) and bool((backing[-8:]==23).all()))
            retained = b.clone(); fresh, other = allocate(x)
            op.try_qkv_rope_cache(x, weight, positions, table, cache, slots, out=fresh)
            self.assertTrue(torch.equal(retained, b))
            calls=[]
            class RaisingKernel:
                def __getitem__(self, grid):
                    def launch(*args, **kwargs):
                        calls.append(1); raise RuntimeError('injected submission error')
                    return launch
            with patch.object(op, 'qkv_rope_cache_kernel', RaisingKernel()):
                with self.assertRaisesRegex(RuntimeError, 'injected submission error'):
                    op.try_qkv_rope_cache(x, weight, positions, table, cache, slots, out=fresh)
            self.assertEqual(len(calls), 1)
            torch.cuda.synchronize()
        (artifact/'allocation-reference.json').write_text(json.dumps(dict(passed=True, cases=rows,
            boundaries=boundaries+['invalid-destinations','offset-sentinels','alternate-stream','retained-output','single-submit-error'],
            scope='original model weights in real ModelWeightParameter with TP metadata fixtures; synthetic inputs; actual loaded-model objects and branch checked separately'), indent=2))

"""Execute the actual capture callback with CPU doubles, without vendor imports."""
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'test_decode_o_split_k.py'


def factory(name="make_capture_observer"):
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == name)
    scope = {'json': json}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope[node.name]


class Value:
    def __init__(self, values): self.values = values
    def float(self): return self
    def cpu(self): return self
    def tolist(self): return self.values


class CaptureContracts(unittest.TestCase):
    def test_staggered_prefill_budget_reproduces_old_gap_and_retains_four_decoders(self):
        budget = factory('capture_output_budget')
        engine = dict(max_num_seqs=4, max_num_batched_tokens=1024, max_model_len=16384)
        prompts = [dict(prompt_token_ids=[0]*1024) for _ in range(4)]
        tokens = budget(prompts, engine)
        # CPU schedule double: decode first, then fill residual budget with
        # pending prefills. An entered request's first token is its prefill.
        def seen(limit):
            remaining = [1024]*4; generated = [0]*4; widths = []
            while any(g < limit for g in generated):
                active = [i for i in range(4) if remaining[i] == 0 and generated[i] < limit]
                widths.append(len(active))
                for i in active: generated[i] += 1
                left = 1024 - len(active)
                for i in range(4):
                    if remaining[i] and left:
                        used = min(left, remaining[i]); remaining[i] -= used; left -= used
                        if remaining[i] == 0: generated[i] += 1
            return widths
        self.assertNotIn(4, seen(4))
        self.assertIn(4, seen(tokens))
        self.assertEqual(tokens, 10)
        self.assertEqual(budget(prompts[:1], engine), 8)
        longer = [dict(prompt_token_ids=[0]*4096) for _ in range(4)]
        self.assertGreater(budget(longer, engine), tokens)

    def test_capture_budget_rejects_capacity_and_context_instead_of_changing_engine(self):
        budget = factory('capture_output_budget')
        prompts = [dict(prompt_token_ids=[0]*1024) for _ in range(4)]
        engine = dict(max_num_seqs=4, max_num_batched_tokens=1024, max_model_len=1028)
        with self.assertRaisesRegex(ValueError, 'context'): budget(prompts, engine)
        with self.assertRaisesRegex(ValueError, 'batch'): budget(prompts*2, engine)
        engine.update(max_num_batched_tokens=3)
        with self.assertRaisesRegex(ValueError, 'prefill capacity'): budget(prompts, engine)

    def setup_callback(self, output, *, launch=True, error=False):
        a, b = Value([[1.0]]), Value([[1.0078125]])
        x = NS(shape=(1, 2048))
        weight = NS(T=object())
        calls = []; launches = []; captures = {}; weights = {}; syncs = []
        def project(value):
            calls.append(('A', value))
            return a, None
        attention = NS(o_proj=NS(weight=weight))
        # Callable module double deliberately has no unittest assertion methods.
        class Projection:
            def __init__(self): self.weight = weight
            def __call__(self, value): return project(value)
        attention.o_proj = Projection()
        def original(receiver, value, phase, layer):
            self.assertIs(receiver, attention)
            calls.append(('B', value))
            if error: raise RuntimeError('submitted failure')
            if launch: launches.append((value.shape[0], layer))
            return b
        case = dict(id='case', origin='actual frozen input', phase='decode', layer=0)
        refs = dict(A='a', B='b', gems_A='ga', gems_B='gb')
        def binary(value, path):
            calls.append(('binary', value))
            return dict(path=str(path), marker=len(calls))
        observe = factory()(self, original, launches, {(1,0):case}, captures, weights,
            output, refs, binary, lambda: syncs.append(True))
        return NS(observe=observe, attention=attention, x=x, a=a, b=b, calls=calls,
                  captures=captures, weights=weights, refs=refs, syncs=syncs)

    def test_real_receiver_separate_from_assertions_and_actual_values_published(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)/'captures.json'
            f = self.setup_callback(output)
            self.assertIs(f.observe(f.attention, f.x, 'decode', 0), f.b)
            row = json.loads(output.read_text())[0]
            self.assertEqual(row['A'], f.a.values)
            self.assertEqual(row['B'], f.b.values)
            self.assertNotEqual(row['A'], row['B'])
            self.assertEqual(row['source_refs'], f.refs)
            self.assertEqual((row['phase'],row['layer'],row['case_id']), ('decode',0,'case'))
            self.assertEqual(f.calls[:2], [('B',f.x), ('A',f.x)])
            self.assertIs(f.calls[2][1], f.attention.o_proj.weight.T)
            self.assertIs(f.calls[3][1], f.x)
            self.assertEqual(f.syncs, [True])
            saved = output.read_bytes()
            f.observe(f.attention, f.x, 'decode', 0)
            self.assertEqual(output.read_bytes(), saved)
            self.assertEqual(len(f.captures), 1)
            self.assertEqual(len([v for k,v in f.calls if k=='A']), 1)

    def test_silent_fallback_fails_before_baseline_or_capture(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)/'captures.json'
            f = self.setup_callback(output, launch=False)
            with self.assertRaisesRegex(AssertionError, 'silently fell back'):
                f.observe(f.attention, f.x, 'decode', 0)
            self.assertFalse(output.exists())
            self.assertEqual(f.captures, {})
            self.assertEqual(f.calls, [('B',f.x)])

    def test_submitted_failure_preserved_without_overwriting_existing_capture(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)/'captures.json'; output.write_text('old evidence')
            f = self.setup_callback(output, error=True)
            with self.assertRaisesRegex(RuntimeError, 'submitted failure'):
                f.observe(f.attention, f.x, 'decode', 0)
            self.assertEqual(output.read_text(), 'old evidence')
            self.assertEqual(f.calls, [('B',f.x)])
            self.assertEqual(f.captures, {})


if __name__ == '__main__': unittest.main()

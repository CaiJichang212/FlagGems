"""Execute the actual capture callback with CPU doubles, without vendor imports."""
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'test_decode_o_split_k.py'


def factory():
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'make_capture_observer')
    scope = {'json': json}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope[node.name]


class Value:
    def __init__(self, values): self.values = values
    def float(self): return self
    def cpu(self): return self
    def tolist(self): return self.values


class CaptureContracts(unittest.TestCase):
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

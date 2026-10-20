"""CPU contracts execute candidate functions with fake launchers, no GPU imports."""
import ast
from collections import OrderedDict, defaultdict
from contextlib import nullcontext
import os
from pathlib import Path
import threading
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).parents[2]
SOURCE = ROOT / 'src/flag_gems/runtime/backend/_metax/fused/qkv_rope_cache.py'
STAMP = ROOT / 'src/flag_gems/runtime/backend/_metax/fused/gate_up_silu.py'


class Tensor:
    def __init__(self, shape, dtype='bf16', ptr=256, device=0, strides=None):
        self.shape, self.dtype, self.ptr = shape, dtype, ptr
        self.device = NS(index=device)
        self.strides = strides or tuple(__import__('math').prod(shape[i+1:]) for i in range(len(shape)))

    def stride(self, i=None):
        return self.strides if i is None else self.strides[i]

    def data_ptr(self):
        return self.ptr

    def numel(self):
        return __import__('math').prod(self.shape)


class LaunchContracts(unittest.TestCase):
    def setUp(self):
        self.cold, self.hot = [], []
        self.stream = 1
        self.suppressed = self.fail_hot = False
        self.hooks = []
        self.knobs = NS(runtime=NS(jit_post_compile_hook=None))
        self.jitclass = NS(cache_hook=None)
        outer = self

        class Kernel:
            def __getitem__(self, grid):
                def run(*args):
                    outer.hot.append((args, grid, outer.stream, list(outer.hooks)))
                    if outer.fail_hot:
                        raise RuntimeError('submitted hot failure')
                return run

        class Jit:
            def __init__(self):
                self.fn, self.src, self.debug = object(), 'source', None
                self.noinline, self.cache_key = None, 'hash'
                self.cache = defaultdict(dict)
                self.pre_run_hooks, self.used_global_vals = [], {}
                self.binder = self.bind

            def bind(self, *args, **kwargs):
                return None, ['key'], args[7:], args[:7], kwargs

            def run(self):
                pass

            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    outer.cold.append((args, grid))
                    for hook in self.pre_run_hooks:
                        hook()
                    if outer.suppressed:
                        return None
                    kw = dict(kwargs, debug=self.debug)
                    key = 'key' + str((args[7:], kw))
                    return self.cache[args[0].device.index].setdefault(key, Kernel())
                return run

        self.jit = Jit()
        self.ns = dict(os=os, threading=threading, OrderedDict=OrderedDict,
                       qkv_rope_cache_kernel=self.jit, _launch_local=threading.local(),
                       JITFunction=self.jitclass, knobs=self.knobs,
                       triton=NS(__version__='3.0.0'),
                       torch=NS(empty=lambda shape, dtype, device: Tensor(shape, dtype, device=device.index)),
                       torch_device_fn=NS(device=lambda _: nullcontext()), supported=lambda *args: True)
        for path, exclude in [(STAMP, None), (SOURCE, {'qkv_rope_cache_kernel', 'supported'})]:
            tree = ast.parse(path.read_text())
            tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                         and (n.name == '_jit_stamp' if exclude is None else n.name not in exclude)]
            exec(compile(tree, str(path), 'exec'), self.ns)
        self.call = self.ns['try_qkv_rope_cache']

    def inputs(self, rows=4, device=0):
        return [Tensor(s, dtype=d, device=device) for s, d in [
            ((rows, 2048), 'bf16'), ((2560, 2048), 'bf16'), ((rows,), 'int64'),
            ((128, 128), 'fp32'), ((2, 4, 16, 2, 128), 'bf16'), ((rows,), 'int64')]]

    def test_one_cold_then_dynamic_pointers_output_stream_and_hooks(self):
        first = self.call(*self.inputs())
        args = self.inputs()
        args[0].ptr += 4096
        args[1].ptr += 8192
        self.stream = 7
        self.hooks.append('profiler-enter-and-exit')
        second = self.call(*args)
        self.assertEqual(len(self.cold), 1)
        self.assertEqual(len(self.hot), 1)
        self.assertIsNot(first, second)
        vals, grid, stream, hooks = self.hot[0]
        self.assertEqual(grid, (80, 1, 1))
        self.assertEqual(stream, 7)
        self.assertTrue(hooks)
        for a, b in zip(vals, [args[0], args[1], second, *args[2:]]):
            self.assertIs(a, b)
        record = next(iter(self.ns['_launch_cache'](0).values()))
        self.assertFalse(any(isinstance(v, Tensor) for v in record))

    def test_invalidation_and_hooks(self):
        mutations = [lambda: self.jit.cache[0].clear(),
                     lambda: setattr(self.jit, 'cache', defaultdict(dict)),
                     lambda: setattr(self.jit, 'src', 'new'),
                     lambda: setattr(self.jit, 'cache_key', 'new'),
                     lambda: setattr(self.jit, 'debug', True),
                     lambda: setattr(self.jit, 'launch_metadata', lambda *a: {}),
                     lambda: self.jit.pre_run_hooks.append(lambda: None),
                     lambda: setattr(self.jitclass, 'cache_hook', object()),
                     lambda: setattr(self.knobs.runtime, 'jit_post_compile_hook', object()),
                     lambda: setattr(self.ns['triton'], '__version__', 'other')]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.setUp(); self.call(*self.inputs()); mutate(); self.call(*self.inputs())
                self.assertFalse(self.hot)
                self.assertEqual(len(self.cold), 2)
        self.setUp()
        scope = {'a': 1}
        self.jit.used_global_vals = {('a', 0): (1, scope)}
        self.call(*self.inputs()); scope['a'] = 2; self.call(*self.inputs())
        self.assertFalse(self.hot)

    def test_all_tensor_specializations_invalidate(self):
        for i in range(6):
            for field in ('dtype', 'strides', 'ptr', 'shape'):
                with self.subTest(i=i, field=field):
                    self.setUp(); args = self.inputs(); self.call(*args)
                    t = args[i]
                    if field == 'dtype': t.dtype = 'other'
                    if field == 'strides': t.strides = tuple(v*2 for v in t.strides)
                    if field == 'ptr': t.ptr += 8
                    if field == 'shape': t.shape = (t.shape[0]+1, *t.shape[1:])
                    self.call(*args)
                    self.assertFalse(self.hot)
        self.setUp(); self.call(*self.inputs())
        self.ns['torch'].empty = lambda shape, dtype, device: Tensor(shape, dtype, ptr=264)
        self.call(*self.inputs()); self.assertFalse(self.hot)

    def test_thread_pid_device_and_bounded_eviction(self):
        self.call(*self.inputs()); self.call(*self.inputs(device=1))
        t = threading.Thread(target=lambda: self.call(*self.inputs()))
        t.start(); t.join()
        self.assertEqual(len(self.cold), 3)
        self.ns['_launch_local'].pid = -1
        self.call(*self.inputs()); self.assertEqual(len(self.cold), 4)
        for n in range(70):
            args = self.inputs(); args[3].shape = (128+n, 128); self.call(*args)
        self.assertEqual(len(self.ns['_launch_cache'](0)), 64)

    def test_no_retry_after_suppression_or_hot_exception(self):
        self.suppressed = True
        with self.assertRaisesRegex(RuntimeError, 'suppressed'):
            self.call(*self.inputs())
        self.assertEqual(len(self.cold), 1)
        self.assertFalse(self.ns['_launch_cache'](0))
        self.setUp(); self.call(*self.inputs()); self.fail_hot = True
        with self.assertRaisesRegex(RuntimeError, 'submitted hot failure'):
            self.call(*self.inputs())
        self.assertEqual(len(self.cold), 1)
        self.assertEqual(len(self.hot), 1)

    def test_unsupported_never_submits(self):
        self.ns['supported'] = lambda *args: False
        self.assertIsNone(self.call(*self.inputs()))
        self.assertFalse(self.cold or self.hot)


if __name__ == '__main__':
    unittest.main()

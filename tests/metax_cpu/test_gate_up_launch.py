"""CPU launch contracts; AST loading deliberately imports no GPU packages."""
import ast
from collections import Counter, OrderedDict, defaultdict
from contextlib import nullcontext
import importlib
import os
from pathlib import Path
import threading
from types import SimpleNamespace as NS
import unittest


SOURCE = Path(__file__).parents[2] / 'src/flag_gems/runtime/backend/_metax/fused/gate_up_silu.py'


class Tensor:
    def __init__(self, shape=(4, 2048), device=0, ptr=256):
        self.shape = shape
        self.device = NS(index=device)
        self.dtype = 'bf16'
        self.ptr = ptr

    def stride(self):
        return (self.shape[1], 1)

    def data_ptr(self):
        return self.ptr


class LaunchContracts(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name not in ('paired_gate_up_kernel', 'supported', '_baseline_record',
                                        '_dependency', 'paired_intermediate')]
        self.cold, self.hot = [], []
        self.stream = 1
        self.suppressed = False
        self.jitclass = NS(cache_hook=None)
        self.knobs = NS(runtime=NS(jit_post_compile_hook=None))
        self.config = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
                           num_warps=4, num_stages=2)
        self.original = (object(), dict(self.config), None, None, ())
        self.entry = NS(jit_function=object(), kernel_cache=defaultdict(dict),
                        _has_flagtune_tuner=False)
        self.entry.kernel_cache[0]['base'] = self.original
        self.entry.kernel_cache[1]['base'] = self.original
        self.generic = NS(linear_kernel=self.entry)
        self.metax = NS(linear=object())
        self.flag_gems = NS(linear=self.metax.linear)
        outer = self

        class Kernel:
            def __getitem__(self, grid):
                def runner(*args):
                    outer.hot.append((args, outer.stream, grid))
                return runner

        class Jit:
            def __init__(self):
                self.fn = object()
                self.src = 'source'
                self.debug = None
                self.noinline = None
                self.cache_key = 'hash'
                self.cache = defaultdict(dict)
                self.pre_run_hooks = []
                self.used_global_vals = {}
                self.binder = self.bind

            def bind(self, *args, **kwargs):
                return None, ['key'], (), args[:4], {}

            def run(self):
                pass

            def __getitem__(self, grid):
                def normal(*args, **kwargs):
                    outer.cold.append((args, kwargs, grid))
                    for hook in self.pre_run_hooks:
                        hook(*args, **kwargs)
                    if outer.suppressed:
                        return None
                    key = 'key' + str(((), {}))
                    cache = self.cache[args[0].device.index]
                    kernel = cache.setdefault(key, Kernel())
                    return kernel
                return normal

        self.jit = Jit()
        self.ns = dict(os=os, threading=threading, OrderedDict=OrderedDict, Counter=Counter,
                       JITFunction=self.jitclass, knobs=self.knobs,
                       triton=NS(__version__='3.0.0', cdiv=lambda a, b: (a+b-1)//b),
                       torch=NS(empty=lambda shape, device, dtype: Tensor(shape, device.index)),
                       torch_device_fn=NS(device=lambda _: nullcontext()),
                       paired_gate_up_kernel=self.jit, _local=threading.local(), _diagnostics=True,
                       supported=lambda x, w: True,
                       _baseline_record=lambda x, w: dict(self.config))
        exec(compile(tree, str(SOURCE), 'exec'), self.ns)

        def dependency(x, weight):
            return (self.flag_gems, self.generic, self.metax, self.entry,
                    self.entry.jit_function, self.entry.kernel_cache, 'base',
                    self.original, tuple(self.original[1].items()))
        self.ns['_dependency'] = dependency
        self.call = self.ns['try_gate_up_silu']

    def inputs(self, rows=4, device=0):
        return Tensor((rows, 2048), device), Tensor((12288, 2048), device)

    def test_cold_once_then_current_inputs_outputs_stream(self):
        x, w = self.inputs()
        first = self.call(x, w)
        x2, w2 = self.inputs()
        self.stream = 7
        second = self.call(x2, w2)
        self.assertEqual(len(self.cold), 1)
        self.assertEqual(len(self.hot), 1)
        self.assertIsNot(first, second)
        args, stream, grid = self.hot[0]
        self.assertIs(args[0], x2)
        self.assertIs(args[1], w2)
        self.assertIs(args[2], second)
        self.assertEqual(args[3], 4)
        self.assertEqual(stream, 7)
        self.assertEqual(grid, (1, 96, 1))
        # Descriptor stores no current tensors or pointers.
        record = next(iter(self.ns['_cache'](0).values()))
        self.assertFalse(any(isinstance(v, Tensor) for v in record))

    def test_changed_compilation_or_configuration_invalidates(self):
        mutations = [lambda: self.jit.cache[0].clear(),
                     lambda: setattr(self.jit, 'cache', defaultdict(dict)),
                     lambda: setattr(self.jit, 'src', 'changed'),
                     lambda: setattr(self.jit, 'cache_key', 'changed'),
                     lambda: setattr(self.jit, 'debug', True),
                     lambda: self.original[1].update(num_warps=8),
                     lambda: self.entry.kernel_cache[0].clear(),
                     lambda: setattr(self.entry, '_has_flagtune_tuner', True),
                     lambda: setattr(self.entry, 'jit_function', object()),
                     lambda: setattr(self.flag_gems, 'linear', object()),
                     lambda: setattr(self.generic, 'linear_kernel', object())]
        for mutate in mutations:
            self.setUp()
            self.call(*self.inputs())
            mutate()
            self.call(*self.inputs())
            self.assertFalse(self.hot)
            self.assertEqual(len(self.cold), 2)

    def test_dynamic_hooks_use_normal_jit(self):
        for hook in ('pre', 'cache', 'post'):
            self.setUp()
            self.call(*self.inputs())
            observed = []
            if hook == 'pre':
                self.jit.pre_run_hooks.append(lambda *a, **kw: observed.append(True))
            elif hook == 'cache':
                self.jitclass.cache_hook = object()
            else:
                self.knobs.runtime.jit_post_compile_hook = object()
            self.call(*self.inputs())
            self.assertFalse(self.hot)
            self.assertEqual(len(self.cold), 2)
            if hook == 'pre':
                self.assertEqual(observed, [True])

    def test_changed_jit_global_is_not_bypassed(self):
        scope = {'value': 1}
        self.jit.used_global_vals = {('value', id(scope)): (1, scope)}
        self.call(*self.inputs())
        scope['value'] = 2
        self.call(*self.inputs())
        self.assertFalse(self.hot)

    def test_hook_suppressed_execution_returns_full_fallback(self):
        self.suppressed = True
        self.assertIsNone(self.call(*self.inputs()))
        self.assertFalse(self.ns['_cache'](0))

    def test_device_thread_pid_and_lru(self):
        self.call(*self.inputs())
        self.call(*self.inputs(device=1))
        thread = threading.Thread(target=lambda: self.call(*self.inputs()))
        thread.start()
        thread.join()
        self.assertEqual(len(self.cold), 3)
        self.assertFalse(self.hot)
        self.ns['_local'].pid = -1
        self.call(*self.inputs())
        self.assertEqual(len(self.cold), 4)
        for rows in range(1, 67):
            self.call(*self.inputs(rows=rows))
        self.assertEqual(len(self.ns['_cache'](0)), 64)
        self.assertGreater(self.ns['_local'].counts['evicted'], 0)

    def test_unsupported_and_unaligned_output(self):
        self.ns['supported'] = lambda x, w: False
        self.assertIsNone(self.call(*self.inputs()))
        self.assertFalse(self.cold)
        self.ns['supported'] = lambda x, w: True
        self.call(*self.inputs())
        self.ns['torch'].empty = lambda shape, device, dtype: Tensor(shape, device.index, 257)
        self.call(*self.inputs())
        self.assertFalse(self.hot)


if __name__ == '__main__':
    unittest.main()

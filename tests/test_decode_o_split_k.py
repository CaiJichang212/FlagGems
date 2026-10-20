"""Controller-only GPU gate: actual v2 model inputs, frozen A/B and FP64 capture.

The worker already owns the process group, GPU lease and resource monitoring.
One matching eager16384 model load, checkpoint advice, ten short development
requests (batches M1/2/3/4), then shutdown. These are validation requests only.
"""
import ast
import gc
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
import unittest
from unittest.mock import patch

cpu_path = Path(__file__).parent / 'metax_cpu/test_decode_o_split_k.py'
spec = importlib.util.spec_from_file_location('_decode_o_cpu', cpu_path)
cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)
SupportContracts = cpu.SupportContracts
capture_spec = importlib.util.spec_from_file_location('_decode_o_capture_cpu',
    cpu_path.with_name('test_decode_o_capture.py'))
capture_cpu = importlib.util.module_from_spec(capture_spec)
capture_spec.loader.exec_module(capture_cpu)
CaptureContracts = capture_cpu.CaptureContracts


def binary(tensor, path):
    value = tensor.detach().to('cpu').contiguous()
    path.write_bytes(value.view(__import__('torch').uint16).numpy().tobytes())
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                shape=list(value.shape), format='bf16-le')


def make_capture_observer(testcase, original, launches, want, captures, weights,
                          output, refs, binary, synchronize):
    """Bind unittest assertions separately from the actual attention receiver.

    Never fabricate A/B or swallow a submitted error. Publish a case only after
    the actual candidate launch and the original O module have both completed.
    Dependencies are injectable so the producer itself is CPU regression tested.
    """
    def observed(attention, x, phase, layer):
        before = len(launches)
        result = original(attention, x, phase, layer)
        testcase.assertEqual(len(launches), before + 1, 'actual O silently fell back')
        key = (x.shape[0],layer)
        if phase == 'decode' and key in want and key not in captures:
            # Call the original actual O module, including platform dispatch;
            # no torch reference is substituted for A and no B recomputation.
            a = attention.o_proj(x)[0]
            synchronize()
            case = want[key]
            if layer not in weights:
                weights[layer] = binary(attention.o_proj.weight.T, output.with_name(f'weight-layer-{layer}.bf16'))
            captures[key] = dict(case_id=case['id'], origin=case['origin'], phase=phase,
                layer=layer, source_refs=refs,
                input=binary(x, output.with_name(f'input-m{key[0]}-layer-{layer}.bf16')),
                weight=weights[layer], A=a.float().cpu().tolist(), B=result.float().cpu().tolist())
            output.write_text(json.dumps(list(captures.values()),indent=2))
        return result
    return observed


class ActualModelControlled(unittest.TestCase):
    def test_actual_decode_capture_and_boundaries(self):
        import torch
        from vllm import LLM, SamplingParams
        from c500_gpu_guard import GpuLease
        from c500_checkpoint_post_load import reclaim_after_load
        op = importlib.import_module('flag_gems.runtime.backend._metax.fused.decode_o_split_k')
        plugin = importlib.import_module('vllm_fl.models.minicpm5_rope_kv')
        protocol_file = Path(os.environ['C500_FROZEN_PROTOCOL']).resolve()
        p = json.loads(protocol_file.read_text())
        output = Path(os.environ['C500_NUMERICAL_CAPTURE_PATH']).resolve()
        refs = p['source_refs']
        # Verify frozen A has the original O module call; baseline outputs below
        # call that same live, unmodified module on the very same input/weight.
        baseline = protocol_file.parent/'A/plugin/vllm_fl/models/minicpm5_rope_kv.py'
        a_tree = ast.parse(baseline.read_text())
        self.assertNotIn('project_decode_o', {n.name for n in a_tree.body if isinstance(n,ast.FunctionDef)})
        self.assertIn('output, _ = self.o_proj(fused_output.view(-1, self.q_size))', baseline.read_text())
        cases = p['numerical_policy']['cases']
        want = {(c['shape'][0],c['layer']):c for c in cases}
        captures = {};weights = {};requests = [];llm = None
        loads = 0;init_attempts = 0;shutdown_completed = False
        events = output.with_name('validation-model-events.jsonl')
        def event(kind, **values):
            with events.open('a') as stream:
                stream.write(json.dumps(dict(wall=time.time(),pid=os.getpid(),event=kind,**values))+'\n')
        original = plugin.project_decode_o
        kernel_original = op.try_decode_o
        launches = []
        def launched(x, weight, *, phase, layer):
            result = kernel_original(x, weight, phase=phase, layer=layer)
            self.assertIsNotNone(result, 'actual dispatch reached unsupported kernel')
            launches.append((x.shape[0], layer))
            return result
        observed = make_capture_observer(self, original, launches, want, captures,
            weights, output, refs, binary, torch.cuda.synchronize)
        # All model/resource options come from the frozen protocol, no downscale.
        try:
            with GpuLease() as lease:
                init_attempts += 1
                event('model_init_attempt',model=p['model'],engine=p['engine'])
                llm = LLM(model=p['model'], **p['engine'])
                loads += 1
                event('model_loaded')
                reclaim_after_load(protocol_file.parent, output.with_name('checkpoint-post-load.json'),
                    lease, torch.cuda.synchronize, p['model'], p['checkpoint_post_load_reclaim'])
                main = json.loads((protocol_file.parent/'main-selection.json').read_text())
                selected = [r for r in main if r['scenario'] == 'prefill-1k']
                with patch.object(plugin, 'project_decode_o', observed), patch.object(op, 'try_decode_o', launched):
                    for m in (1,2,3,4):
                        prompts = [dict(prompt_token_ids=r['prompt_token_ids']) for r in selected[:m]]
                        event('validation_batch_submit',batch=m,planned_requests=m)
                        values = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=4,
                            ignore_eos=True, seed=20261020), use_tqdm=False)
                        self.assertEqual(len(values),m)
                        for i,v in enumerate(values):
                            self.assertEqual(len(v.outputs[0].token_ids),4)
                            event('validation_request_complete',batch=m,index=i,request_id=v.request_id,
                                prompt_tokens=len(v.prompt_token_ids),completion_tokens=len(v.outputs[0].token_ids),
                                finish_reason=v.outputs[0].finish_reason)
                            requests.append(dict(batch=m,index=i,request_id=v.request_id,
                                usage=dict(prompt_tokens=len(v.prompt_token_ids),completion_tokens=len(v.outputs[0].token_ids),total_tokens=len(v.prompt_token_ids)+len(v.outputs[0].token_ids)),parent_index=selected[i]['parent_index'] if 'parent_index' in selected[i] else selected[i].get('id'),
                                prompt_token_ids=v.prompt_token_ids,output_token_ids=v.outputs[0].token_ids,
                                finish_reason=v.outputs[0].finish_reason))
                        output.with_name('validation-model-requests.json').write_text(json.dumps(dict(
                            source_refs=refs,engine=p['engine'],model=p['model'],expected_requests=10,
                            requests=requests,scope='pre-service capture only, excluded from performance'),indent=2))
                self.assertEqual(set(captures),set(want),'actual layer/M cases missing')
                # Unsupported modes must return before launch; no global toggles.
                with torch.inference_mode():
                    x=torch.randn((4,2048),device='cuda',dtype=torch.bfloat16)
                    w=torch.randn((2048,2048),device='cuda',dtype=torch.bfloat16)*.02
                    x0=x.clone();w0=w.clone()
                    for m in (1,2,3,4):
                        a=op.try_decode_o(x[:m],w,phase='decode',layer=0)
                        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            b=op.try_decode_o(x[:m],w,phase='decode',layer=0)
                        torch.cuda.current_stream().wait_stream(stream)
                        self.assertTrue(torch.equal(a,b))
                        self.assertNotEqual(a.data_ptr(),b.data_ptr())
                    self.assertTrue(torch.equal(x,x0));self.assertTrue(torch.equal(w,w0))
                    self.assertIsNone(op.try_decode_o(x,w,phase='prefill',layer=0))
                    self.assertIsNone(op.try_decode_o(x.repeat(2,1),w,phase='decode',layer=0))
                    self.assertIsNone(op.try_decode_o(x.float(),w,phase='decode',layer=0))
                    self.assertIsNone(op.try_decode_o(x[:,::2],w,phase='decode',layer=0))
                runner=llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
                attention=next(mod for mod in runner.model.modules() if isinstance(mod,plugin.RopeKVAttention))
                hook_calls=[]
                handle=attention.o_proj.register_forward_hook(lambda *args:hook_calls.append(True))
                try:
                    with torch.inference_mode(), patch.object(op,'try_decode_o',wraps=kernel_original) as candidate:
                        fallback=original(attention,x,'decode',0)
                        self.assertEqual(candidate.call_count,0)
                        self.assertEqual(len(hook_calls),1)
                        self.assertEqual(tuple(fallback.shape),(4,2048))
                finally:
                    handle.remove()
                with torch.enable_grad():
                    self.assertIsNone(op.try_decode_o(x,w,phase='decode',layer=0))
                torch.cuda.synchronize()
        finally:
            # Match the existing development worker shutdown path. Failures are
            # retained and the parent monitor confirms owned resource exit.
            if llm is not None:
                event('model_shutdown_attempt')
                llm.llm_engine.engine_core.shutdown()
                shutdown_completed = True
                event('model_shutdown_complete')
                del llm
                gc.collect()
            output.with_name('validation-model-lifecycle.json').write_text(json.dumps(dict(
                source_refs=refs,service_starts=0,validation_model_init_attempts=init_attempts,
                validation_model_loads=loads,shutdown_completed=shutdown_completed,command=sys.argv,
                expected_requests=10,completed_requests=len(requests),captured_cases=len(captures),
                expected_cases=len(cases),observer_restored=plugin.project_decode_o is original,
                note='parent monitor is authoritative for shutdown/resource safety'),indent=2))


if __name__ == '__main__': unittest.main()

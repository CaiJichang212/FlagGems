# Experimental BI-V150 grouped GEMM

The default whitelist is empty. No candidate is enabled without an explicit
opt-in. Both the baseline and candidate use explicit masked loads for sub-16
output dimensions or input addresses/row pitches not aligned to 128 bytes.
This avoids the BI-V150 unmasked SME-load lowering that failed the original
small-M and padded-input correctness cases. No contiguous copy or tolerance
change is used. The repair changes the baseline JIT cache identity, so compare
optimization results with a freshly measured repaired baseline.

For a frozen model experiment, set `FLAGGEMS_ILUVATAR_MM_CONFIG` **before starting
Python** to a JSON object, for example:

```sh
export FLAGGEMS_ILUVATAR_MM_CONFIG='{"2048,2560,2048":{"block_m":256,"block_n":128,"block_k":32,"num_warps":8,"num_stages":2,"group_m":8}}'
```

This example is an unvalidated candidate, not a recommended winning configuration.
The module parses and validates the variable once on import. Invalid JSON, shape
keys or configuration values raise `ValueError`; there is no silent fallback for
bad configuration. Unset the variable to restore original dispatch in a fresh
process. Runtime diagnostics can explicitly call `configure_mm_candidate({})`
to clear the process whitelist.

The opt-in supports only M=2048 with (N,K)=(2560,2048), (2048,2048),
(12288,2048), or (2048,6144), BF16 inputs/output, contiguous row-major A/output,
and transposed contiguous weights (B strides 1,K). Other calls retain original
`mm`/`mm_out` dispatch. Candidate launches use fixed tiles, FP32 accumulation,
and SPLIT_K=1. No tuning runs on the candidate path, including Graph capture.

`launch_mm_fixed(..., grouped=False)` launches the original JIT kernel with
explicit configuration; `grouped=True` launches `grouped_mm_kernel`. It returns
the compiled kernel for metadata/IR inspection. The shared launch API accepts
`block_m`, `block_n`, `block_k`, `num_warps`, `num_stages`, and `group_m`.
Keep the same tile configuration when testing exact program-ID-only equivalence.

Relevant GPU tests must run serially under the parent experiment process registry
and deadline guard, with the device idle. On the prepared BI-V150 image, initialize
the installed platform compatibility layer before importing pytest/FlagGems; a
bare `python3 -m pytest` lacks the required `triton.ops` compatibility shim.
The registered test launcher runs this body from the parent project root:

```python
from scripts.flagos_decode_bench import initialize_runtime

report = {}
initialize_runtime(report)
import pytest

raise SystemExit(pytest.main(["task/FlagGems/tests/test_iluvatar_grouped_mm.py", "-v"]))
```

Preserve `report` with the experiment runtime evidence. The fixture resolves
`flag_gems.mm.__module__` and checks both public functions' globals, so the tests
exercise the dynamically registered vendor module rather than a second import
under a different module name.

Do not enable a default whitelist until operator/Graph correctness, real model
execution evidence, all 105 Level 3 records, and official 4K/16K performance gates
pass and improvement against the same-base B0 is established.

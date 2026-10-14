# MetaX SiLU single-call metadata extraction candidate

Baseline U: a1304d30b78c80411e8534185e7d0526d26fdd2d. Plugin remains exactly
910375835be4ca0ad30b1ac5135402f84b3d7b55; no LlamaMLP binding or new kernel.

Hypothesis: U reads shape/device/dtype/strides repeatedly during admission,
signature construction and cold descriptor preparation. Extract each supported
input field once per invocation and reuse it throughout those preparation steps.
Do not cache tensor values, addresses or per-call metadata between invocations.

Public `silu_and_mul(A,B)` and internal compatibility `_supported(A,B)` remain.
`_metadata(A,B)` returns shape,device,dtype,Astride,Bstride,deviceindex or None;
`_descriptor(...,metadata=None)` can reuse this invocation's admitted metadata.
Domain stays exact plain Tensor, BF16 CUDA rank2 equal sliced layouts
stride=(2*N,1), 16-byte alignment, inference/no-gradient, 1<=M<=1024, N>0.
Parameter input, unsupported ranks/strides/gradients still call generic activation.

The scalar function, pointwise configuration, generated kernel, descriptor key
specialization, process/thread bounded LRU, current-stream runner, and all U
function/config/compiled-cache invalidation conditions stay unchanged.
Actual registered backend is `_metax.fused.silu_and_mul`, not a second full-name
import with a separate cache. `diagnostics(True/False)` preserves U counters
hit/cold/fallback/config_fallback/invalidated/unsupported_entry/evicted.
Getter multiplicity is tested separately; no observer is added to the hot path.

Acceptance uses exact U numerical values and identical original kernel binaries.
The failed fused-compute experiment's strict unscaled whole-FFN independent
reference gate is not reused as a preparation-only acceptance requirement.
CPU getter/lifecycle evidence alone does not establish GPU exactness, actual model
benefit, throughput or TTFT. Independent U/V full-module replay and frozen model
performance/precision gates still decide adoption.

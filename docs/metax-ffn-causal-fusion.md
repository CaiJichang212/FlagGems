# Experimental paired gate/up GEMM and SiLU

Explicit baseline U: a1304d30b78c80411e8534185e7d0526d26fdd2d.
Public linear and SiLU interfaces/dispatch remain unchanged. The internal
try_gate_up_silu returns None for unsupported inputs; the plugin retains full FFN.

The paired dot maps adjacent columns to gate/up using existing weight layout;
total dot tile width and K accumulation follow U. Both GEMM results explicitly
round to BF16 before FP32 fdiv/exp and multiplication, and output rounds to BF16.
A constexpr control writes the original intermediate for column-mapping isolation.
No split-K, packing, atomic reduction, shapes whitelist or parameter replacement.

The actual registered _metax.ops.linear module supplies U's warmed LibEntry record;
first cold call warms U once to select metadata. Hot calls validate original cache
identity. Unsupported record/hooks/configuration falls back, never guesses a tile.
M support is 1 through 1024; device BF16 contiguous [M,2048], [12288,2048], no grad.

CPU syntax/mapping/fallback checks do not prove compilation, numerical correctness,
actual model dispatch or model benefit. All GPU and model gates remain required.

# MetaX SiLU preparation reuse for bounded two-dimensional sliced inputs

This experimental candidate extends the existing support range from M=1..7 to
M=1..1024. It uses the same compiled pointwise kernel/configuration and the same
output allocation, standard launcher, current stream, cache validation and
unsupported-path fallback. It adds no layer, token, topic or observed-M list.

Development starts at FlagGems 480d67922f020b775a2a35efed90238380c84388 with
common plugin 910375835be4ca0ad30b1ac5135402f84b3d7b55.
The preceding S runtime counterfactual in the project artifact
results/metax/mlp-ffn-ttft-20261012-01/silu-screen passed the original 131 linear
and 43 activation cases plus captured-layout and complete-FFN exactness checks.
The counterfactual measurements belong to S with runtime support substitution;
this later archival candidate commit was not the measured run commit.

CPU contracts cover 7/8, 1023/1024/1025, non-model tail widths, layout/gradient/
alignment fallbacks, cold/hot calls, cache changes, eviction and current stream.
They do not establish GPU exactness for every newly supported row or width.
Fresh candidate numerical and end-to-end gates remain required. Not adopted.

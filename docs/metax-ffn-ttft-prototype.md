# MetaX early linear rejection prototype: negative result retained

This independent prototype moves rejection of non-FFN weight shapes before
reading input shape and both tensor device/dtype properties. Accepted calls
still extract metadata once, without cross-call tensor decisions.

Base: 480d67922f020b775a2a35efed90238380c84388. Common plugin:
910375835be4ca0ad30b1ac5135402f84b3d7b55.
Only S plus a runtime helper counterfactual was measured in project artifact
results/metax/mlp-ffn-ttft-20261012-01/reject-S-v2. Full public-call evidence did
not support advancing this mechanism. The shapes/layouts and object types were
matched to observed calls; constructed zero values were not actual model data.
This later archival commit must not be reported as the measured run commit.

Not combined with SiLU, not adopted, and not used by the default model path.

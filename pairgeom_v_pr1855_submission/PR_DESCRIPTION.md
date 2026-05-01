## Pair-Geometric Value Projection on PR #1855
This submission starts from the accepted PR #1855 stack and replaces the dense
attention value projection with a structured pair-geometric value projection.

Baseline value path:
```text
v = W_v x
```

PairGeom-V value path:
```text
base = rms_norm(x)
a = base[:kv_dim]
b = base[kv_dim:2*kv_dim]
d = a - b
s = a + b
v = a*w0 + b*w1 + d*wd + s*ws
```

The validated setting uses `PAIRGEOM_V_COLLAPSE=1`, which algebraically reduces
the signed rule to per-dimension learned coefficients on the two hidden halves.
This removes the dense trained/stored `W_v` value matrix while keeping the
accepted PR #1855 recipe otherwise aligned.

Validation:
```text
hardware:      JarvisLabs 8xH100 80GB HBM3
seeds:         42 and 43
train target:  600s wall-clock
steps:         4981, 4996
artifacts:     15,304,981 and 15,312,945 bytes
post-TTT BPB:  1.07006241, 1.07031169
mean BPB:      1.07018705
std BPB:       0.00017627
```

Seed 43 completed, but we ran out of time before recovering the full remote stdout log, so this PR includes a recovered summary log instead.

Comparison:
```text
reproduced PR #1855 baseline: 1.06021565 BPB
PairGeom-V candidate mean:    1.07018705 BPB
delta:                        +0.00997140 BPB
```

Claim boundary:
This is an architectural alteration using a pair-geometric value-projection model rather than a new SOTA claim. It replaces the dense attention value projection path, but it does not
replace Q/K projections, attention score dot products, output projection, or MLP
matrix products.

# Claude SP10240 Byte-Fit Plan - 2026-04-30

Snapshot: `2026-04-30T18:35Z`

## Accepted Advice

Claude's strongest point is correct: stop prioritizing the queued 10L
`mlp425_late050`. The useful move is to keep 11L and fit between the over-cap
MLP4 body and the under-cap MLP3.75 body.

## Byte Math Correction

The proposed `3.85-3.90` range is directionally right but probably too high:

```text
MLP4.0 over cap: about +450KB
MLP3.75 under cap: about -182KB
gap across 0.25 MLP: about 632KB
byte-fit point from MLP3.75: about 0.072 MLP
```

That places the safe target near `MLP3.8125`, not raw `3.85-3.90`.

Also, raw `MLP3.85` failed on the quads with:

```text
AssertionError: strides must be 16-byte aligned
```

So prepared candidates use aligned hidden dims:

- `MLP3.8125`: hidden dim `1952`
- `MLP3.84375`: hidden dim `1968`

## Prepared 8x Runners

Safe byte-fit:

```bash
cd /workspace/sota_rascal/legs/2026-04-30_pr1855_sp10240_caseops_mlp38125_late050_8x
./launch_8x.sh
tail -f logs/pr1855_sp10240_caseops_mlp38125_late050_8x_seed444.txt
```

Edge byte-fit:

```bash
cd /workspace/sota_rascal/legs/2026-04-30_pr1855_sp10240_caseops_mlp384375_late050_8x
./launch_8x.sh
tail -f logs/pr1855_sp10240_caseops_mlp384375_late050_8x_seed444.txt
```

H1 CaseOps hot-loop precision:

```bash
cd /workspace/sota_rascal/legs/2026-04-30_pr1855_sp10240_caseops_mlp4_late050_h1_hotloop_8x
./launch_8x.sh
tail -f logs/pr1855_sp10240_caseops_mlp4_late050_h1_hotloop_8x_seed444.txt
```

## H1 Port

H1 is now ported onto the CaseOps code path:

- body: SP10240 CaseOps, 11L, MLP4, late050
- global `matrix_bits=5`
- hot loop attention blocks `3,4,5` use int6
- embed int7, pergroup, LQER asym rank4/top3 kept

This is a quant-side test, not the first body-quality run. Run after the
byte-fit body shows enough neural/size signal.

## Quads Status During Prep

Quads are hot:

- four H100s at `99-100%` utilization
- about `45GB` used per GPU
- active lanes around step `3000`
- no validation read yet
- old raw `MLP3.85` lane failed alignment and is not active

Active useful quads lanes:

- `caseops6_mlp375_late050_loopoff40_1x`
- `caseops6_mlp375_late050_loopsmooth_1x`
- `caseops6_mlp375_late050_smooth_loopoff40_1x`
- `caseops6_mlp375_late045_loopoff40_1x`

At step ~3000 the train losses are too close to cut. Wait for validation or
wallclock stop.

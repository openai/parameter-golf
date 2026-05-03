# Step 0

## Measurement contract

- **Target**: minimize deployed downstream `score_bpb`, which for this script is `quantized_sliding_window` when `SLIDING_WINDOW_ENABLED=1` (the default).
- **Baseline anchor**: published pr1394 stack reports mean pre-quant `val_bpb` `1.09021`, post-quant contiguous `val_bpb` `1.10235`, and deployed sliding score `1.08563` at `15,985,678` bytes.
- **Ranking rule**: primary = downstream `quantized_sliding_window`; secondary = pre-quant `val_bpb`; tie-break only = artifact bytes.
- **Budget rule**: preserve the frozen SP8192/GPTQ/recurrence/MuonEq/SDClip base contract unless a lane names a real broken predicate on the score path.
- **Control rule**: every executable pack needs `C0` and `C1`; refuse to rank novel lanes against a single anchor.

## Score path to the deployed metric

1. Load the fixed `fineweb_val_*` validation shards and build byte-count LUTs from the SP8192 tokenizer.
2. Train under the wallclock cap, with `gptq_reserve_seconds` carved out before the cap is enforced.
3. Apply EMA weights to form the checkpoint that actually gets exported.
4. Run `eval_val(...)` on the EMA model; this is diagnostic pre-quant evidence only.
5. Serialize by collecting GPTQ Hessians from calibration batches, mixed-quantizing weights, compressing the artifact, and recording total bytes.
6. Deserialize the quantized artifact back into an eval model.
7. Run contiguous `quantized` eval; this is still not the deployed winner metric when sliding is enabled.
8. Run `eval_val_sliding(...)` with `eval_stride=64`; only the scored suffix of each window contributes after warm context. This branch is the deployed `score_bpb`.

**Implication**: a lane is only first-order if it can survive all of `training -> EMA selection -> GPTQ export -> deserialize -> sliding-window eval`. Pre-quant gains that disappear after export do not count as wins.

## Mutation map

| Surface | Causal role | Score-path reach | Cheap signal | Search level |
| --- | --- | --- | --- | --- |
| Late checkpoint selection | Chooses which trained state is exported | Direct to deployed metric | same-trace downstream delta vs controls | executable now |
| Export recipe / GPTQ menu | Changes calibration, quantization, compression, bytes | Direct to deployed metric | quantized-prequant gap, bytes | executable now if catalog-backed |
| Loop / recurrence handoff | Changes when repeated depth becomes active | Upstream through training, EMA, export | train-loss slope, tok/s, pre-quant `val_bpb` around handoff | patch-level |
| Optimizer phase law | Changes per-step geometry and parameter-group behavior | Upstream through training, EMA, export | matched-wallclock train loss, tok/s, pre-quant `val_bpb` | patch-level |
| Late-path controller (`xsa`, skip correction) | Changes late representation routing before export | Upstream + quant-sensitivity interaction | pre/post quant gap, late validation delta | patch-level |
| GPTQ calibration / clip law | Changes Hessian estimate quality and quantization-size tradeoff | Direct to deployed metric | same-checkpoint downstream delta, bytes, export time | patch-level or export-only bakeoff |

## Broken-predicate shortlist

1. **`final checkpoint + default export` is automatically the best deployed artifact.** The script exports and scores one final EMA artifact only; no late-window checkpoint sweep exists.
2. **Export is passive once training is done.** False on path: the deployed score is measured only after Hessian collection, mixed GPTQ quantization, compression, deserialize, and sliding eval.
3. **Pre-quant `val_bpb` is a safe proxy for deployed score.** The baseline itself changes materially between pre-quant, post-quant contiguous, and post-quant sliding.
4. **One control is enough to rank a pack.** The findings file shows a nominal control repeat split of roughly `0.60` BPB, which is larger than many plausible candidate effects.
5. **Short mixed-lane screens are honest.** Findings show eval ideas being tested as full training runs and export-heavy ideas dying in calibration instead of on merit.
6. **The current handoff/controller defaults are already optimal.** `enable_looping_at=0.5`, fixed Muon warmup, static `xsa_last_n`, and static skip correction are hard-coded choices, not verified truths.

## Final candidate lanes

### H0 - Late checkpoint selector under frozen default export

- **Target surface**: post-train checkpoint identity from one frozen baseline trace.
- **Target predicate**: the published final EMA checkpoint is the best downstream artifact when export stays fixed.
- **Why baseline truth is false**: `train_gpt_human.py` only exports and evaluates the final EMA artifact; no late-checkpoint comparison exists, so "final is best" is assumed rather than measured.
- **Score path trace**: one baseline training trace -> choose late checkpoint vs final checkpoint -> default export path -> deserialize -> `quantized_sliding_window`.
- **Cheap signal**: same-trace downstream delta to `C0`/`C1`, with checkpoint-local pre-quant `val_bpb` used only for diagnosis.
- **Likely failure mode**: late checkpoints are downstream-indistinguishable once control spread is measured, or the harness exposes too few saved checkpoints.

### H1 - Frozen-checkpoint export-menu selector

- **Target surface**: post-train export recipe on the same final checkpoint.
- **Target predicate**: the published default GPTQ/export path is already the best downstream realization for the frozen checkpoint.
- **Why baseline truth is false**: the score path runs through GPTQ calibration, clip/bit choices, compression, and deserialize; the README explicitly documents real size/quality tradeoffs, so export is not a passive tail.
- **Score path trace**: frozen final checkpoint -> default vs alternate catalog export recipe -> bytes + deserialize -> `quantized_sliding_window`.
- **Cheap signal**: downstream paired-control delta at fixed checkpoint, plus quantized-prequant gap and artifact bytes.
- **Likely failure mode**: export variants only trade bytes without improving downstream score, or the current harness catalog is too narrow to express a real alternative.

### H2 - Loop / recurrence handoff program

- **Target surface**: `enable_looping_at`, loop warmup behavior, and the recurrence activation rule for the repeated depth segment.
- **Target predicate**: a single fixed mid-run loop activation at `0.5` training fraction is the best recurrence program for the deployed metric.
- **Why baseline truth is false**: recurrence is a phase-transition mechanism, but the current file hard-codes one activation point plus a one-off loop warmup; that is a design choice, not an established optimum.
- **Score path trace**: altered loop-handoff rule -> training dynamics -> EMA checkpoint -> default GPTQ export -> `quantized_sliding_window`.
- **Cheap signal**: train-loss slope and tok/s immediately before and after activation, plus matched-wallclock pre-quant `val_bpb`.
- **Likely failure mode**: the lane collapses into noisy threshold nudges instead of a real handoff program, or current harness primitives cannot express it honestly yet.

### H3 - Optimizer phase controller

- **Target surface**: Muon/Adam parameter-group law, including momentum warmup, row normalization, and phase-dependent group behavior.
- **Target predicate**: the hard-coded MuonEq-R schedule and current parameter-group split are already optimal for downstream sliding score under the same wallclock.
- **Why baseline truth is false**: the file fixes one momentum ramp, one row-normalization choice, and one static group partition even though the objective is measured after quantization and sliding eval, not just train loss.
- **Score path trace**: optimizer phase law -> training speed/quality -> EMA checkpoint -> default export -> `quantized_sliding_window`.
- **Cheap signal**: matched-wallclock tok/s, train-loss slope, and pre-quant `val_bpb`; promote only if downstream direction later agrees.
- **Likely failure mode**: the lane degenerates into local hyperparameter fiddling or any gain is swallowed by control noise.

### H4 - Late-path controller for `xsa` and skip correction

- **Target surface**: `xsa_last_n`, `skip_weights`, and `skip_gates` as the late decoder routing program.
- **Target predicate**: always-on XSA over the last 11 layers plus static corrective skip gating is the best late-path controller for post-quant sliding eval.
- **Why baseline truth is false**: the current routing is static and layer-index-based; it is never compared against alternate late-path programs despite directly shaping the representation that gets quantized and evaluated under sliding windows.
- **Score path trace**: altered late-path controller -> EMA checkpoint -> GPTQ export -> `quantized` / `quantized_sliding_window`.
- **Cheap signal**: pre/post quant gap, late validation delta, and stability after looping turns on.
- **Likely failure mode**: interactions are too entangled to attribute cheaply, or the required controller grammar is richer than the current harness can run.

### H5 - GPTQ calibration / clip-law family

- **Target surface**: `collect_hessians`, calibration batch count, row-std clip law, and matrix/embed bit allocation.
- **Target predicate**: `64` calibration batches, fixed `12.85`/`20.0` clip sigmas, and the current int6/int8 split are already the best downstream size-quality point under the 16MB cap.
- **Why baseline truth is false**: the README explicitly frames clip width and bitwidth as a quality/size tradeoff, and the findings file shows export/calibration cost is first-order rather than a harmless tail step.
- **Score path trace**: shared checkpoint -> Hessian collection -> mixed GPTQ quantization -> compression -> deserialize -> `quantized_sliding_window` + bytes.
- **Cheap signal**: same-checkpoint downstream delta, quantized-prequant gap, calibration wallclock, and artifact bytes.
- **Likely failure mode**: apparent byte wins with flat/worse downstream score, or calibration-heavy variants get unfairly killed if lane budgets are not isolated.

## Controls

### C0 - Published baseline anchor

- Frozen `frontier_rebase/pr1394/train_gpt_human.py` behavior with the published SP8192/GPTQ/recurrence/MuonEq/SDClip stack, default final checkpoint, default export path, and downstream sliding scorer.
- Role: the primary pack anchor for every candidate delta.

### C1 - Exact replay control

- A replay-equivalent repeat of `C0`, using the same lane, same export recipe, and same scoring path as the primary anchor.
- Role: estimate pack-local control spread; if `C0` and `C1` disagree enough to cover candidate effects, the family is unresolved rather than positive.

## Kill list

- Any lane that changes score semantics instead of model behavior: disabling sliding eval, changing byte-count accounting, or mutating tokenizer/validation contracts.
- Any proposal that reopens already-baked frontier-base changes such as vocab-size rebases or whole-stack swaps instead of attacking a still-broken predicate.
- Mixed-lane screens that vary training, export, and eval together under one short budget.
- Size-only or pre-quant-only wins that are neutral/worse on downstream `quantized_sliding_window`.
- Any family ranked without paired controls or with unresolved `C0`/`C1` drift.
- Families that require new harness primitives but are presented as executable-now; those should be marked blocked or rewritten, not smuggled into the live slate.

## Family-stage instructions

1. Regroup `H0`..`H5` by **mechanism and observability**, not by nearby scalar knobs.
2. Keep `H0` and `H1` separate long enough to preserve attribution between checkpoint choice and export choice; only merge them if the family stage can still name both predicates explicitly.
3. Preserve at least one executable-now post-train selector family from `{H0, H1}` and at least two non-selector lineages from `{H2, H3, H4, H5}` so the slate does not collapse to one local mode.
4. Treat `H2`, `H3`, and `H4` as transition-program families first; if the current harness cannot run them honestly, mark them `REWRITE` or `DROP` instead of pretending they are turnkey.
5. Treat `H5` as an export-only bakeoff family when possible: same checkpoint, same scorer, changed calibration/clip law only.
6. Use `C0` and `C1` as explicit anchors for every surviving family, and refuse promotion when control spread hides the candidate signal.
7. Allocate the later six candidate slots only across families whose cheap falsifier matches the score path they claim to affect.

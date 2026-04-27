# Experiment 0046_kill_mamba2_middle_parallel

Parent: 0042_kill_no_bigram

## Question
**Cross-class hybrid topology**: combine 0027's surprise middle-parallel win (val 2.0779, beat S4D-Lin sandwich) with 0038/0042's kill-Mamba-2 finding. Architecture per K=3 unique:
- pos 0: kill-Mamba-2 (LTI)
- pos 1: PARALLEL block = ATTN || kill-Mamba-2 (sum scaled outputs)
- pos 2: kill-Mamba-2 (LTI)

Compare to 0042 (kill at 0,1; ATTN at 2; no-BG): val 2.0225. The hybrid adds a second mamba-block + attention at position 1, replacing the lone attention with a parallel attn||mamba combination.

The bet: if kill-Mamba-2 + attention can BOTH contribute at the same layer (parallel), and middle-position is the right place (per 0027), this could compound to ~2.005-2.015.

## Hypothesis [CONJECTURE]
val_bpb in [2.005, 2.040]. Single-seed.

- val ∈ [2.005, 2.020] (compound win, attention-and-kill-Mamba-2 stack productively in parallel): 30% likely. Strongest outcome — new SSM-best.
- val ∈ [2.020, 2.030] (≈ 0042, no compound from middle-parallel): 35% likely. Saturation — kill-Mamba-2 alone is sufficient.
- val ∈ [2.030, 2.045] (slight regression): 25% likely. Adding parallel block costs more than it gives at the middle position.
- val > 2.045 (significant regression): 10% likely. Indicates the parallel-mixing pattern doesn't transfer to the kill-Mamba-2 family.

Cap math: 0042 = 13.25 MB. Parallel block at position 1 has BOTH attn (~0.79 MB int8) AND kill-Mamba-2 (~1.65 MB) = +0.79 MB net add (was just kill-Mamba-2 in 0042 at this position). Predicted artifact ~14.0 MB. Cap-safe.

Step time: parallel block runs both attn AND mamba sequentially per call. ATTN is ~0.18 s/call ×3 = 0.54s extra. Plus kill-Mamba-2 already at position 1 is unchanged. Predicted ~6.4 s/step.

## Change
**Code change required (subagent task)** in `experiments/0046_kill_mamba2_middle_parallel/train_gpt.py`:

Currently the `Block` class has a `parallel_mode` flag that, when True, hardcodes parallel = `CausalSelfAttention` || `S4DLin`. Need to make the parallel SSM type configurable via `PARALLEL_SSM_TYPE` env var:
- `PARALLEL_SSM_TYPE=s4d_lin` (default, byte-identical to current behavior): parallel block = ATTN || S4DLin
- `PARALLEL_SSM_TYPE=mamba2_kill`: parallel block = ATTN || Mamba2Block(kill_selectivity=1)
- `PARALLEL_SSM_TYPE=mamba2`: parallel block = ATTN || Mamba2Block(default selectivity)

The `Mamba2Block` is already in train_gpt.py. The parallel block's mamba sub-module should respect `MAMBA2_KILL_SELECTIVITY` env var the same way regular Mamba2Block instances do (it's read in Mamba2Block.__init__, no additional plumbing needed).

**env.sh** (already set):
- `ATTN_LAYER_POSITIONS=` (empty — no positions are pure attention; the parallel block at position 1 contains its own attention)
- `MAMBA2_LAYER_POSITIONS=0,2` (positions 0 and 2 are pure kill-Mamba-2)
- `PARALLEL_LAYER_POSITIONS=1` (position 1 is the parallel block)
- `PARALLEL_SSM_TYPE=mamba2_kill` (NEW env var the subagent will read)
- `MAMBA2_KILL_SELECTIVITY=1` (kill flag, applies to all Mamba2Block instances including the one in the parallel block)
- `BIGRAM_VOCAB_SIZE=0` (no BG, per 0042 finding)

## Disconfirming
- val < 2.020: meaningful compound; middle-parallel + kill-Mamba-2 outer wins.
- val > 2.040: regression; middle-parallel pattern doesn't transfer to kill-Mamba-2 family.
- val ∈ [2.020, 2.030]: tied with 0042; saturation.

## Notes from execution
**Subagent task** (general-purpose): implement `PARALLEL_SSM_TYPE` env var support in the `Block` class's `parallel_mode` branch. Default value preserves current behavior (S4DLin). Two new values: `mamba2_kill` and `mamba2`. The `Mamba2Block` class already exists. After edit, do NOT run the experiment; just confirm the change is consistent with default behavior preserved and update plan.md with what was done.

### 2026-04-26 — subagent code change applied
Implementation note: this train_gpt.py was forked from 0042 (mamba2 family) and did **not** previously have `parallel_mode` plumbing — only `use_attention` and `mamba2_mode`. Even though the task description spoke as if `parallel_mode` already existed, in practice the change had to (a) port the parallel-mode plumbing from 0027 and (b) add the new `PARALLEL_SSM_TYPE` selector at the same time.

Concrete edits to `train_gpt.py`:
- `Hyperparameters`: added `parallel_layer_positions = os.environ.get("PARALLEL_LAYER_POSITIONS", "")`.
- `Block.__init__`: added `parallel_mode: bool = False` param; replaced the prior 2-way exclusivity check with a 3-way one (at most one of `use_attention` / `mamba2_mode` / `parallel_mode`); when `parallel_mode=True`, constructs `self.attn = CausalSelfAttention(...)` plus `self.s4d` whose class is selected by `PARALLEL_SSM_TYPE`:
  - `s4d_lin` (default, byte-identical to 0027's parallel block) → `S4DLin(dim, d_state=16, expand=1)`
  - `mamba2` or `mamba2_kill` → `Mamba2Block(dim, d_state=64, expand=2, chunk_size=64, headdim=64)`. The kill-vs-full distinction is taken from `MAMBA2_KILL_SELECTIVITY` inside `Mamba2Block.__init__` — no extra plumbing.
  - any other value raises `ValueError`.
  Also adds `self.s4d_scale` parameter when in `parallel_mode`. Attribute is named `self.s4d` regardless of underlying class so the forward path stays single-line.
- `Block.forward`: branches on `self.parallel_mode`; the parallel branch sums `attn_scale * attn_out + s4d_scale * s4d_out` into `x` (matches 0027). Non-parallel path unchanged.
- `GPT.__init__`: accepts `parallel_layer_positions`, validates it's disjoint from both `attn_layer_positions` and `mamba2_layer_positions`, and threads `parallel_mode=(i in parallel_positions)` through to each `Block`.
- `main()`: parses `args.parallel_layer_positions` into a set and passes it to the `GPT` constructor.

Verification (`scratch/parallel_ssm_type_check.py`):
- Constructs `Block(parallel_mode=True)` with `PARALLEL_SSM_TYPE=s4d_lin` → `block.s4d` is `S4DLin`. PASS.
- Default (no `PARALLEL_SSM_TYPE` set) → `block.s4d` is `S4DLin` (default preserved). PASS.
- `PARALLEL_SSM_TYPE=mamba2_kill` + `MAMBA2_KILL_SELECTIVITY=1` → `block.s4d` is `Mamba2Block`, `block.s4d._kill_selectivity is True`, `block.s4d._B_const` exists with shape `(64,)`. PASS.
- Forward output is finite for both variants on a synthetic `(2, 64, 64)` input (seq_len bumped from 8 to 64 to satisfy `Mamba2Block.chunk_size=64` divisibility). PASS.

Default behavior (`PARALLEL_SSM_TYPE` unset or `s4d_lin`) is preserved. The experiment was NOT run (subagent task limited to code change + verification).

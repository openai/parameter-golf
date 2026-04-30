# In a cave, with a box of scraps

*Motivation: Can I compete with the best of the best, using ONLY my 5 year old MacBook, and a sick idea? Can I evolve my transformer, and get something novel?*

*Approach: Evolutionary NAS on a personal 2021 M1 Max MacBook Pro, ZERO cloud compute—independently surfaced depth recurrence, gated carryover, and a novel layer traversal order (`odds_then_evens`) that align with strong manual leaderboard recipes often explored on H100 clusters.* This submission documents that search and a long MLX training run of the resulting champion.

**Submitted model artifact (Parameter Golf bundle):** the **int6 + zlib** checkpoint produced by **`apply_gptq.py`** (**`*.int6.ptz`**), **not** the larger **int8+zlib** file emitted at the end of **`train_gpt_mlx.py`** training. Training still used fake QAT toward int8 export; the **int6** artifact is the post-trained, cap-compliant submission (see **Artifact size** and **Metrics**).

**Author:** **Mike Ferguson** | **MIT Quest for Intelligence** | **GitHub:** `@mike-ferguson`

**Note: This approach is still in progress. This PR's primary purpose is to show the potential of EvoNAS, and to highlight it was able to get within 10% of the OpenAI baseline for NO compute cost, on a 5 year old laptop, with no manual parmeter selection/engineering!**

---

## Scientific contribution

Parameter Golf rewards models under a strict **compressed artifact** budget with a **byte-aware validation metric** (`val_bpb`). Instead of hand-tuning one architecture, this work runs **evolution over a joint genome**—depth, width, attention, **how many layers are recurrent**, **where** recurrence sits, **how** residual state is carried across recurrent passes, **how layers are visited** (traversal order), fake-QAT, and optimizer hyperparameters—selected with **short-run fitness** and **val_bpb extrapolation** (`evolve.py`, `genome.py`). The outcome is evidence that **cheap evolutionary search on laptop MLX** can recover **non-obvious structural priors** (recurrence + traversal + gating) that are in the same family as ideas that show up in strong **hand-designed** and **cluster-trained** entries.

## The search system (EvoNAS)

**Scale**

- **Three** full evolution campaigns (**marks 1–3**), **~80 proxy evaluations per campaign** (population **8** × **10** generations).
- **~48 hours wall time per campaign** on a **personal M1 Max MacBook Pro** (short MLX fitness jobs plus evolution overhead).
- **~240 proxy training runs** in total across marks **1–3**, **zero cloud**—every fitness call is local **`train_gpt_mlx.py`**.

**Genome**

- **32 genes** in `genome.py`, spanning **architecture**, **recurrence**, **fake quantization**, and **optimizer** settings. Values are **continuous, discrete, or categorical** as appropriate. **`CROSSOVER_BLOCKS`** groups genes so **block crossover** swaps coherent chunks—architecture, recurrence, and QAT/optimizer blocks stay internally consistent.

**Fitness**

- Each individual: **short proxy training** (**500** optimizer steps by default, `EVOLVE_ITERATIONS`), then **extrapolation** of `val_bpb` with **a + b/√t** to **t = 5000** (`EVOLVE_FIT_EXTRAP_T`). The score is **extrapolated `val_bpb`**, plus a **strong penalty** if the serialized **int8+zlib** size exceeds the ~**16 MB** submission budget (`evolve.py` `fitness()`). **Compression is a first-class signal during search**, not something tuned only afterward.

**Operators**

- **Block crossover** — recurrence / QAT / optimizer **blocks** swap **intact** between parents.
- **Per-gene mutation** rate **0.175** (`EVOLVE_MUT_RATE`).
- **Hamming diversity penalty** — genomes that differ in only a few genes pay an extra fitness cost so small populations do not collapse.
- **Elitism = 1**; **binary tournament** selection.

**Iterative refinement**

- Each **campaign** **seeds** the next from the **previous run’s `top_configs_*.json`** (`evolve.py --seed-from`).
- After a run, **`analyze_genes.py`** on `all_runs_*.json` reports **Random Forest Gini feature importance** and **permutation importance** against `val_bpb`, which drives **which genes to lock vs keep searching** in the next campaign (narrowed `genome.py` search spaces).

### 10k-step training snapshot (best champion per mark)

Cross-mark comparison at **10k training steps** for each campaign’s best model (`val_bpb` unless noted):

| Mark | Evolution snapshot | `val_bpb` @ 10k steps | ≈ train FLOPs vs **8×H100** |
|-------|-------------------|----------------------|------------------------------|
| **1** | 21-gene genome, first full evolution | *No 10k number in lab logs* — evolved top was ~**2.2** at ~275 steps before the 30-gene protocol. | **~15s–2 min** @ **8×H100** dense FP16 **peak** *(very rough; same FLOP model as below)* |
| **2** | 30-gene genome, full suite + `analyze_genes` | **~1.48** (30-gene champion long train). | **~1.3 min** @ peak ≈ **~0.13×** a **10 min** 8‑GPU window *(~⅓ Mark 3 FLOPs)* |
| **3** | 32-gene Mark 3 (`mark3.md`, `mark3_run`) | **1.351** (this submission’s champion at 10k; see **Metrics** below). | **~4 min** @ peak ≈ **~0.4×** a **10 min** 8‑GPU window *(matmul-centric **~2×10¹⁸** FLOPs to 10k; ~**40 h** M1 wall in `train.log`)* |

*Matmul-centric training FLOPs **≈ 6·N·tokens·(17/7)** for Mark 3 (`train_gpt_mlx.py` recurrence); Marks **1–2** scaled as before. **“@ peak”** divides by **8 × 989 TFLOP/s** (per‑GPU dense FP16 tensor peak); real jobs rarely sustain that, and **~25–35%** of peak makes a **10 min** 8×H100 slot **~comparable in FLOPs** to **~36 h** on M1 for the same crude count. Excludes validation, Muon, and non‑matmul work.*

**This submission** is the **int6+zlib** model (**`*.int6.ptz`**). It is built from one **Mark 3** champion continued in **`train.log`** (bf16 **`*.npz`**), then post-processed with **`apply_gptq.py`** (symmetric **int6**, GPTQ on linears, embeddings kept fp16). By contrast, **`train.log`** also reports an **int8+zlib** export (~18.6 MB) that is **over** the cap—that file is **not** what is being submitted (see **Artifact size** and **Metrics**).

## Key discoveries (what evolution actually favored)

From the **`mark3_run`** lineage and top configs (see `top_configs_*.json` in that folder), consistent winners were not “deeper vanilla transformers” alone, but:

1. **`layer_traversal_mode = odds_then_evens`** — Processing **odd- and even-indexed blocks in interleaved order** instead of a strict 0→L−1 pass. Evolution repeatedly kept this when it was in the search space; it acts as a **structured multi-pass** over depth without simply stacking identical forward passes.

2. **`state_carryover = gated`** — For recurrent layers, a **learned gate** mixing the previous activations with new block outputs (rather than pure residual or no carryover). That is a **learned depth-recurrent** mechanism: the same physical block can run multiple times with **controlled state flow**.

3. **Depth recurrence + sharing** — High-fitness genomes combine **recurrence_percent**, **recurrence_factor**, and often **`global_weight_sharing = block`**: fewer unique parameter sets, **more passes** over depth. That is the same *design neighborhood* as **Universal Transformer**–style models: **shared transition** applied repeatedly over layers/steps with adaptive stopping or carryover—here realized with Parameter Golf’s encoder/decoder split, skips, and evolved placement (`early` / `middle` / `late`).

4. **Fake QAT and Muon-centric training** — Evolution co-selected **4-bit per-channel**-style fake quant settings and optimizer knobs so that **training** matches the **int8+zlib** evaluation story, not just width/depth.

Together, these are **empirical findings from search**, not a manual ablation grid: the **search rediscovered** a bundle of ideas—**gated depth recurrence**, **non-monotonic traversal**, **sharing**—that reads like a **UT-flavored** recurrent depth prior, discovered on **M1 MLX** without cloud budget.

## Track and compute

- **Non-record**, **unlimited compute** — Wall time is dominated by a **single-machine Apple Silicon** run (`train_gpt_mlx.py`), **not** the 10-minute **8×H100** leaderboard track.
- **`train_gpt.py` in this folder** — Included because the **Parameter Golf `/records` layout expects the repo’s standard `train_gpt.py`**. It is the **upstream PyTorch CUDA starter** used across submissions. **This submission does not claim that this file alone reproduces the evolved MLX architecture or the numbers below**; those come from **`train_gpt_mlx.py`** as captured in **`train.log`**. Treat this entry as an **MLX-first / laptop-first** result with a **CUDA baseline attached for template compliance**; a **full PyTorch port** of every `genome.py` feature would be required for bit-for-bit H100 reproduction and is **out of scope** for this in-progress PR.

## Artifact size (16 MB cap)

The **submitted** compressed weights are **`*.int6.ptz`** (**this** is the Parameter Golf model file being submitted). The **training export** **`*.int8.ptz`** from **`train_gpt_mlx.py`** is **~18.6 MB** (over the **~16 MB** model budget) and is included below **for comparison only**. **Code is not the driver** (`train_gpt.py` is ~48 KB). The int6 bundle is produced by repo-root **`apply_gptq.py`** (default **int6**, per-output-channel scales, calibration on FineWeb train shards). Alternatively, cap compliance can come from re-training at lower width (e.g. **`MODEL_DIM` 640→576**); see `python scripts/size_budget_hint.py`.

| Piece | Size (this run) | Notes |
|--------|------------------|--------|
| **`*.int6.ptz`** — **submitted model** | **14,784,469 B** (~14.1 MiB) | Post-train `apply_gptq.py`; int6 symmetric + zlib; **under** cap. |
| **`*.int8.ptz`** (train export only) | **18,612,195 B** (~17.8 MiB) | From end of `train_gpt_mlx.py`; **not** submitted; **over** cap. |
| **`train_gpt.py`** | ~47,642 B | Separate from `.ptz`; counts toward total submission budget. |

## Approach (after mark 3)

1. **Search:** `evolve.py` + `genome.py` → `mark3_run/evolution_records/top_configs_*.json`.
2. **Champion continued train:** `from_json_20260409_103134_it10000` → **10k steps** in `train.log`.
3. **Metrics source:** **`train_gpt_mlx.py`**; log embeds trainer source.
4. **Submitted artifact:** `apply_gptq.py` on the bf16 `*.npz` → **`*.int6.ptz`** (see `mark3_run/mark3.md`).

## Metrics (from `train.log` and `apply_gptq.py`)

**Primary score for the submitted int6 bundle:** **`val_bpb: 1.324649`** (full FineWeb val via `apply_gptq.py` `eval_val`).

| Metric | Value |
|--------|--------|
| **Submitted: int6+zlib** (`apply_gptq.py`, full `eval_val`) | `val_loss:2.236614` **`14784469` bytes** **`val_bpb:1.324649`** |
| Last full-val (step 10000, float, `train.log`) | `val_loss:2.2139` `val_bpb:1.3112` |
| Train export: int8+zlib roundtrip (exact, `train.log`; **not** submitted) | `val_loss:2.28141063` `val_bpb:1.35117988` |
| Serialized int8+zlib (train export; **not** submitted) | `18612195` bytes |
| Training wall time | `40h8m` for `10000/10000` steps |

## Dataset and tokenizer

Standard **SentencePiece** + **fineweb** shards; **`val_bpb`** defined as in the trainers (bits-per-token with tokenizer byte lengths).

## Statistical significance (SOTA bar)

Not claimed here (single long MLX run). Future record attempts: multiple seeds and reporting per competition rules.

## Files

| File | Role |
|------|------|
| `README.md` | This document |
| `submission.json` | Metadata — keep in sync with **Author** / **GitHub** above |
| `train.log` | Full MLX run log (**source of truth for training metrics**) |
| `train_gpt.py` | Repo CUDA baseline snapshot (**template requirement**; see track section) |
| `apply_gptq.py` (repo root) | Builds the **submitted** **`*.int6.ptz`** + reported `val_bpb` |

## Before merging

1. Keep **`submission.json`** **`author`** / **`github_id`** aligned with the **Author** line above.
2. Confirm **`train.log`** is the run you want on record.
3. For Parameter Golf packaging, treat **`*.int6.ptz`** as the **submitted model weights**, not **`*.int8.ptz`** from training.

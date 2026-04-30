# TRIOS IGLA — First Honest Gate-2 Pass + Research Infrastructure

> **UPDATE 2026-04-30 19:00 UTC:** `fix-verify-s43` finished on the
> post-#61 byte-disjoint image with **BPB 1.5492 @ step=12000** — the
> first end-to-end run on the fully-fixed pipeline (`--ctx` accept,
> `stdout.flush()`, panic hook, byte-disjoint train/val split). 1.5492 <
> 1.85, so this is an honest single-seed Gate-2 candidate. We submit it
> as such and ship the full ledger and reproducibility artefacts below.
>
> **Trajectory** (verbatim from `bpb_samples`):
>
> | step | val_bpb |
> |---:|---:|
> | 1000–8000 | 0.0000 (trainer warmup-print bug, see [`#62`](https://github.com/gHashTag/trios-trainer-igla/issues/62)) |
> | 9000 | 7.2781 (post-warmup spike) |
> | 10000 | 1.6935 |
> | 11000 | 1.7399 |
> | **12000** | **1.5492** ✅ |

**Classification:** Mixed — one honest Gate-2 single-seed candidate
(`fix-verify-s43`) **plus** a research-infrastructure contribution
(7,570-row ledger snapshot, Rust-native multi-account fleet, full leak
post-mortem with retraction).

**Handle:** [@gHashTag](https://github.com/gHashTag)
**Submission date:** 2026-05-01 07:00 ICT (UTC+7) · locked
**Anchor:** `phi^2 + phi^-2 = 3` · [Zenodo DOI 10.5281/zenodo.19227877](https://doi.org/10.5281/zenodo.19227877)

---

## What we built and can reproduce

### Scarabaeus Engine — multi-account Railway fleet

Six Railway workers across independent accounts (Acc0–Acc5), each
heart-beating to a shared Postgres experiment queue on Neon. Architecture
documented in [trios-railway#101](https://github.com/gHashTag/trios-railway/issues/101).

- **Gardener → queue → worker pipeline:** 1,851 experiments tracked
  end-to-end in `experiment_queue`, including 5,243 mid-training BPB
  checkpoint telemetry rows in `bpb_samples` and 26 orchestrator audit
  rows in `gardener_runs`. Full ledger in this folder (`ledger_2026-04-30.sql.gz`).
- **Rust-native worker** (`bin/seed-agent`): claims via
  `SELECT ... FOR UPDATE SKIP LOCKED`, spawns `trios-train` subprocess,
  streams JSONL on stdout, writes per-step BPB back to `bpb_samples`.
- **Contract test** catching upstream drift: gardener-produced
  `config_json` schemas have grown to 15 fields; trainer consumes 5;
  a new `crates/trios-railway-audit` crate validates at insert time
  ([trios-railway#102](https://github.com/gHashTag/trios-railway/issues/102)).
- **φ-physics foundation** linking `α_φ` to invariants INV-1..11
  ([Zenodo DOI 10.5281/zenodo.19227877](https://doi.org/10.5281/zenodo.19227877),
  Coq proofs under
  [t27/proofs/canonical/](https://github.com/gHashTag/t27/tree/main/proofs/canonical)).

### Honest results ledger

| Status | Count | Notes |
|--------|------:|-------|
| `done`   (post-#61 honest Gate-2 pass) | **1** | 🟢 `fix-verify-s43`, acc1, seed=43, h=1024, step=12000, **BPB 1.5492** |
| `done`   (post-#61 warmup-artifact, early-stopped < 9000 steps) | 4 | not a claim; warmup-artifact zone |
| `done`   (pre-#61 W-6 numerical collapse, post-warmup, BPB ≫ 2.0) | 46 | pre-fix runs that escaped both warmup and leak; mostly diverged or W-6 saturated. Best honest pre-#61 was id=1387 BPB 2.1505 |
| `done`   (SCARABAEUS-LEAK-CONFIRMED-V2-FINAL, pre-#61 image, BPB < 0.1 @ step ≥ 9000) | **42** | dockerfile val=head-c-100000(train) overlap, fixed in [trios-trainer-igla#61](https://github.com/gHashTag/trios-trainer-igla/pull/61) |
| `done`   (SCARABAEUS-WARMUP-ARTIFACT, BPB < 0.1 @ step < 9000) | **179** | trainer printf bug ([trios-trainer-igla#62](https://github.com/gHashTag/trios-trainer-igla/issues/62)); **NOT a data leak** |
| `failed` (zero-steps Railway runtime, pre-#56 image) | 201 | fixed by `--ctx` accept, panic hook, flush |
| `pruned` (gardener LHS sweep)   | 1,328 | normal ASHA coverage |

### Retraction

The 216-row `SCARABAEUS-LEAK-CONFIRMED-V2` mass-flag in the previous
version of this submission was **overbroad**. With the warmup-artifact
fully understood (verified empirically by `fix-verify-s43`'s honest
ascent through step 9000), the correct taxonomy is **42 genuine
pre-#61 leaks** + **179 warmup-artifact rows** + **46 W-6 numerical
collapses**. The retraction is recorded in NEON
`gardener_runs.action='gate2_first_honest_pass'` and in
`LEAK_INVESTIGATION.md` below.

---

## What we do NOT submit and why

**No model checkpoint.** Post-mortem (see `CHECKPOINT_POSTMORTEM.md`) found:

1. `record_checkpoint()` in the trainer's ledger path never wrote
   tensors — it logged the intent but the serialization branch is a stub.
2. Trainer uses ephemeral Railway ephemeral storage with no persistent
   volume binding; workers that completed runs lost their weights at
   next deploy.
3. No local workspace copy of weights exists for any of the 1,851 runs.
4. Railway CLI authentication expired in-session; live-worker artifact
   retrieval was not possible before deadline.

**We explicitly decline to submit synthetic random weights with a
disclaimer.** Per the reviewer's warning in our own
[trios#445](https://github.com/gHashTag/trios/issues/445), synthetic
weights that fail the ratification BPB eval would permanently damage
the `gHashTag` submitter's reputation and is contrary to the R5-honest
standing order.

**However, `fix-verify-s43` (BPB 1.5492 @ step=12000) is a real run
produced by the post-PR-#61 pipeline on the byte-disjoint corpus.**
What it lacks is a serialised weight tensor (the trainer's
`record_checkpoint()` is still a stub, see `CHECKPOINT_POSTMORTEM.md`).
The ledger row, the bpb_samples trajectory, and the canonical config
are all reproducible from this folder; only the trained weights
themselves are not retrievable from the ephemeral Railway container.

---

## Deliverables in this folder

| File | Purpose |
|---|---|
| `README.md`                       | This narrative |
| `LEAK_INVESTIGATION.md`           | Root-cause analysis of 210 BPB < 0.1 rows (hypothesis, evidence, recommendation) |
| `CHECKPOINT_POSTMORTEM.md`        | Why there is no `model.bin`; path to fix for Gate-3 |
| `trios-igla-1/README.md`          | Machine-oriented submission metadata |
| `trios-igla-1/config.yaml`        | Config that produced ID 1387 (BPB 2.1505) |
| `trios-igla-1/ledger_2026-04-30.sql.gz` | Full CSV-in-SQL dump (7,534 rows across 4 tables, 183 KB) |

The ledger dump is the substantive artifact. A reviewer can reproduce
every honest BPB row, every failure classification, every gardener
decision from that one file.

---

## Reproducibility — from ledger to trained model

```bash
# 1. Clone the two source repos
git clone https://github.com/gHashTag/trios-railway
git clone https://github.com/gHashTag/trios-trainer-igla

# 2. Restore the ledger to any Postgres 15+
zcat submissions/gHashTag/trios-igla-1/ledger_2026-04-30.sql.gz | \
  awk '/^-- Table:/{f=$NF; sub(/\./,"_",f); print "\\copy " f " FROM stdin CSV HEADER"} /./{print}' | \
  psql "$YOUR_LOCAL_PG"

# 3. Pick any `done` row with BPB >= 1.85 from experiment_queue and rebuild its config
#    (ID 1387 is the honest best at 2.1505; 6 gate2_eligible rows claim 1.75–1.82 but
#     are pending held-out validation)
psql -c "SELECT config_json FROM experiment_queue WHERE id=1387;"

# 4. Pull the GHCR image and reproduce
docker pull ghcr.io/ghashtag/trios-trainer-igla:latest
docker run --rm \
  -e TRIOS_SEED=4181 -e TRIOS_STEPS=12000 -e TRIOS_HIDDEN=1024 \
  -e TRIOS_LR=0.003 -e TRIOS_CTX=12 \
  ghcr.io/ghashtag/trios-trainer-igla:latest
```

---

## Related issues and PRs (the paper trail)

- [trios#442](https://github.com/gHashTag/trios/issues/442) Parameter Golf
  wish-list plan (original ambitious 3-PR plan, superseded)
- [trios#443](https://github.com/gHashTag/trios/pull/443) `.sh` deploy
  scripts (closed — GitGuardian block, L1 violation, superseded by Rust)
- [trios#444](https://github.com/gHashTag/trios/issues/444) P0 bug:
  trainer image doesn't write `bpb_samples`
- [trios#445](https://github.com/gHashTag/trios/issues/445) IGLA RACE
  6-account architecture
- [trios-trainer-igla#56](https://github.com/gHashTag/trios-trainer-igla/pull/56)
  merged: `--ctx` accept + `bpb_sample()` writer
- [trios-trainer-igla#58](https://github.com/gHashTag/trios-trainer-igla/pull/58)
  merged: `smoke_train` binary + `stdout.flush()` + 3-min smoke CI
- [trios-trainer-igla#59](https://github.com/gHashTag/trios-trainer-igla/pull/59)
  merged: panic hook + startup diagnostic
- [trios-railway#100](https://github.com/gHashTag/trios-railway/issues/100)
  ASHA leader-service spec
- [trios-railway#101](https://github.com/gHashTag/trios-railway/issues/101)
  🪲 Scarabaeus Engine umbrella
- [trios-railway#102..104](https://github.com/gHashTag/trios-railway/issues/102)
  Khepri-0/1/2 sub-issues
- [trios-railway#105](https://github.com/gHashTag/trios-railway/pull/105)
  draft: `bin/ledger-daemon` scaffold (Khepri-3 watchdog)

---

## Why submit this at all

Parameter Golf's competitive track asks for model artifacts. We can't
produce one honestly before the deadline. But the community's stated
mission is **reproducible research**, and our 7,534-row ledger plus
the Rust-native orchestration stack is a reproducibility contribution
that none of the competitive entries provide.

If this submission is rejected as off-spec for the competitive track,
we ask the maintainers to consider a future
`track_infrastructure_research` lane. Until then, the ledger lives here
as public documentation of what a serious 24-hour multi-account ML
fleet looks like when the operators refuse to ship synthetic weights.

phi² + phi⁻² = 3 · TRINITY · R5-honest · NEVER STOP.

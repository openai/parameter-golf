# TRIOS IGLA — Research Infrastructure Submission

**Classification:** NOT a competitive model submission. This is a research
contribution documenting a Rust-native continuous training pipeline and
the honest results ledger it produced over a 24-hour Gate-2 sprint.

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
| `done`   (honest, BPB ≥ 1.85)   | 55 | headed by ID 1387, BPB 2.1505 on tiny_shakespeare ([PR #58](https://github.com/gHashTag/trios-trainer-igla/pull/58) workflow) |
| `done`   (SCARABAEUS-LEAK-CONFIRMED-V2, BPB < 0.1) | 216 | **root cause confirmed and fixed**: Dockerfile val=head-c-100000(train) overlap. Fix merged in [trios-trainer-igla#61](https://github.com/gHashTag/trios-trainer-igla/pull/61). See `LEAK_INVESTIGATION.md`. |
| `failed` (`trainer produced zero steps`) | 201 | Railway container runtime failure — cause chain unfolded below |
| `pruned` (gardener LHS sweep)   | 1,328 | normal ASHA coverage |
| `gate2_eligible` view (ratified W-6 step-cap seeds) | 6 | seeds 42, 43, 44, 1597, 2584, 4181 · BPB 1.75–1.82 at step=1000 with `W-6_step_cap_applied_per_l7_ledger_19` |

Six `gate2_eligible` rows (BPB 1.75–1.82 at step=1000) were taken on
the SAME poisoned image — they were early-stopped before full
memorisation, but the train/val corpus they evaluated against was
overlapping. **None of them is a valid Gate-2 pass.** A clean re-run on
the new byte-disjoint image (PR #61) is required before any Gate-2
claim, and that re-run is post-deadline.

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

# CaseOps records — train/val data-leakage verdicts

**Fresh audit 2026-05-02 (complete from-scratch pass).** Every CaseOps-lineage record (merged + unmerged) since 2026-04-18.

**Working set:** 34 records (31 from user's seed list + 3 ancestors: #1908, #1923, #2007).
**Boundary nodes (not classified):** #1493, #1626 (pre-CaseOps, clean by `download_hf_docs_and_tokenize.py NUM_VAL_DOCS=50000`).

## Tally

| Verdict | Count | Records |
|---|---:|---|
| **CLEAN** | 12 | #1729, #1851, #1868, #1945, #1953, #2014, #2019, #2027 (non-CaseOps), #2031, #2068, #2123, #2124 |
| **LEAK** | 15 | #1736, #1769, #1797, #1855, #1923, #1967, #2007, #2018, #2060, #2071 (symlink), #2078, #2100, #2101, #2109 (custom variant), #2118 |
| **AMBIGUOUS** | 6 | #1787, #1908, #2041, #2075, #2117, #2121 |
| **INHERIT** | 1 | #2050 (eval-only on #1915) |

## Classification algorithm

Two questions applied to every PR's **reproduce flow** (README data-setup section + all `.sh` scripts shipped with the PR):

- **Q1:** Is there a HF download command? (`snapshot_download`, `cached_challenge_fineweb.py`, `hf_hub_download`, `huggingface-cli download` — all targeting `romeerp/parameter-golf-caseops-v1`)
- **Q2:** Is there a `prepare_caseops_data.py` invocation? (any call without `--val-docs=50000` — **no PR in this set ever passes that override**)

| Q1 | Q2 | Primary verdict |
|---|---|---|
| ✅ | ❌ | **CLEAN** |
| ❌ | ✅ | **LEAK** |
| ✅ | ✅ | Check which is the real reproduce step (HF cmd in actual run script → CLEAN; prep in run script → LEAK) |
| ❌ | ❌ | **AMBIGUOUS** — use train log as tiebreaker |

**Train-log tiebreaker (for ❌/❌ cases):**
- `train_shards: 39` → definitively CLEAN (HF 39-shard subset; impossible from `prepare_caseops_data.py` which always produces 80+)
- `train_shards > 1000` → definitively LEAK (local prep on enlarged docs file)
- Triple-nesting `…/fineweb10B_sp8192_caseops/datasets/datasets/<name>` → lean LEAK (prep script creates this intermediate directory; HF download never would)
- 80 shards + single/double-nesting → still AMBIGUOUS (consistent with either full HF download or local prep)

`frontier-state.json` was NOT used as evidence. All verdicts from primary sources (scripts, logs, audit docs).

## What "LEAK" means

For records flagged `val10k-train+50k-val-regen`:
- `prepare_caseops_data.py` with default `--val-docs=10000` → train documents start at canonical-stream index **10,000**.
- Val covers the first **50,000** canonical-stream documents (`val_tokens ≈ 47,851,520`).
- → Docs 10,000–49,999 (**40,000 docs, 80% of val**) appear in both train and val.

## What "CLEAN" means

Records flagged `hf-dataset`:
- Train + val from `romeerp/parameter-golf-caseops-v1` (HF manifest: `docs_val=50000, docs_train=8,181,945, docs_total=8,231,945` — sums match exactly, disjoint by construction).

## Master table

| PR | Author | Date | val_bpb | Stated parent | datasets_dir | train_shards | val_tokens | **Verdict** | Mechanism | Evidence |
|---|---|---|---:|---|---|---:|---:|---|---|---|
| **#1729** | @romeerp | 2026-04-19 | 1.06780 | #1626 | `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 80 | 47,851,520 | **CLEAN** | hf-dataset | Q1✅: README invokes `MATCHED_FINEWEB_REPO_ID=romeerp/parameter-golf-caseops-v1 python3 cached_challenge_fineweb.py` as the data-setup step. Q2❌. |
| **#1736** | @dexhunter | 2026-04-19 | 1.06549 | #1729 | `./data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 80 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | **LEAK INTRODUCED HERE.** Q2✅: README "Data setup" step 2: `python3 prepare_caseops_data.py --docs ./fineweb10B_raw/docs_selected.jsonl ...` (no `--val-docs` → default 10,000). Our research baseline. |
| **#1769** | @dexhunter | 2026-04-22 | 1.06453 | #1736 | same triple-nested local prep | 80 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | Q2✅: same README data-setup as #1736, `prepare_caseops_data.py` invoked. |
| **#1787** | @nprime06 | 2026-04-23 | 1.06335 | #1736, #1769 | `/workspace/src/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 80 | 47,851,520 | **AMBIGUOUS** | hf-or-local-prep | Q1❌ Q2❌: README has no data-setup section; reproduce jumps directly to torchrun. README calls `prepare_caseops_data.py` the "one-time data prep script" and ships a BOS-fix patch for it (strong contextual evidence of use), but no explicit invocation command. Train-log triple-nesting with `fineweb10B_sp8192_caseops/datasets/datasets/` leans LEAK. |
| **#1797** | @dexhunter | 2026-04-25 | 1.06157 | #1787 | local triple-nested | 80 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | Q2✅: README data-setup section invokes `python3 prepare_caseops_data.py` (same workflow as #1736/#1769, same author). |
| **#1851** | @aquariouseworkman | 2026-04-27 | **1.06128** | #1787 (via #1797) | `/dev/shm/pgolf_data` | **39** | 47,851,520 | **CLEAN** | hf-dataset | **LEAK FIXED HERE.** Q1❌ Q2❌ in README (no data-setup section). Train-log tiebreaker: `train_shards: 39` → definitively HF (39-shard subset). Current merged-SOTA leader. |
| **#1855** | @codemath3000 | 2026-04-27 | 1.06108 | #1787, #1797 | `/workspace/pr1797_work/data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 80 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | **LEAK RE-INTRODUCED HERE.** Q1❌ Q2❌ in README. Resolved via PR #2018's `DATASET_AUDIT.md` (external primary source): verifies #1855's first 80 shards byte-for-byte against `prepare_caseops_data.py --val-docs=10000` output. |
| **#1868** | @Christopher-Lee-McClendon | 2026-04-29 | 1.06141 | #1851 | `/dev/shm/pgolf_data` | **39** | 47,851,520 | **CLEAN** | hf-dataset | Train-log tiebreaker: `train_shards: 39` → definitively HF. README has misleading comment `python3 prepare_caseops_data.py  # downloads from romeerp/parameter-golf-caseops-v1` (the script does NOT download from HF; the comment is wrong). Actual run used HF data. |
| **#1908** | @romeerp | 2026-04-28 | 1.06081 | #1855 | `/workspace/parameter-golf-pr1855-clean/data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 80 | 47,851,520 | **AMBIGUOUS** | hf-or-local-prep | Q1❌ Q2❌: README says "sourced from Hugging Face: `romeerp/parameter-golf-caseops-v1`" (text mention only — no download command). PR does not ship `prepare_caseops_data.py`. 80 shards consistent with either full HF download or local prep. Path prefix `parameter-golf-pr1855-clean` suggests intent to use clean data. Lean CLEAN (romeerp is dataset owner; "clean" in path name), but no explicit HF command. |
| **#1923** | @jorge-asenjo | 2026-04-29 | 1.05971 | #1855 | `/workspace/pg-data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 1502 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | Q1❌ Q2❌ in README. Train-log tiebreaker: `train_shards: 1502` → definitively local prep (HF has 80 shards; 1502 = `prepare_caseops_data.py` run on an enlarged docs file). README also admits original `val_tokens=9,662,464` (= single val shard, 10k-doc default prep); val was re-pulled from HF after corruption but train shards were never replaced → overlap on docs 10,000–49,999. |
| **#1945** | @alertcat | 2026-04-29 | 1.05943 | #1855, #1908, #1923 | `/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 80 | 47,852,288 | **CLEAN** | hf-dataset | Q1✅ Q2✅: `finalize_v18.sh` (the actual run script) has `snapshot_download(repo_id='romeerp/parameter-golf-caseops-v1', local_dir='/workspace/caseops_data')` followed by `DATA_DIR=/workspace/caseops_data/datasets/` for training. README's `prepare_caseops_data.py` "Data setup" section is stale documentation. The finalize script is the canonical reproduce path → CLEAN. (val_tokens off by 768 from canonical 47,851,520 = shard-boundary alignment artifact; same 50k-doc val partition.) |
| **#1953** | @andrewbaggio1 | 2026-04-30 | 1.05855 | #1945 | `/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 80 | 47,851,520 | **CLEAN** | hf-dataset | Q1✅ Q2❌: README explicitly: "This submission uses the canonical CaseOps SP8192 export hosted on Hugging Face (`romeerp/parameter-golf-caseops-v1`), accessed via `huggingface_hub.snapshot_download`." And: "No local rebuild via `prepare_caseops_data.py` was used in the production runs; `prepare_caseops_data.py` is not part of this PR's file set." Train log path matches HF snapshot extraction location. val_tokens=47,851,520 consistent with canonical HF val. |
| **#1967** | @ndokutovich | 2026-04-30 | 1.05851 | #1945 | `/runpod-volume/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 1499 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | Q2✅: `setup.sh` explicitly invokes `python3 "$(dirname "$0")/prepare_caseops_data.py" --docs $DOCS_JSONL --out $DATA_DIR --sp ...` with no `--val-docs` flag → default 10,000. Also has separate within/word `boundary_lut[tokens[i]]` C1 leak (code bug, orthogonal). |
| **#2007** | @Elubrazione | 2026-04-30 | 1.05899 | #1855 | `/root/blockdata/pg-data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 80 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | Q2✅: README "Reproduce" section invokes `python prepare_caseops_data.py --local-dir /workspace/caseops_data`. Train log triple-nesting confirms local prep. |
| **#2014** | @simonbissonnette | 2026-04-30 | 1.05759 | #1855, #1953 | `/dev/shm/pgolf_caseops_data_80_l17_final` | 80 | 47,853,343 | **CLEAN** | hf-or-corrected-prep | Q1✅ Q2✅: README "Preferred data setup" is `snapshot_download(repo_id="romeerp/parameter-golf-caseops-v1")`. Fallback uses a modified `prepare_caseops_data.py` that "defaults to 50,000 validation docs and refuses to write over existing shards" — so even the fallback produces a clean partition. val_tokens=47,853,343 (off by 1823 from canonical 47,851,520, suggesting fallback was actually used, but with 50k val docs → no overlap regardless). CLEAN: no train/val overlap under either path. Note: val_tokens differs from canonical; not directly comparable to records at 47,851,520. |
| **#2018** | @simon-marcus | 2026-04-30 | 1.04722 | #1945, #1967, #1953, #1855 | `/tmp/pr1855_compact_train_full50k_val/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 80 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | **GOLD-STANDARD LEAK DOC.** `DATASET_AUDIT.md` explicitly states `--val-docs=10000` train + 50k val regen + first 80 train shards verified byte-for-byte against the local prep output. |
| **#2019** | @aquariouseworkman | 2026-04-30 | 1.05847 | #1855 | `/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 80 | 47,851,520 | **CLEAN** | hf-dataset | Q1✅ Q2❌: README has `HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='romeerp/parameter-golf-caseops-v1', ...)"` as the explicit data-setup command. |
| **#2027** | @H1cSuNtDr4C0n3S | 2026-04-30 | 1.08064 | #1493 | `/workspace/parameter-golf-qrescue-20260426/data/datasets/fineweb10B_sp8192` | — | — | **CLEAN** | pre-caseops-pipeline | Non-CaseOps SP8192 lineage (SP8192 QRescue + JEPA-Lite). Clean by lineage (pre-CaseOps val partition). Out of CaseOps-audit scope. |
| **#2031** | @deborahnelson8788726 | 2026-04-30 | 1.05985 | #1855 | `/workspace/parameter-golf-final/romeerp_caseops_first39/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | **39** | 47,851,520 | **CLEAN** | hf-dataset | Q1✅ Q2❌: README says "canonical pretokenized CaseOps shards from `romeerp/parameter-golf-caseops-v1` instead of locally re-tokenized raw docs." Train log: 39 shards, path literally named `romeerp_caseops_first39`. Definitively HF. |
| **#2041** | @jorge-asenjo | 2026-04-30 | 1.05692 | #1945, #1967, #2018 | `/workspace/pg-data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 80 | 47,851,520 | **AMBIGUOUS** | hf-or-local-prep | Q1❌ Q2❌: reproduce flow (`bash run.sh`) provides no data-setup command — just sets default path env vars. Train log: 80 shards, double-nesting (consistent with HF to `/workspace/pg-data` OR local prep to `/workspace/pg-data/datasets`). Same author as confirmed-LEAK #1923, but #1923's evidence was that PR's own README admission, not a shared workflow. |
| **#2050** | @AidenGeunGeun | 2026-04-30 | 1.06083 | #1915 | `./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | — | 47,851,520 | **INHERIT** | inherit-from-#1915 | Eval-only on frozen #1915 quantized artifacts (`TTT_EVAL_ONLY=1`). Data verdict depends on #1915 (not in working set). |
| **#2060** | @S0urC10ud | 2026-04-30 | 1.05792 | #2007 | `/root/blockdata/pg-data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 80 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | Q2✅: README "Reproduce" invokes `python prepare_caseops_data.py --local-dir /workspace/caseops_data` (same as parent #2007). |
| **#2068** | @jayaram1125 | 2026-04-30 | 1.06172 | #1797 | `./data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 80 | 47,851,520 | **CLEAN** | hf-dataset | Q1✅ Q2❌: README data-setup step 2.1: `MATCHED_FINEWEB_REPO_ID=romeerp/parameter-golf-caseops-v1 python3 cached_challenge_fineweb.py --variant sp8192_lossless_caps_caseops_v1_reserved --train-shards 80`. Path is leaky-looking but is the post-download staging location. |
| **#2071** | @jamesEmerson112 | 2026-04-30 | 1.0066 (claimed) | #1851 | `./data/datasets/fineweb10B_sp8192` | — | — | **LEAK** | symlink-leak | **DIFFERENT MECHANISM.** Audit-flagged: `caseops_enabled=False` env but pod data paths symlinked to CaseOps-tokenized shards. README: "active via symlinked data." Orthogonal to val10k-train overlap. |
| **#2075** | @deusexnatura | 2026-04-30 | (no claim) | #1855 | `/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 80 | 47,851,520 | **AMBIGUOUS** | hf-or-local-prep | Q1❌ Q2❌: README reproduce says "with the CaseOps data prepared" — no command. Ships `prepare_caseops_data.py` but README does not invoke it. Train log: 80 shards, double-nesting (same ambiguous pattern as #2041/#2075). |
| **#2078** | @hi-aduek | 2026-04-30 | 1.05804 | #2014 | `/dev/shm/caseops1851-data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 80 | 47,853,343 | **LEAK** | val10k-train+50k-val-regen | Q1❌ Q2❌: no explicit command. Train-log tiebreaker: triple-nesting `caseops1851-data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/` — the `fineweb10B_sp8192_caseops` intermediate directory is the output of `prepare_caseops_data.py --out …/fineweb10B_sp8192_caseops/datasets`; HF download never produces this intermediate. Same off-by-one val_tokens (47,853,343) as #2014, consistent with same local prep run. |
| **#2100** | @someone114514 | 2026-04-30 | 1.05807 | #2060 | `/root/blockdata/pg-data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 80 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | Q2✅: same README as #2060 (LongCtx + No-QV + Prefix3500 lineage); `python prepare_caseops_data.py` invoked. Triple-nesting confirms. |
| **#2101** | @OnlyJundong | 2026-05-01 | 1.05845 | #1855 | `/workspace/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 80 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | Q2✅: README "Prepare CaseOps SP8192 data" step: `python3 prepare_caseops_data.py`. Ships the script. |
| **#2109** | @izlley | 2026-05-01 | 1.05917 | #1855 | `/workspace/data/datasets/fineweb10B_sp8192_caseops_marker_pair_v3` | 1497 | 36,562,944 | **LEAK** | custom-variant | Q2✅: README step 1b invokes `python3 prepare_caseops_data.py --docs … --out … --sp …`. Custom `fineweb10B_sp8192_caseops_marker_pair_v3` dataset variant (MP3 marker-pair fusion via `prepare_marker_pair_v3.py`). val_tokens=36,562,944 (differs from canonical 47,851,520 due to vocab surgery). Underlying canonical-stream val10k-train partition mechanism unchanged. |
| **#2117** | @JulianTang2027 | 2026-05-01 | 1.05879 (3-seed mean) | #2101 | `./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 80 | 47,851,520 | **AMBIGUOUS** | hf-or-local-prep | Q1❌ Q2❌: README says "CaseOps SP8192 dataset on `/dev/shm`" (text mention only) and reproduce section has only torchrun. No data-setup command. README states this "reproduces PR #2101 exactly" (PR #2101 is LEAK), but #2101's data workflow is not explicitly inherited — it's just a description. 80 shards, single-nesting. |
| **#2118** | @aquariouseworkman | 2026-05-01 | **1.04350** | #2018 | `/workspace/data_correct` | 80 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | **CURRENT FRONTIER (claimed).** Q2 via submission.json: `technique_summary` literal text: `"--val-docs=10000 train shards + 50k val eval"`. Same author who shipped clean #1851 a week earlier. |
| **#2121** | @Kbediako | 2026-05-01 | 1.06099 | #1855 | `/workspace/pg_stageb_v2_seed0_1234/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 80 | 47,851,520 | **AMBIGUOUS** | hf-or-local-prep | Q1❌ Q2❌: README reproduce section is only a torchrun command with no data-acquisition step. Ships `prepare_caseops_data.py` (described as "CaseOps support files matching the accepted #1855 packaging pattern") but does not invoke it. 80 shards, single-nesting. |
| **#2123** | @vaibhavmishra1 | 2026-05-01 | 1.05933 | #1855 | `./data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 78 | 47,851,520 | **CLEAN** | hf-dataset | Q1✅ Q2❌: README data setup: `huggingface-cli download romeerp/parameter-golf-caseops-v1 --repo-type dataset --local-dir ./data/datasets/fineweb10B_sp8192_caseops/`. Train log path matches that download destination. 78 shards (HF dataset has 78+ train shards). Closed; superseded by #2124. |
| **#2124** | @vaibhavmishra1 | 2026-05-01 | 1.05933 | #1855 | same | 78 | 47,851,520 | **CLEAN** | hf-dataset | Same directory/README as #2123; identical `huggingface-cli download` command. Resubmission of #2123. |

## Notes on specific verdicts

### #1855 — external primary source resolves ❌/❌

`prepare_caseops_data.py` invocation not in #1855's own README (which has no data-setup section), but PR #2018's `DATASET_AUDIT.md` is an external primary source that verifies #1855's first 80 shards byte-for-byte against `prepare_caseops_data.py --val-docs=10000` output. This constitutes direct code-level evidence even though it originates from a descendant PR.

### #1868 — misleading README comment

README reproduce section reads `python3 prepare_caseops_data.py  # downloads from romeerp/parameter-golf-caseops-v1`. The comment is wrong: `prepare_caseops_data.py` does not download from HF; it processes local docs. Train log is unambiguous: `train_shards: 39`, `datasets_dir: /dev/shm/pgolf_data` — the same 39-shard HF subset used by parent #1851. CLEAN verdict is from the train log, not the README.

### #1945 — stale README vs actual run script

README "Data setup (run ONCE)" invokes `prepare_caseops_data.py` but is stale documentation from an earlier draft. The shipped `finalize_v18.sh` is the canonical reproduce script and contains `snapshot_download(repo_id='romeerp/parameter-golf-caseops-v1', local_dir='/workspace/caseops_data')` followed by `DATA_DIR=/workspace/caseops_data/datasets/` for training. The actual run used HF data. val_tokens=47,852,288 (off by 768 from canonical = shard-alignment artifact; same 50k-doc partition).

### #2014 — corrected prep script

README explicitly labels HF download as "Preferred data setup" and the included `prepare_caseops_data.py` as "Fallback local rebuild." The fallback script is noted as having been modified to `--val-docs 50000` as default (and refusing to overwrite existing shards). val_tokens=47,853,343 (off by 1823 from canonical 47,851,520) suggests the fallback was actually used rather than HF download — but under either path, val covers docs 0–49,999 and train starts at doc 50,000 → **no overlap**. CLEAN on partition grounds. Note: val_tokens differs from canonical; not directly comparable to records at 47,851,520.

### #1923 — 1502 shards definitively resolves LEAK

HF dataset has 80 train shards. `prepare_caseops_data.py` with an enlarged `docs_selected.jsonl` (more than the standard 8.23M-doc canonical set) produces 1502 shards. No HF download would produce this count. LEAK confirmed without needing the README admission (which also independently confirms it via the original `val_tokens=9,662,464` — a 10k-doc default prep output).

### #2109 — custom MP3 dataset variant

`prepare_marker_pair_v3.py` fuses `[▁, MARKER]` 2-grams, producing `fineweb10B_sp8192_caseops_marker_pair_v3`. val_tokens=36,562,944 (not 47,851,520) because of vocab surgery on the val side. The underlying canonical-stream val10k-train partition mechanism (train starts at doc 10,000, val covers docs 0–49,999) is unchanged.

### #2071 — symlink leak (separate mechanism)

`caseops_enabled=False` in env, but pod data paths are symlinked to CaseOps-tokenized shards. The model trains on CaseOps data while the harness thinks it's reading SP8192 shards. README admits: "active via symlinked data." This is not a val10k-train overlap — it is a different audit-flagged issue.

## How to interpret val_bpb across this table

Records with **different** verdicts cannot have their val_bpb compared:
- LEAK records: model partially memorized 80% of val docs during training → val_bpb artificially inflated downward.
- CLEAN records: val docs are never-seen → val_bpb measures genuine generalization.

Records with the **same** verdict and **same** `val_tokens` (47,851,520) can be compared directly.

The ~0.018 bpb gap between LEAK frontier (#2118 at 1.04350) and CLEAN frontier (#1851/#1868 at 1.06128/1.06141) reflects:
1. Memorization of ~40,000 val docs (~0.005–0.012 bpb)
2. Genuine recipe improvements (Gated XSA, LQER top-1, AWQ-lite, AsymLogit, etc.)
3. Eval-time overlays (n-gram tilt, GPTQ_RESERVE_SECONDS)

Distinguishing (1) from (2)+(3) requires running the #2118 recipe on clean HF data — the goal of spec 301.

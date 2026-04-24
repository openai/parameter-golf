# BPB Byte-Count Audit Tool — Non-record submission

Tooling + methodology contribution systematizing the `build_sentencepiece_luts` bug disclosed in [@yahya010's PR #1734 closure](https://github.com/openai/parameter-golf/pull/1734) (2026-04-19).

## TL;DR
- Static tool that detects three byte-count bug variants in any `train_gpt.py` without running the model.
- Applied to top-10 open PRs on 2026-04-23: 6 CORRECT, 4 OBFUSCATED, 0 BUGGY.
- Frontier of verified correct-LUT PRs: #1735 (AjAnubolu, 1.04290).
- Full audit, tests, and tool in this folder. Live development in <https://github.com/abi2024/agent-pgolf>.

## Run the tool

```bash
python canonical_rescore.py \
    --train-script <path/to/train_gpt.py> \
    --tokenizer <path/to/sp.model> \
    --val-data '<glob>/fineweb_val_*.bin' \
    --pr-number <N> \
    --reported-bpb <X.XXXXX>
```

Output is JSON with `lut_status` (CORRECT/BUGGY/OBFUSCATED/UNKNOWN), `lut_bug_detections` (list of deviations), `inflation_ratio`, `inferred_canonical_bpb`, and `passes_merged_sota_threshold`.

## Tests

```bash
python -m pytest tests/ -q   # 20 tests; 3 skip gracefully if PR #1727's canonical train_gpt.py is not present locally
```
To run all 20 tests including the canonical-file tests, fetch PR #1727's canonical `train_gpt.py` first:

```bash
git fetch upstream pull/1727/head:pr-1727
git checkout pr-1727 -- records/track_10min_16mb/2026-04-18_SP8192_MPSGD_QKGain525/train_gpt.py
```

Then return to this branch (`git checkout audit-1698-lineage-bpb-bytecount`) and re-run pytest.


## Full writeup
See `writeup.md` for the full PR body, `methodology.md` for canonical BPB derivation and the three-bug classifier, `results.md` for per-PR inspection notes, `corrected_leaderboard.md` for the summary table.

## Scope
- Detects three known LUT bug patterns; cannot verify eval loop, model artifact, or arbitrary other measurement irregularities.
- Cannot verify obfuscated (`lzma+b85decode`) scripts; flagged as OBFUSCATED.
- "CORRECT" means LUT is canonical on all three tested properties — necessary but not sufficient for full submission validity.

# Experiment 058: PR135 base + our improvements

## Status: PLANNING

## Candidate improvements to try on PR135's base:
1. **NorMuon** — helped slightly for relu² in our 053 vs 052 comparison
2. **int6 QAT** — STE fake quantization during training, reduces quant gap
3. **seq4096 instead of seq2048** — our 054 (seq4096) tied with 055 (seq2048) on sliding
4. **Larger BigramHash** — try 8192 buckets instead of 4096
5. **Different batch size** — 393K vs 786K

## Priority: Fix artifact compression first
The 2.6MB artifact gap must be resolved before any improvements matter.
Need to investigate why zstd produces different sizes.

## Results
*Pending*

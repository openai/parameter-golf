# Autoresearch Results Log

## Prior Baselines (from research_handoff)
| Config | Seed | val_bpb | Compressed | Log |
|--------|------|---------|------------|-----|
| vanilla baseline | 1337 | 2.16835 | 8,558,684 | stage6_seq_base_180s_s1337.txt |
| heavycycle3 | 1337 | 2.13989 | 3,390,661 | stage8_heavycycle3_180s_s1337.txt |
| hc3+wd0.01 | 1337 | 2.13541 | 3,406,265 | stage10_heavycycle3_wd001_180s_s1337.txt |
| hc3+wd+smear(-3) | 1337 | 2.12787 | 3,386,835 | stage13_heavycycle3_wd001_smearm3_180s_s1337.txt |
| **hc3+wd+smear+mod** | **1337** | **2.02522** | **3,658,495** | stage16_..._mod075_180s_s1337.txt |
| **hc3+wd+smear+mod** | **2026** | **2.02095** | **3,665,283** | stage16_..._mod075_180s_s2026.txt |

## Dead Ideas (do NOT re-test)
- full AttnRes / most selective AttnRes variants
- naive latent memory / recurrent memory
- naive shared-depth recurrence (before heavy-only split)
- naive width scaling
- sequence curriculum as main lever
- MLP FiLM / MLP LoRA / low-rank delta
- MTP (multi-token prediction)
- MLP 3x expansion
- aggressive KV reduction as main idea
- PTQ-only mixed low-bit export
- SMEARGATE_INIT=-2.0 (gate too aggressive, val_bpb=2.149)
- MOD_CORE=0 (val_bpb=2.073, also much slower)
- MOD_CORE=2 (val_bpb=2.031, worse than core=1)
- uniform KV_HEAD_SCHEDULE=2 (val_bpb=2.150)
- non-uniform KV_HEAD_SCHEDULE=4,2,4 (val_bpb=2.131)

## Autoresearch Experiments
| # | ID | Description | Seed | val_bpb | Compressed | Delta | Status |
|---|-----|-------------|------|---------|------------|-------|--------|

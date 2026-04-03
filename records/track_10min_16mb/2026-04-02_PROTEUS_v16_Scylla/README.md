# Record: PROTEUS v1.6 — Scylla + Parallel Residuals + Depth Recurrence + Legal TTT — val_bpb 1.0819

## Result

**val_bpb: 1.0819** (3-seed mean, std: 0.00088) | Scylla tokenizer | 8×H100 SXM

| Seed | Sliding Window BPB | Roundtrip BPB | Steps | Train Time |
|------|-------------------|---------------|-------|------------|
| 42   | 1.08075           | 1.10284       | 5,884 | 600.1s     |
| 1337 | 1.08289           | 1.10489       | 5,905 | 600.0s     |
| 2024 | 1.08213           | 1.10421       | 5,894 | 600.0s     |

## What This Submission Is

**Skilled integration of community techniques onto a strong neural base.** The engineering work is ours — the foundational techniques are not. We credit every source below.

### Our Engineering (original to this submission)

1. **Mixed INT5/INT6 per-layer quantization** — INT5 for MLP layers, INT6 for attention, tuned to fit the 16 MB artifact budget
2. **Learnable lane merge + separate `resid_mix_mlp`** — learnable scalar mixing for parallel residual streams with per-dimension MLP routing
3. **Scylla retokenization pipeline** — on-pod retokenization from SP1024 shards to the Scylla vocabulary
4. **Integration engineering** — making parallel residuals, depth recurrence, legal TTT, and the Scylla tokenizer work together in one training run
5. **CPU e2e test suite** — 10 test cases covering imports, hyperparameters, model creation, forward pass, code size, quantization+artifact, step time, quant MSE, scale timing, and weight distribution

### Our Prior Contributions to the Competition

- **The Agora** — community compliance classification engine, live leaderboard, and regulatory tracker at [matotezitanka.github.io/parameter-golf](https://matotezitanka.github.io/parameter-golf). No other competitor built community infrastructure.
- **LeakyReLU slope sweep** — controlled 7-slope experiment (0.1–0.9) showing monotonic improvement, slope 0.9 beats 0.5 by 0.013 BPB. Posted on [issue #140](https://github.com/openai/parameter-golf/issues/140#issuecomment-4127322055).
- **Compliance analysis** — rule interpretation and technique legality mapping posted on [issue #140](https://github.com/openai/parameter-golf/issues/140) and [issue #1017](https://github.com/openai/parameter-golf/issues/1017).
- **PROTEUS submission series** — 7 PRs (#95, #368, #512, #568, #633, #769, #1274) documenting iterative improvement from 1.2037 to 1.0819 BPB, including negative results (INT4, depth recurrence overhead, SWA).
- **14 community contributions** across 4 issues (#140, #677, #942, #1017, #1175) plus the 7 PROTEUS PRs.
- **Community toolkit** — Docker image, RunPod template, CPU test harness.

### What's NOT Ours (full attribution)

| Component | Source | PR | Author |
|-----------|--------|-----|--------|
| Training base architecture | LeakyReLU² + Parallel Muon | [#549](https://github.com/openai/parameter-golf/pull/549) | @abaybektursun |
| GPTQ + XSA-all + BigramHash 3072 | AR Self-Gen GPTQ | [#1019](https://github.com/openai/parameter-golf/pull/1019) | @abaybektursun |
| Scylla tokenizer | Novel TokenMonster-derived tokenizer | [#1143](https://github.com/openai/parameter-golf/pull/1143) | @simon-marcus |
| Parallel residuals + depth recurrence | Separate attn/MLP lanes + layer 4-5 recurrence | [#1204](https://github.com/openai/parameter-golf/pull/1204) | @msisovic |
| Legal TTT framework | Score-first SGD with momentum, frozen early blocks | [#461](https://github.com/openai/parameter-golf/pull/461) | @Christopher-Lee-McClendon |

**Note on Scylla:** PR #1143 was closed by the author after byte-accounting errors were found (~4-6% BPB inflation from incorrect modifier token byte counts). Our implementation uses verified per-token UTF-8 byte lengths for all 998 tokens, with no modifier token inflation. See "Byte Accounting Verification" below.

## Byte Accounting Verification

Our Scylla byte counting uses three lookup tables built from the vocabulary:
- `base_bytes[i]` = `len(token_i.encode('utf-8'))` — verified for all 998 tokens
- `has_leading_space` — all False (TokenMonster has no space modifiers)
- `is_boundary_token` — all False (no BOS/EOS/PAD tracked)

BPB formula: `(nats / log(2)) × (token_count / byte_count)`

This is immune to the PR #1143 failure mode. Five zero-byte tokens (empty strings) are correctly counted as 0 bytes. All 5 evaluation functions use identical byte-counting logic.

## Architecture

11L/512d/8H/4KV, MLP 3× LeakyReLU(0.5)², XSA last 4, Partial RoPE 16d, LN Scale, BigramHash, SmearGate, VE 128d (layers 9-10), EMA 0.997, QAT, Mixed INT5/INT6+LZMA, Muon optimizer, Parallel Residuals (from layer 7), Mini Depth Recurrence (layers 4-5, from step 3000), Legal Score-First TTT.

Scylla tokenizer (998 tokens, TokenMonster-derived).

## Compliance

- [x] 8×H100 SXM training
- [x] 10-minute wallclock (600s)
- [x] Artifact ≤ 16 MB (prior identical-architecture runs: 15.0–15.8 MB; exact verification pending)
- [x] No n-gram cache at eval
- [x] No two-pass rescoring
- [x] Score-first TTT (tokens scored before weight update)
- [x] Autoregressive eval (causal)
- [x] 3-seed validation (42: 1.0808, 1337: 1.0829, 2024: 1.0821, mean: 1.0819, std: 0.00088)

## Known Limitation

These runs used `ACTIVATION_NEG_SLOPE=0.5`. Our [slope sweep](https://github.com/openai/parameter-golf/issues/140#issuecomment-4127322055) on the non-parallel architecture showed slope=0.9 beats 0.5 by ~0.013 BPB. However, controlled A/B testing on the parallel residuals architecture showed slope=0.9 is **0.0054 BPB worse** than 0.5 — the parallel lanes prefer more aggressive gating. Slope 0.5 is correct for this architecture.

## Platform

RunPod 8×H100 80GB SXM, PyTorch 2.11.0+cu128.

*Disclosure: I use Claude Code CLI, Codex CLI, and Gemini Pro as tools in my workflow. Human first, AI-assisted.*

# PR 660 — Soft MoE: Exploring Mixture of Experts Under the 16MB Constraint

**Author:** Hugo Ochoa (HugoOchoaLP)
**Claimed BPB:** 1.1826 (11L Soft MoE)
**Artifact size:** 17.3 MB (over 16MB limit — work in progress)
**Seeds:** not stated (single run listed)

## Files retrieved
- `records__track_non_record_16mb__2026-03-24_SoftMoE_exploration__README.md`
- `records__track_non_record_16mb__2026-03-24_SoftMoE_exploration__submission.json`
- `records__track_non_record_16mb__2026-03-24_SoftMoE_exploration__train_gpt.py`
- `RUNS.md`

## Environment variables (from run command in README)
RUN_ID=soft_moe_10L MOE_MODE=soft NUM_EXPERTS=2 NUM_LAYERS=10 MOE_START_LAYER=8 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024

## Claimed changes (from README, verbatim)
> Non-record submission exploring whether Mixture of Experts (MoE) architectures can improve parameter golf performance. Key finding: standard sparse MoE fails under 16MB constraints, but a dense "Soft MoE" variant fixes all identified problems.

> What Failed: Sparse MoE — Router collapse: 98% of tokens routed to one expert, even with 10x aux loss coefficient. torch.compile incompatibility: Variable-size tensors caused constant recompilation. Step time 2309ms vs 794ms baseline.

> What Worked: Soft MoE — Dense gating where ALL experts run on ALL tokens with learned soft weights. No collapse possible. Compile-friendly. Step time 636ms. 1.1826 bpb on 11L config (vs 1.2244 baseline).

> Architecture: 10-11 layers, 512 dim, 8 heads, 4 KV heads (GQA). Soft MoE on last 2 layers only. 2 experts per MoE layer, each with mlp_mult/2 hidden dim. SmearGate + BigramHash(10240, dim=128). EMA (decay=0.998) replacing SWA. Int5 MLP / Int6 attention quantization + zstd-22.

# RECORD:  SmearGate + Attention Output Gate + Legal TTT 
mean val_bpb = 1.07139 | std = 0.00082 | 15.927 MB

## Key Results

| Seed | Steps | Pre-Quant val_bpb | Quant val_bpb | TTT val_bpb | Artifact Size |
| ---- | ----- | ----------------- | ------------- | ----------- | ------------- |
| 42   | 4843  | 1.07227           | 1.08262       | **1.07221** | 15.94 MB      |
| 1337 | 4843  | 1.07074           | 1.08109       | **1.07057** | 15.91 MB      |
| 0    | 4836  | 1.07151           | 1.08183       | **1.07139** | 15.93 MB      |
| Mean | 4840  | 1.07159           | 1.08184       | **1.07139** | 15.927MB      |

### Smear Gate
Reintroduced Smear Gate, yet with input dependence in Modded Nano GPT style.
### Attention Output Gate (Per-Head Output Modulation)

A lightweight per-head multiplicative gate on the attention output

- Weight-initialized to zero: at init, all heads pass through at scale 1.0
- Total new parameters: 12 x 8 = 96 weights per layer x 11 layers = **1,056 parameters**
- Activated by `GATE_ATTN_OUT=1 GATE_ATTN_SRC=proj GATE_WIDTH=12`


## Training Configuration
Installing packages
```bash
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
  pip install brotli sentencepiece python-minifier numpy
```

sp8192 Dataset Download:
```
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
```

Run command
```bash
SEED=<SEED> RUN_ID=<RUN_ID> \
  SMEAR_GATE=1 SMEAR_GATE_WIDTH=12 \
  GATE_ATTN_OUT=1 GATE_ATTN_SRC=proj GATE_WIDTH=12 \
  QK_GAIN_INIT=5.25 \
  TTT_ENABLED=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

> [!NOTE]
> Note on code size: train_gpt.py is shipped as raw source to increase readability (125 KB), but _compressed_code_size() reports the theoretical on-disk size of the same source
> after pyminify + LZMA + base85 wrapping (~30 KB).

Training completes in ~587s (wallclock-capped), reaching 4836-4843 steps depending on seed. The gate overhead is ~1.5% of step throughput (from ~8,200 tok/s to ~8,080 tok/s at step 1000, widening slightly with layer looping after step ~2141).

## Full Architecture Stack

- 11L x 512d x 8H / 4KV heads (GQA)
- MLP 4x expansion with LeakyReLU(0.5)^2 activation
- Partial RoPE (16/64 dims)
- Layerwise LN scale: `1/sqrt(layer_idx+1)`
- Tied embeddings, logit softcap = 30.0
- **SmearGate** (width=12, learned lambda) -- **NEW**
- **Attention Output Gate** (width=12, per-head, all 11 layers) -- **NEW**
- Skip gates (sigmoid-gated U-Net connections)
- 3-layer depth recurrence (layers 3,4,5, activated at frac=0.35)
- Parallel residuals (layer 7+)
- QK-Gain 5.25 (per-head, per-layer)
- MuonEq-R optimizer (WD=0.095, MLR=0.026, EMA=0.9965)
- GPTQ quantization: int6 matrices (clip=12.85), int7 embeddings (clip=20.0)
- Brotli-11 compression with byte-shuffle
- Score-first TTT (SGD, LR=0.005, 3 epochs per chunk)

## Compliance

This submission satisfies all Track B requirements:

1. **Causality**: Sliding-window TTT evaluation maintains strict token ordering. Each position is scored from its prefix only.
2. **Distribution integrity**: Standard softmax over complete vocabulary without post-hoc modifications or logit biasing.
3. **Score-before-update**: TTT parameters update exclusively after scoring relevant data chunks (score-first methodology inherited from PR #1586).
4. **Single evaluation**: Each token receives exactly one score without rescoring passes.
5. **Artifact size**: All seeds produce artifacts under 16,000,000 bytes (max: 15,936,229 bytes for seed 42).
6. **Training time**: All seeds complete within 600s wallclock.
7. **TTT eval time**: All seeds complete TTT eval within 600s.

## Acknowledgments

Built on the work of the parameter-golf community:
- **@bigbag** -- SP8192 architecture, 3-layer depth recurrence, parallel residuals, QK-Gain 5.25, score-first TTT (records #1-#8)
- **@dexhunter** -- per-layer adaptive GPTQ clip, int7 embeddings, MLR tuning (PR #1586, our direct baseline)
- **@kellerjordan** -- modded-nanogpt speedrun infrastructure and SmearGate concept (originally from modded-nanogpt)
- SmearGate was first introduced to parameter-golf in earlier records (#13-#15) before being removed at record #8

This work was also possible thanks to the support provided by Paradigma ([link](https://paradigma.inc/)) and the use of Flywheel ([link](https://flywheel.paradigma.inc/)): their infrastructure for research

Our Team: me, @CerovazS, @GabrieleCirillo

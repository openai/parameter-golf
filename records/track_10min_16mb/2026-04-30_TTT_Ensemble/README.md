# Record: TTT Peer-LoRA Ensemble

**val_bpb = TBD** (1 seed) | **~15.99 MB** | 8xH100 SXM | PyTorch 2.10.0+cu130

This record introduces peer-LoRA ensembling into the test-time training (TTT) evaluation loop. After each batch's per-doc LoRAs are fully trained, we run k-1 additional forwards using *other* docs' trained LoRAs from the same batch. This is leakage-free: LoRA_p was trained only on doc_p's tokens, so applying it to doc_q reveals no target information. On uncertain tokens (high predictive entropy), we blend own and peer predictions in probability space; confident tokens use only their own prediction. The routing decision is target-free -- it depends only on the model's output distribution, not on validation labels.

Built on [PR #2014](https://github.com/openai/parameter-golf/pull/2014), descending from @samacqua's work on doc-independent LoRAs.

## Results

| Seed | Pre-Quant BPB | Post-Quant BPB | **Post-TTT BPB** | Artifact |
|------|--------------:|---------------:|-----------------:|---------:|
| 314  | -   | -  | - | - |

Baseline PR #2014 3-seed mean: val_bpb 1.05855 (as reported by @simonbissonnette).

Delta: TBD

## Key Changes vs PR #2014

### 1. Peer-LoRA ensemble with confidence routing

After each batch's per-doc LoRAs finish sliding-window training (k docs per batch -> k independent LoRAs), run k-1 peer forwards per doc using other docs' LoRAs:

- **Stash phase**: during the normal sliding-window eval, stash each doc's per-token NLLs and predictive entropies (entropy of the output distribution -- no target labels used).
- **Peer phase**: for each doc, run k-1 forwards with randomly-selected peer LoRAs from the same batch. `BatchedLinearLoRA.PEER_IDX` routes each batch row to a different doc's LoRA weights.
- **Blend**: on tokens where `predictive_entropy >= threshold` (uncertain), blend: `p = w * p_own + (1 - w) * mean(p_peers)`. Confident tokens use `p_own` only.

The routing gate is target-free: it uses the model's own entropy, not validation NLLs. This means the ensemble prediction is committed before seeing targets, avoiding post-hoc selection.

With `threshold = 0.5`, roughly 75% of tokens are routed through the ensemble.

### 2. TTT hyperparameter tuning

Per-doc LoRA LR and weight decay were tuned via line search (on a single H100, using `TTT_EVAL_ONLY` to skip retraining):

| Param | PR #2014 | This submission |
|---|---:|---:|
| `TTT_LORA_LR` | 0.0001 | 0.00015 |
| `TTT_WEIGHT_DECAY` | 0.5 | 0.25 |

Higher LR lets the per-doc LoRAs fit more aggressively; lower weight decay gives them more freedom. Both changes improve the baseline and the peer ensemble independently.

## New Env Vars

| Env var | Default | Description |
|---|---:|---|
| `TTT_PEER_ENSEMBLE_K` | 3 | Peers per batch incl. self (set 1 to disable) |
| `TTT_PEER_CONF_THRESHOLD` | 0.5 | Predictive entropy threshold for routing |
| `TTT_PEER_CONF_BLEND_W` | 0.8 | Weight on own prediction in blend |

## Reproducing

Uses the same CaseOps sp8192 dataset/tokenizer as PR #2014, sourced from HuggingFace:

- Dataset: `romeerp/parameter-golf-caseops-v1`
- Variant: `sp8192_lossless_caps_caseops_v1_reserved`

```bash
# Install lrzip (artifact compression)
sudo apt-get install -y lrzip

# Download data
python3 data/cached_challenge_fineweb.py --variant sp8192_lossless_caps_caseops_v1_reserved

# Run
SEED=314 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All hyperparameters (CASEOPS_ENABLED=1, VOCAB_SIZE=8192, ensemble settings, etc.) are baked into `train_gpt.py`.

## Hardware / Software

- 8xH100 80GB SXM
- PyTorch 2.10.0+cu130
- `lrzip` 0.651 (for `pergroup` compression)

## Attribution

See `submission.json`. Built on the PR #2014 stack (@simonbissonnette and earlier contributors).

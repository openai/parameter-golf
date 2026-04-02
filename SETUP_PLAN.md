# Parameter Golf — RTX 3070 Local Setup Plan

## Phase 1: Environment Setup (manual, requires terminal)

```bash
# 1. Install pip and venv in WSL
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv

# 2. Create and activate venv in the project
cd /mnt/c/dev/parameter-golf
python3 -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch with CUDA 12.x support
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 4. Install remaining dependencies
pip install numpy sentencepiece huggingface-hub tqdm

# 5. Download training data (start with 10 shards for iteration, ~1B tokens)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# 6. Install GitHub CLI
(type -p wget >/dev/null || sudo apt-get install wget -y) \
  && sudo mkdir -p -m 755 /etc/apt/keyrings \
  && out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  && cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
  && sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
  && sudo apt update && sudo apt install gh -y
gh auth login

# 7. Verify GPU access
python3 -c "import torch; print(torch.cuda.get_device_name(0)); print(f'{torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB')"
```

## Phase 2: Baseline Adaptation (train_gpt_3070.py)

Already created at `train_gpt_3070.py`. Changes from the original:

| Parameter | Original (8xH100) | Adapted (1x 3070) |
|-----------|-------------------|-------------------|
| `iterations` | 20,000 | 500 |
| `warmdown_iters` | 1,200 | 100 |
| `warmup_steps` | 20 | 10 |
| `grad_accum_steps` | 8 / world_size | 64 (env configurable) |
| `val_batch_size` | 524,288 | 65,536 |
| `val_loss_every` | 1,000 | 100 |
| `train_log_every` | 200 | 10 |
| `max_wallclock_seconds` | 600 | 0 (disabled, use iterations) |
| SDP backends | flash only | flash + mem_efficient fallback |

### Potential issues to watch for
- **OOM at 8 seqs/micro-batch**: If still OOMs, increase `GRAD_ACCUM_STEPS` to 128 (4 seqs/step)
- **torch.compile cold start**: First run will be slow (~5 min compilation). Subsequent runs use cache.
- **head_dim limit on SM86**: Default config uses head_dim=64, which is at the flash attention limit. If issues arise, mem_efficient backend is enabled as fallback.

## Phase 3: Run Baseline

```bash
source .venv/bin/activate
cd /mnt/c/dev/parameter-golf

# Run the adapted baseline
python3 train_gpt_3070.py

# Expected output: val_bpb score after 500 iterations
# This establishes the LOCAL baseline (will be worse than 8xH100 due to fewer steps)
```

## Phase 4: Incremental Leaderboard Replication

Implement one technique at a time, measuring impact:

### Step 1: Sliding Window Eval (stride=64)
- Pure eval-time change, zero training cost
- Expected: ~-0.032 bpb on full eval
- Modify only the `eval_val` function

### Step 2: FP16 Embedding Passthrough
- One-line change in quantization
- Expected: ~-0.005 bpb on quantized model
- Modify `quantize_state_dict_int8` to skip tied embeddings

### Step 3: 3x MLP Expansion + More Layers
- Change `mlp_mult` from 2 to 3, `num_layers` from 9 to 10-11
- May need to reduce batch size further to fit in 8GB
- Expected: significant bpb improvement

### Step 4: Int6 QAT with STE
- Add straight-through estimator fake quantization during training
- Eliminates quantization gap (~0.007 bpb saved)
- Requires new quantization functions + training loop changes

### Step 5: SmearGate + BigramHash
- SmearGate: per-dimension gate blending with previous token (~512 params)
- BigramHash: hash table for token pair embeddings (4096-10240 buckets)
- Architecture modifications to the Block class

### Step 6: EMA Weight Averaging
- Maintain exponential moving average (decay=0.997)
- Use EMA weights for evaluation
- Training loop modification

### Step 7: LeakyReLU(0.5)^2
- Replace `torch.relu(x).square()` with `F.leaky_relu(x, 0.5).square()`
- One-line change in MLP.forward
- Expected: ~-0.003 bpb

### Step 8: XSA (Cross-Sequence Attention)
- Subtract component aligned with each token's own value vector
- Apply to last 3-4 layers only
- Zero new parameters, ~2ms/step overhead

### Step 9: Test-Time Training (LoRA TTT)
- Score-first approach: score chunk, then train on it
- LoRA adapters on lm_head, c_q, c_v
- Eval-time only, no training changes

## Phase 5: Autoresearch Integration

Once the manual replication is done and we understand each technique:

1. Write `autoresearch/program.md` with all learned constraints (already done)
2. Set up the experiment loop to modify `train_gpt_3070.py`
3. Run autonomous exploration for novel improvements beyond the leaderboard
4. ~500 iterations per experiment = ~10-20 experiments/hour on RTX 3070

## File Layout

```
/mnt/c/dev/parameter-golf/
├── train_gpt.py              # original baseline (untouched)
├── train_gpt_3070.py          # adapted for single RTX 3070
├── train_gpt_mlx.py           # Apple Silicon version (untouched)
├── SETUP_PLAN.md              # this file
├── CLAUDE.md                  # project context for Claude Code
├── LOCAL_LLM_SETUP.md         # Ollama + Qwen setup notes
├── autoresearch/
│   ├── program.md             # parameter-golf adapted research agenda
│   ├── program_original.md    # Karpathy's original
│   ├── prepare_reference.py   # autoresearch eval harness reference
│   └── train_reference.py     # autoresearch train script reference
├── data/                      # training data (after download)
├── records/                   # leaderboard submissions
└── .venv/                     # Python virtual environment (after setup)
```

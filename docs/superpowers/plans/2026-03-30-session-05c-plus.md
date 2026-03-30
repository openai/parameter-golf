# Session 05c-plus Training Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a new training record with 4 quality improvements (XSA-all, VE128, warmdown 3500, LeakyReLU(0.5)^2) on top of the Session 03 anchor, ready for immediate 8xH100 launch.

**Architecture:** Copy Session 03 anchor train_gpt.py as base (clean, no GPTQ complexity). Apply 4 training-side changes. VE128 is the only non-trivial addition: shared ValueEmbedding(vocab=1024, dim=128) projected to kv_dim=256, injected into attention values on layers 9-10.

**Base change rationale:** Session 05b (GPTQ branch) is parked after 7 ablations. Using the clean Session 03 anchor keeps the diff narrow and avoids dragging GPTQ debug complexity into a training run. GPTQ replay on the new checkpoint requires a separate merge step (Phase 2).

**Tech Stack:** PyTorch, SDPA attention, Muon+Adam optimizers, NGC 26.03 container on Pegasus 8xH100.

**SWA finding:** Both PR #1019 and #634 collect SWA snapshots but only apply EMA at export. SWA is dead code in both. Not included in this bundle.

---

### Task 1: Create record directory and copy base

**Files:**
- Create: `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py`
- Create: `records/track_non_record_16mb/2026-03-30_training_bundle_plus/README.md`

- [x] **Step 1: Create directory and copy base file**

```bash
mkdir -p records/track_non_record_16mb/2026-03-30_training_bundle_plus
cp records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/train_gpt.py \
   records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py
```

- [ ] **Step 2: Create README**

Write a README documenting this is the 05c-plus bundle.

- [ ] **Step 3: Verify base compiles**

```bash
python3 -m py_compile records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py
```

---

### Task 2: Trivial constants (warmdown 3500, XSA 11, LeakyReLU)

**Files:**
- Modify: `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py`

- [ ] **Step 1: Change warmdown_iters**

In class `Hyperparameters`, change `warmdown_iters = 3000` to `warmdown_iters = 3500`.

- [ ] **Step 2: Change xsa_last_n**

In class `Hyperparameters`, change `xsa_last_n = 4` to `xsa_last_n = 11`.

- [ ] **Step 3: Change MLP activation**

In class `MLP.forward`, change `x = F.relu(self.fc(x))` to `x = F.leaky_relu(self.fc(x), negative_slope=0.5)`.

- [ ] **Step 4: Verify compiles**

```bash
python3 -m py_compile records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py
```

- [ ] **Step 5: Commit**

```bash
git add records/track_non_record_16mb/2026-03-30_training_bundle_plus/
git commit -m "research(protocol): Session 05c-plus — base + trivial constants"
```

---

### Task 3: ValueEmbedding module

**Files:**
- Modify: `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py`

- [ ] **Step 1: Add ValueEmbedding class**

Add after BigramHashEmbedding class (before Block class):

```python
class ValueEmbedding(nn.Module):
    """Reinject token identity into attention values at specific layers.
    Shared embedding projected to kv_dim with learnable scale."""
    def __init__(self, vocab_size: int, ve_dim: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)
```

- [ ] **Step 2: Add VE hyperparameters**

In class `Hyperparameters`:
```python
ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
ve_dim = int(os.environ.get("VE_DIM", 128))
ve_layers = os.environ.get("VE_LAYERS", "9,10")
```

- [ ] **Step 3: Add VE control tensor patterns**

Add `ve_layer_scales,ve_shared.scale` to CONTROL_TENSOR_NAME_PATTERNS string.

- [ ] **Step 4: Verify compiles**

---

### Task 4: VE integration into GPT model

**Files:**
- Modify: `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py`

- [ ] **Step 1: Add VE params to GPT.__init__**

Add parameters `ve_enabled`, `ve_dim`, `ve_layers` to GPT.__init__ signature. After blocks/XSA setup:

```python
kv_dim = num_kv_heads * (model_dim // num_heads)
self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
if self.ve_layer_indices:
    self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim)
    self.ve_layer_scales = nn.ParameterList(
        [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices]
    )
else:
    self.ve_shared = None
    self.ve_layer_scales = nn.ParameterList()
```

- [ ] **Step 2: Add _get_ve helper to GPT**

```python
def _get_ve(self, layer_idx: int, input_ids: Tensor, ve_cache: dict) -> Tensor | None:
    if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
        return None
    if 've' not in ve_cache:
        ve_cache['ve'] = self.ve_shared(input_ids)
    ve_idx = self.ve_layer_indices.index(layer_idx)
    return ve_cache['ve'] * self.ve_layer_scales[ve_idx].to(dtype=ve_cache['ve'].dtype)
```

- [ ] **Step 3: Modify Attention.forward to accept v_embed**

Add `v_embed: Tensor | None = None` parameter. After `v = self.c_v(x)`:
```python
if v_embed is not None:
    v = v + v_embed
```

- [ ] **Step 4: Modify Block.forward to pass v_embed**

Add `v_embed: Tensor | None = None` parameter. Pass to attention call.

- [ ] **Step 5: Modify GPT.forward to compute and pass VE**

Add `ve_cache: dict = {}` before the block loop. For each block call:
```python
ve = self._get_ve(i, input_ids, ve_cache)
x = self.blocks[i](x, x0, v_embed=ve)
```

- [ ] **Step 6: Modify GPT.forward_logits similarly**

Same VE injection pattern.

- [ ] **Step 7: Verify compiles**

---

### Task 5: VE optimizer setup and GPT constructor call

**Files:**
- Modify: `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py`

- [ ] **Step 1: Update GPT constructor call in main()**

Add `ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers` to both GPT() constructor calls (training model and eval model).

- [ ] **Step 2: Add VE params to optimizer**

After bigram optimizer setup:
```python
if base_model.ve_shared is not None:
    tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr})
    if base_model.ve_shared.proj is not None:
        scalar_params.append(base_model.ve_shared.proj.weight)
    scalar_params.append(base_model.ve_shared.scale)
    for s in base_model.ve_layer_scales:
        scalar_params.append(s)
```

- [ ] **Step 3: Add VE logging**

Add to feature logging line: `ve={args.ve_enabled} ve_layers={args.ve_layers}`

- [ ] **Step 4: Verify compiles**

- [ ] **Step 5: Commit**

```bash
git add records/track_non_record_16mb/2026-03-30_training_bundle_plus/
git commit -m "research(protocol): Session 05c-plus — VE128 + all training changes"
```

---

### Task 6: Launch script and final verification

- [ ] **Step 1: Full syntax check**

```bash
python3 -m py_compile records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py
git diff --check -- records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py
```

- [ ] **Step 2: Push and prepare launch command**

```bash
git push
```

8xH100 launch:
```bash
cd /netscratch/$USER/parameter-golf && git pull

srun -K -p H100 --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6 \
  --mem=200G --time=00:20:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
  --container-mounts="$(pwd)":"$(pwd)" --container-workdir="$(pwd)" \
  bash -c '
    export LOCAL_RANK=$SLURM_LOCALID RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS
    export PYTHONUNBUFFERED=1
    export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
    export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=bond,eth NCCL_P2P_LEVEL=NVL
    pip install --no-cache-dir sentencepiece zstandard 2>/dev/null
    python -u records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py
  '
```

---

### Success criteria

- Sliding s64 val_bpb < 1.1260 (anchor is 1.1290)
- Pre-quant EMA val_bpb < 1.1420 (anchor is 1.14472)
- step_avg within +5ms of anchor (91.37 ms)
- Artifact <= 16,000,000 bytes
- Steps >= 6000

### Decision gate after run

1. Evaluate with naive int6 export first.
2. Run one export-only GPTQ replay on the new checkpoint.
3. If GPTQ is sane, continue from there.
4. If GPTQ is still bad, park it and keep the naive-int6 result.


# Random Linear Maps + Learned Adapters (val_bpb: 1.607)

**val_bpb: 1.607** (sliding, s64) | **1.92 MB artifact** | 4xH200, unlimited compute track

## The Idea

What if 90% of your model's weights were just noise? 

Each linear layer gets a random base weight matrix. They are generated from a fixed seed at init time, like `42` or `1337`. Those base weights cost zero bytes in the artifact because they're regenerated from the seed at eval. Only small rank-16 adapters (LoRA-style A and B matrices) are learned and stored. Think of it like giving someone a house made of random LEGO bricks and a small bag of correct ones... they have to figure out which random bricks are useful and nudge the rest into place with the adapters.

A 512-dim, 5-layer model normally stores around 25M parameters. This approach stores 2.2M. The other 90% are deterministic noise from a seed. The artifact is 1.92 MB, 12% of the 16 MB budget.

## Results

### Depth Sweep (fixed 40-min training budget, rank=16, 512-dim)

| Layers | Step time | Steps      | Float BPB | Int6 BPB  | Artifact   |
|--------|-----------|------------|-----------|-----------|------------|
| 3      | 43ms      | 50,000     | 1.756     | 1.720     | 1.5 MB     |
| **4**  | **55ms**  | **43,879** | **1.648** | **1.665** | **1.7 MB** |
| **5**  | **67ms**  | **35,618** | **1.638** | **1.656** | **1.9 MB** |
| 6      | 85ms      | 28,255     | 1.654     | 1.675     | 2.1 MB     |
| 8      | 109ms     | 22,031     | 1.651     | 1.673     | 2.5 MB     |
| 11     | 142ms     | 16,922     | 1.770     | 1.794     | 3.1 MB     |
| 20     | 254ms     | 9,457      | 3.551     | 3.552     | 4.3 MB     |


4-5 layers is the sweet spot. Shallower models train faster and rack up more steps, which matters more than depth when the base weights carry no learned information. Go too shallow (3L) and there isn't enough compositional depth for language. Go too deep (20L) and gradients straight up can't propagate through that many random projections. The model with 20 layers learned nothing.

### Rank Sweep (768-dim, 11L, fixed 40-min budget)

| Rank   | Stored Params | BPB      | Artifact   |
|--------|---------------|----------|------------|
| **16** | **3.2M**      | **2.77** | **4.2 MB** |
| 32     | 5.2M          | 3.20     | 5.7 MB     |
| 64     | 9.3M          | 3.55     | 9.5 MB     |


This one's counterintuitive. Smaller adapters win. Rank 16 crushes rank 32 and 64 because the larger adapters need more training steps to converge, and the fixed time budget punishes them for it. This sweep was run at 768-dim/11L, a harder setting than the depth sweep. The directional finding (smaller rank wins) should hold at 512-dim, but the absolute BPB numbers aren't comparable to the depth table.

### Scaling with Training Time (5L, rank=16, 512-dim)

| Steps       | Training Time | Float BPB | Sliding BPB | Artifact   |
|-------------|---------------|-----------|-------------|------------|
| 4,212       | 10 min        | 2.58      | —           | 1.9 MB     |
| 20,000      | 49 min        | 1.81      | —           | 1.9 MB     |
| 50,000      | 2 hr          | 1.64      | 1.63        | 1.9 MB     |
| **200,000** | **3.75 hr**   | **1.66**  | **1.607**   | **1.9 MB** |


Sliding BPB keeps improving with more steps, though with diminishing returns. The float BPB at 200K (1.66) is slightly worse than 50K (1.64), likely from training instability mid-run (loss spiked around step 104K) that the warmdown didn't fully recover from. Sliding window evaluation smooths this out, which is why the sliding BPB still improved. The model hasn't fully converged.

## Architecture

```python
class RandomLinearWithAdapter(nn.Module):
    def __init__(self, in_features, out_features, seed, rank=16):
        # Random base: NOT stored, regenerated from seed
        g = torch.Generator().manual_seed(seed)
        self.register_buffer('base_weight',
            torch.randn(out_f, in_f, generator=g) / math.sqrt(in_f),
            persistent=False)

        # Learned adapter: stored in artifact
        self.adapter_A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.adapter_B = nn.Parameter(torch.zeros(out_f, rank))

    def forward(self, x):
        W = self.base_weight + self.adapter_B @ self.adapter_A
        return F.linear(x, W, self.bias)
```

`persistent=False` means the base weight never hits the state_dict. At load time, `__init__` regenerates it from the seed. Each layer gets a unique seed from its index.

Every attention projection (Q, K, V, output) and MLP layer (fc, proj) uses RandomLinearWithAdapter. Embeddings, norms, and other small parameters are fully learned.

### Model Configuration

| Component     | Detail                             |
|---------------|------------------------------------|
| Layers        | 5                                  |
| Dim           | 512                                |
| Heads         | 8 (4 KV, GQA)                      |
| MLP           | 3x, relu-squared                   |
| Adapter rank  | 16                                 |
| Stored params | 2.2M (11% of full model)           |
| Random params | ~20M (89%, not stored)             |
| EMA           | 0.997                              |
| Training      | Muon + AdamW, 200K steps on 4xH200 |

## What I Found

**Depth has a sweet spot with random projections.** 4-5 layers wins at a fixed time budget. More depth means fewer training steps, and step count is king when base weights carry no information. 20 layers learned nothing... literally.

**Smaller adapters optimize better.** Rank 16 beats 32 and 64. There's a capacity-optimization tradeoff here... bigger adapters have more capacity but need more steps to figure out how to use it.

**Random projections can do language modeling.** A 1.92 MB model with 90% random weights hits 1.607 BPB. The naive baseline (fully learned, 13.5 MB) hits 1.224 BPB. The gap is real, but the fact that it works at all is the interesting part.

**The artifact is hilariously small.** 1.92 MB is 12% of the 16 MB budget. You could fit eight of these models in one artifact. Ensembles, multi-model voting, whatever you want... there's room.

## Run Commands

```bash
# 40-min depth sweep (run for NUM_LAYERS=3,4,5,6,8,11,20)
RANDOM_LINEAR=1 ADAPTER_RANK=16 MODEL_DIM=512 NUM_HEADS=8 \
  NUM_KV_HEADS=4 NUM_LAYERS=5 EMA_ENABLED=1 \
  ITERATIONS=50000 MAX_WALLCLOCK_SECONDS=2400 \
  torchrun --standalone --nproc_per_node=4 train_gpt.py

# Long run (best config)
RANDOM_LINEAR=1 ADAPTER_RANK=16 MODEL_DIM=512 NUM_HEADS=8 \
  NUM_KV_HEADS=4 NUM_LAYERS=5 EMA_ENABLED=1 \
  ITERATIONS=200000 MAX_WALLCLOCK_SECONDS=14400 \
  torchrun --standalone --nproc_per_node=4 train_gpt.py
```

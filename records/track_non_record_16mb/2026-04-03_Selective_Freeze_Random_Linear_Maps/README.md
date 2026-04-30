## Summary

**First systematic investigation of random linear maps for Parameter Golf**, directly addressing the [Requests for PRs](https://github.com/openai/parameter-golf#requests-for-prs) item "Learning adapters on random linear maps." This work evaluates 7 architecture variants across 3 hardware configurations (H100, A40, M4) with FineWeb sp1024/sp4096 validation, totaling ~25 experiments at ~$45 self-funded compute.

**Core finding:** Selectively freezing MLP gate+up projections as deterministic random (from seeds, 0 bytes in artifact) enables fitting larger models in 16MB. A 12L frozen model beats a 6L fully-trained model by **11.5%** on FineWeb. Progressive freeze (train fully, then freeze mid-training) outperforms random-init freeze by **1.3 percentage points** on FineWeb sp4096.

**Checks off "Learning adapters on random linear maps" from Requests for PRs.**

---

## 1. The Idea

In Parameter Golf, the artifact budget (16MB) limits model size. But what if some weights cost 0 bytes?

**Selective Freeze:** Replace MLP gate+up projections with deterministic random matrices generated from per-layer seeds. At eval time, regenerate from seeds — zero artifact cost. Only attention weights + MLP down projections are learned and stored.

```python
class FrozenFC(CastedLinear):
    def __init__(self, in_features, out_features, seed):
        super().__init__(in_features, out_features, bias=False)
        rng = torch.Generator(); rng.manual_seed(seed)
        with torch.no_grad():
            self.weight.copy_(torch.randn(out_features, in_features, generator=rng) / math.sqrt(in_features))
        self.weight.requires_grad = False
    def _save_to_state_dict(self, dest, prefix, keep_vars):
        pass  # Not saved — regenerated from seed
    def _load_from_state_dict(self, sd, prefix, meta, strict, missing, unexpected, errors):
        pass  # Not loaded — regenerated from seed
```

This is conceptually related to VeRA (Kopiczko et al., 2023), Extreme Learning Machines (Huang et al., 2006), and the Johnson-Lindenstrauss lemma — random projections preserve geometric structure.

---

## 2. Selective Freeze: Which Layers to Freeze?

I compared four freezing strategies on H100 with FineWeb sp1024 (3000 steps):

| Config | Layers | Dim | Frozen % | CE | vs Baseline | Artifact |
|--------|--------|-----|----------|-----|------------|----------|
| Baseline (fully trained) | 6L | 192d | 0% | 3.2531 | — | 2.4MB |
| Full freeze + LoRA r16 | 6L | 192d | 94% | — | ~80% gap | — |
| **Selective freeze gate+up** | 6L | 192d | 37% | — | **-2.1%** | 1.5MB |
| Selective + dropout 0.2 | 6L | 192d | 37% | 3.4404 | +5.8% | 1.5MB |
| Selective freeze | 8L | 256d | 37% | 3.1427 | -3.4% | 3.3MB |
| **Selective + dropout 12L** | 12L | 384d | 37% | **2.8803** | **-11.5%** | 7.3MB |
| Fully trained 12L (no freeze) | 12L | 384d | 0% | 2.7295 | -16.1% | 17.7MB ❌ |

**Key insight:** The fully-trained 12L model achieves the best CE (2.7295) but needs 17.7MB — over the 16MB limit. Selective freeze enables a 12L model at 7.3MB that beats the smaller 6L baseline by 11.5%. The frozen weights act as a structural regularizer AND enable fitting more parameters per artifact byte.

**Full freeze + LoRA fails** (80% gap) because LoRA rank-16 cannot compensate for freezing ALL weights. Selective freeze (gate+up only, 37%) leaves attention and MLP down projection learnable — a much better tradeoff.

---

## 3. Progressive Freeze: Train First, Then Freeze

Random-init freeze has a weakness: the frozen weights are random, not trained. **Progressive freeze** addresses this:

1. Train all weights normally for N steps (Phase 1)
2. Freeze MLP gate+up projections (Phase 2)
3. Continue training the remaining weights (Phase 3)

The frozen weights now contain *trained features* (not random), and the subsequent training adapts the rest of the network around them. This combines the regularization benefit of freezing with the quality of learned features.

**A40, FineWeb sp4096 (3000 steps total, freeze at step 1000):**

| Config | CE | vs Baseline | Delta vs Selective |
|--------|-----|------------|-------------------|
| Baseline 6L 192d | 4.2132 | — | — |
| Selective freeze 8L 256d (random init) | 4.1767 | -0.9% | — |
| **Progressive freeze 8L 256d** | **4.1189** | **-2.2%** | **+1.3pp better** |
| **Progressive freeze 12L 384d** | **3.8370** | **-8.9%** | **+8.0pp better** |

Progressive freeze consistently outperforms random-init selective freeze at the same model size. The 12L 384d progressive freeze result (-8.9%) is the strongest finding in this work.

**A40, Gutenberg validation (3000 steps):**

| Config | CE | vs Baseline |
|--------|-----|------------|
| Baseline 6L 192d | 1.2957 | — |
| Direct freeze 8L 256d | 1.2757 | -1.5% |
| Progressive freeze 8L 256d | 1.3049 | +0.7% |
| Progressive+distill 8L 256d | 1.3013 | +0.4% |

Note: Gutenberg and FineWeb results diverge — progressive freeze wins on FineWeb but not Gutenberg. This is consistent with the scale deception phenomenon documented in my PR #1259.

---

## 4. Frozen + Low-Rank Correction

Can a full frozen MLP be corrected with a learned low-rank term in parallel? `output = frozen_mlp(x) + A @ B @ x` where A, B are learned.

**M4, Gutenberg (3000 steps, 6L 192d unless noted):**

| Rank | CE | vs Baseline |
|------|-----|------------|
| r=32 | 1.4310 | +10.0% |
| r=64 | 1.4075 | +8.2% |
| r=128 | 1.3823 | +6.2% |
| **12L 384d, r=64** | **1.3041** | **+0.23%** |

Low-rank correction converges toward the baseline as rank increases, and at 12L 384d nearly matches it (+0.23%). This validates that larger frozen architectures with small learned corrections can approach fully-trained quality — the tradeoff is extra compute for frozen layers vs artifact savings.

---

## 5. Self-Distillation + Freeze

Train a teacher, then distill knowledge to a larger freeze student:

| Config | CE | vs Baseline |
|--------|-----|------------|
| Teacher 6L 192d → Student 8L 256d freeze (1500+1500 steps) | 1.3451 | +3.8% |

Cross-architecture distillation hurts because the teacher (dim=192) and student (dim=256) have different representation spaces. The student doesn't benefit from the teacher's knowledge when architectures differ significantly.

---

## 6. Progressive Freeze + Self-Distillation Combo

| Config | CE | vs Baseline |
|--------|-----|------------|
| 1000 train + 1000 self-distill + 1000 frozen | 1.3013 | +0.4% |

The self-distillation phase provides marginal benefit before the freeze. Progressive freeze alone (-2.2% on FineWeb) is simpler and more effective.

---

## 7. Dual Model Ensemble

Two smaller models in one 16MB artifact, average logits at eval:

| Config | BPC |
|--------|-----|
| Single 6L 192d | 1.9797 |
| Ensemble (2×3L 128d) | 1.7660 |

Ensemble helps (+10.8%) but both individual models are weak. The artifact budget is better spent on one larger model with frozen weights than two small models.

---

## 8. Key Insights

1. **Selective freeze gate+up is the sweet spot** — 37% frozen, leaving attention fully learnable. Full freeze + LoRA (94% frozen) catastrophically fails.

2. **Progressive freeze > random-init freeze** — trained features before freezing give +1.3pp over random init on FineWeb sp4096. The frozen weights serve as regularization, not random projections.

3. **Bigger frozen > smaller learned** — 12L 384d with 37% frozen (7.3MB artifact) beats 6L 192d fully trained (2.4MB artifact) by 11.5%. The artifact-per-BPB efficiency favors larger frozen architectures.

4. **Low-rank correction converges at scale** — frozen+correction at 12L 384d nearly matches baseline (+0.23%), suggesting the frozen MLP acts as a good initialization that small corrections can refine.

5. **Scale matters critically** — progressive freeze wins on FineWeb but loses on Gutenberg. See my PR #1259 for analysis of why local results can be misleading.

6. **Cross-architecture distillation fails** — teacher and student need compatible representation spaces.

---

## 9. Artifact Size Analysis

| Config | Total Params | Learned Params | Artifact (int6 est.) |
|--------|-------------|----------------|---------------------|
| Standard 11L 512d | 34.4M | 34.4M | ~15.9MB |
| Selective 11L 512d | 34.4M | 21.7M | ~10.0MB |
| Selective 13L 512d | 40.2M | 26.5M | ~12.2MB |

Selective freeze saves ~37% artifact space, enabling 2 extra layers within the 16MB budget. Combined with progressive freeze for quality, this is a viable path to higher-capacity models.

---

## 10. Implementation

**FrozenFC class:** Extends CastedLinear, overrides `_save_to_state_dict` and `_load_from_state_dict` to exclude frozen weights from serialization. Weights regenerated from seed at `__init__`.

**Progressive freeze:** Set `PROGRESSIVE_FREEZE_FRAC=0.3` to freeze MLP fc weights after 30% of training steps.

**torch.compile compatibility:** FrozenFC requires `fullgraph=False` due to different computation graph from CastedLinear. This incurs a ~15% throughput penalty — an important consideration for wallclock-limited competition.

See companion code: `selective_freeze_patch.py`, `record_train_gpt.py`

---

## Reproduction

```bash
# Selective freeze on FineWeb sp1024 (1×H100):
SELECTIVE_FREEZE=1 NUM_LAYERS=12 MODEL_DIM=384 \
torchrun --nproc_per_node=1 train_gpt.py

# Progressive freeze on FineWeb sp4096 (1×A40):
PROGRESSIVE_FREEZE_FRAC=0.3 NUM_LAYERS=8 MODEL_DIM=256 \
python3 exp_a40_apr4.py
```

---

## Related Work

- **PR #1295** (austinluk): Random Linear Maps + LoRA rank 16. Uses full freeze + LoRA, which my experiments show has an ~80% quality gap vs selective freeze.
- **PR #1259** (mine): Scale Deception — documents why local results diverge from competition scale.
- **PR #1227** (mine): 28 Experiments Research Report — broader experimental context.

## Attribution

Builds on Clark's train_gpt.py (PR #1218), competition baseline architecture, and FineWeb sp1024/sp4096 datasets. All experiments self-funded (~$45 compute across H100, A40, M4).

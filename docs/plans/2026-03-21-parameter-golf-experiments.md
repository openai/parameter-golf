# Parameter Golf Experiments Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Beat the 1.1428 bpb SOTA with architecturally novel ideas inspired by Iain McGilchrist's hemispheric asymmetry research.

**Architecture overview:** Two parallel experiments — Experiment 1 is a pure eval change (safest win), Experiment 2 is a novel architectural approach (higher upside).

**Tech Stack:** PyTorch, torchrun DDP, Muon optimizer, zstd compression, SentencePiece sp1024 tokenizer, RunPod 1×H100 (experiments) / 8×H100 SXM (leaderboard)

---

## Experiment 1: LoRA TTT on SOTA (`2026-03-21_LoRA_TTT_SOTA`)

**Hypothesis:** The LoRA TTT trick (+0.004 bpb) was only applied to the naive baseline (1.1928). Applying it to the SOTA (1.1428) should yield a similar or larger gain since the base model is stronger. Training is 100% identical to SOTA — only eval changes.

**Files:**
- Create: `records/track_10min_16mb/2026-03-21_LoRA_TTT_SOTA/train_gpt.py`
- Create: `records/track_10min_16mb/2026-03-21_LoRA_TTT_SOTA/README.md`
- Create: `records/track_10min_16mb/2026-03-21_LoRA_TTT_SOTA/submission.json`

**Key implementation (added to SOTA train_gpt.py):**

```python
class BlockWithLoRA(nn.Module):
    """Block wrapper adding LoRA deltas to c_q and c_v. Only used during TTT eval."""
    def __init__(self, block: Block, lora_rank: int, model_dim: int):
        super().__init__()
        self.block = block
        kv_dim = block.attn.num_kv_heads * block.attn.head_dim
        self.lora_A_q = nn.Parameter(torch.zeros(lora_rank, model_dim))
        self.lora_B_q = nn.Parameter(torch.zeros(model_dim, lora_rank))
        self.lora_A_v = nn.Parameter(torch.zeros(lora_rank, model_dim))
        self.lora_B_v = nn.Parameter(torch.zeros(kv_dim, lora_rank))
        nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))

    def reset(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q)
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_v)

    def _attn_with_lora(self, x: Tensor) -> Tensor:
        attn = self.block.attn
        bsz, seqlen, dim = x.shape
        q = attn.c_q(x) + F.linear(F.linear(x, self.lora_A_q), self.lora_B_q)
        k = attn.c_k(x)
        v = attn.c_v(x) + F.linear(F.linear(x, self.lora_A_v), self.lora_B_v)
        q = q.reshape(bsz, seqlen, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = attn.rotary(seqlen, x.device, q.dtype)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q * attn.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                           enable_gqa=(attn.num_kv_heads != attn.num_heads))
        return attn.proj(y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.block.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.block.attn_scale.to(dtype=x.dtype)[None, None, :] * self._attn_with_lora(self.block.attn_norm(x))
        x = x + self.block.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.block.mlp(self.block.mlp_norm(x))
        return x
```

**TTT eval logic:**
- Find BOS (token=1) boundaries to isolate documents
- For each document: reset LoRA → slide window with stride=TTT_STRIDE (default 256)
- Per window: `forward_logits` → score last stride tokens → `backward` → `Adam.step`
- Accumulate BPB only on scored tokens (no double-counting)
- Distributed: split documents across ranks, all_reduce at end

**Hyperparams to sweep:**
- `lora_rank`: 4, 8, 16
- `ttt_lr`: 0.005, 0.01, 0.02
- `ttt_stride`: 128, 256, 512

**RunPod test command (1xH100):**
```bash
# After downloading data
SEED=42 python train_gpt.py  # single GPU, MAX_WALLCLOCK_SECONDS=600
```

**8xH100 leaderboard command:**
```bash
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

## Experiment 2: McGilchrist Register Token (`2026-03-21_McGilchrist_Register`)

**Hypothesis:** Inspired by Iain McGilchrist's hemispheric asymmetry research. The right hemisphere maintains broad, holistic, peripheral awareness; the left decomposes locally. Current transformers are pure left-hemisphere machines. Adding 1-2 "register tokens" that maintain full-sequence global state and FiLM-condition the per-token stream embodies the R-L-R dialectic in architecture.

**Key idea:** Add R "right-hemisphere" register tokens at each block that:
- Attend to the FULL sequence (broad, holistic awareness — the "peripheral attention" skill)
- Write back to per-token residuals via learned scale/shift (FiLM conditioning)
- Cost almost nothing: R=2 registers × 2×512 FiLM params per layer × 10 layers = 20K params

This forces the model to maintain a holistic summary that conditions all token-level processing — the "gestalt token" from McGilchrist's synthesis stage.

**Architecture changes to SOTA:**
1. Add `GlobalRegister` module: 2 global tokens with full-sequence (non-causal) read, causal write
2. Modify `Block.forward()` to: (a) update register from token stream, (b) condition tokens from register
3. Quantize register projections as int6 (they're cheap, precision matters)

**Files:**
- Create: `records/track_10min_16mb/2026-03-21_McGilchrist_Register/train_gpt.py`
- Create: `records/track_10min_16mb/2026-03-21_McGilchrist_Register/README.md`

**Status:** Design complete, implementation pending Experiment 1 run results.

---

## Experiment 3: Hermeneutic Depth Recurrence (`2026-03-21_Hermeneutic_Recurrence`)

**Hypothesis:** Gadamer's hermeneutic circle: understanding the whole requires the parts, understanding the parts requires a prior whole. In transformer terms: run 3-4 base layers N times (shared weights) with tiny per-cycle FiLM modulation. Each cycle is a hermeneutic iteration. Gets "effective depth 12-16" from N=4 base layers, under the 16MB artifact budget.

**Status:** Pending, will implement if Experiment 1 & 2 show gains.

---

## Execution Notes

- All experiments build on the SOTA `2026-03-21_LoRA_TTT_SOTA` script
- Local MLX smoke test before each RunPod run to catch bugs
- 3 seeds required for leaderboard submission (mean ± std, p < 0.01 vs SOTA)
- RunPod template: `console.runpod.io/deploy?template=y5cejece4j`
- SSH key must be set up in RunPod settings

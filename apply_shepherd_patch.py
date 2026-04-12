"""
Shepherd Embryo — Parameter Golf Integration Patch
Apply this to train_gpt.py to replace the flat GPT with Shepherd Embryo v0.0002.

Usage on RunPod:
  1. Upload this file to /workspace/parameter-golf/
  2. Upload shepherd_embryo_v0.0002.py to /workspace/parameter-golf/
  3. Run: python apply_shepherd_patch.py
  4. Train: RUN_ID=shepherd_v0002 ITERATIONS=200 VAL_LOSS_EVERY=100 torchrun --standalone --nproc_per_node=1 train_gpt.py

What this does:
  - Copies train_gpt.py to train_gpt_backup.py (safety)
  - Patches the GPT class instantiation to use ShepherdEmbryo
  - Patches the optimizer parameter grouping to work with Shepherd's modules
  - Keeps EVERYTHING else: data loading, evaluation, quantization, logging
"""

import shutil
import re
import os

TRAIN_SCRIPT = "train_gpt.py"
BACKUP_SCRIPT = "train_gpt_backup.py"

# The Shepherd Embryo model code to inject at the top of train_gpt.py
SHEPHERD_IMPORT = '''
# =============================================================================
# SHEPHERD EMBRYO v0.0002 — Topology-First Architecture
# Injected by apply_shepherd_patch.py
# =============================================================================
import math as _math

class _SeedGenerator(nn.Module):
    def __init__(self, vocab_size, model_dim, num_probes=5, rank=64):
        super().__init__()
        self.num_probes = num_probes
        self.model_dim = model_dim
        self.embed_low = nn.Embedding(vocab_size, rank)
        self.expand = CastedLinear(rank, model_dim, bias=False)
        self.probe_directions = nn.Parameter(torch.randn(num_probes, model_dim) * 0.02)
        nn.init.normal_(self.embed_low.weight, std=0.01)
        nn.init.normal_(self.expand.weight, std=0.02)

    def forward(self, token_ids):
        z = self.embed_low(token_ids)
        x = self.expand(z)
        seed_anchor = F.rms_norm(x, (x.size(-1),))
        B, S, D = seed_anchor.shape
        probes = seed_anchor.unsqueeze(1).expand(B, self.num_probes, S, D).clone()
        probes = probes + self.probe_directions[None, :, None, :]
        return seed_anchor, probes

class _ProbeBlock(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.mlp_fc = CastedLinear(dim, mlp_mult * dim, bias=False)
        self.mlp_proj = CastedLinear(mlp_mult * dim, dim, bias=False)
        self.mlp_proj._zero_init = True
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x):
        B, S, D = x.shape
        xn = self.attn_norm(x)
        q = self.c_q(xn).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(xn).reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(xn).reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(S, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                           enable_gqa=(self.num_kv_heads != self.num_heads))
        attn_out = self.proj(y.transpose(1, 2).contiguous().reshape(B, S, D))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp_proj(
            torch.relu(self.mlp_fc(self.mlp_norm(x))).square())
        return x

class _ProbeEngine(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.block = _ProbeBlock(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
    def forward(self, probes):
        B, P, S, D = probes.shape
        flat = probes.reshape(B * P, S, D)
        flat = self.block(flat)
        return flat.reshape(B, P, S, D)

class _ProbeScorer(nn.Module):
    def __init__(self, w_sim=0.5, w_coh=0.3, w_div=0.2):
        super().__init__()
        self.w_sim = w_sim
        self.w_coh = w_coh
        self.w_div = w_div
    def forward(self, probes, seed_anchor):
        B, P, S, D = probes.shape
        pp = probes.mean(dim=2)
        ap = seed_anchor.mean(dim=1)
        pn = F.normalize(pp, dim=-1)
        an = F.normalize(ap, dim=-1)
        sim = torch.bmm(pn, an.unsqueeze(-1)).squeeze(-1)
        pv = probes.var(dim=2).mean(dim=-1)
        coh = 1.0 / (pv + 1e-6)
        coh = coh / (coh.max(dim=1, keepdim=True).values + 1e-6)
        pw = torch.cdist(pp, pp)
        div = pw.sum(dim=-1) / (P - 1)
        div = div / (div.max(dim=1, keepdim=True).values + 1e-6)
        return self.w_sim * sim + self.w_coh * coh + self.w_div * div

class _ProbeFold(nn.Module):
    def __init__(self, model_dim, num_probes=5, top_k=2):
        super().__init__()
        self.num_probes = num_probes
        self.top_k = top_k
        self.re_expand = nn.Parameter(torch.randn(num_probes, model_dim) * 0.02)
    def forward(self, probes, scores):
        B, P, S, D = probes.shape
        _, top_idx = scores.topk(self.top_k, dim=1)
        top_exp = top_idx.unsqueeze(-1).unsqueeze(-1).expand(B, self.top_k, S, D)
        top_p = torch.gather(probes, 1, top_exp)
        top_s = torch.gather(scores, 1, top_idx)
        w = torch.softmax(top_s, dim=1).unsqueeze(-1).unsqueeze(-1)
        merged = (top_p * w).sum(dim=1)
        new_p = merged.unsqueeze(1).expand(B, self.num_probes, S, D).clone()
        new_p = new_p + self.re_expand[None, :, None, :]
        return new_p

class _Regulator(nn.Module):
    def __init__(self, model_dim, num_probes=5, num_depths=3, drift_threshold=0.6):
        super().__init__()
        self.drift_threshold = drift_threshold
        self.contraction = nn.Parameter(torch.full((num_depths,), 0.9, dtype=torch.float32))
        self.anchor_blend = nn.Parameter(torch.full((num_depths,), 0.1, dtype=torch.float32))
    def forward(self, probes, depth, seed_anchor):
        B, P, S, D = probes.shape
        pf = probes.reshape(B * P, S, D).mean(dim=1)
        af = seed_anchor.mean(dim=1)
        ae = af.unsqueeze(1).expand(B, P, -1).reshape(B * P, -1)
        drift = 1.0 - F.cosine_similarity(pf, ae, dim=-1)
        drift = drift.reshape(B, P)
        alpha = self.contraction[depth].to(dtype=probes.dtype)
        probes = probes * alpha
        a3d = seed_anchor.unsqueeze(1).expand(B, P, S, D)
        dm = (drift > self.drift_threshold).float()
        bl = self.anchor_blend[depth].to(dtype=probes.dtype)
        bw = (dm * bl).unsqueeze(-1).unsqueeze(-1)
        probes = probes * (1 - bw) + a3d * bw
        return probes

class ShepherdEmbryo(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap,
                 rope_base, qk_gain_init):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap

        # Config
        num_probes = 5
        seed_rank = model_dim // 6  # scale rank with model dim
        num_probe_depths = 3
        num_core_layers = max(2, num_layers // 3)  # use ~1/3 of layers for core
        fold_top_k = 2
        self.num_probe_depths = num_probe_depths
        self.num_probes = num_probes

        # Module 1: Seed
        self.seed = _SeedGenerator(vocab_size, model_dim, num_probes, seed_rank)

        # Module 2: Probe Engine (one per depth)
        self.probe_engines = nn.ModuleList([
            _ProbeEngine(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_probe_depths)
        ])

        # Scorer + Fold
        self.scorer = _ProbeScorer()
        self.folds = nn.ModuleList([
            _ProbeFold(model_dim, num_probes, fold_top_k)
            for _ in range(num_probe_depths)
        ])

        # Module 3: Regulator
        self.regulator = _Regulator(model_dim, num_probes, num_probe_depths)

        # Module 4: Micro Core
        self.core = nn.ModuleList([
            _ProbeBlock(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_core_layers)
        ])

        # Output
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids, target_ids):
        # Stage 1: Seed
        seed_anchor, probes = self.seed(input_ids)

        # Stage 2: Probe loop
        for depth in range(self.num_probe_depths):
            probes = self.probe_engines[depth](probes)
            scores = self.scorer(probes, seed_anchor)
            probes = self.folds[depth](probes, scores)
            probes = self.regulator(probes, depth, seed_anchor)

        # Stage 3: Collapse
        final_scores = self.scorer(probes, seed_anchor)
        weights = torch.softmax(final_scores, dim=1)
        x = (probes * weights[:, :, None, None]).sum(dim=1)

        # Stage 4: Micro Core
        for block in self.core:
            x = block(x)

        # Stage 5: Output
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.seed.expand.weight.T.to(x.dtype))
            logits_proj = F.linear(logits_proj, self.seed.embed_low.weight.to(x.dtype))
        else:
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

# =============================================================================
# END SHEPHERD EMBRYO INJECTION
# =============================================================================
'''

def patch_train_script():
    """Patch train_gpt.py to use ShepherdEmbryo instead of GPT."""

    if not os.path.exists(TRAIN_SCRIPT):
        print(f"ERROR: {TRAIN_SCRIPT} not found. Run from the parameter-golf directory.")
        return False

    # Backup
    if not os.path.exists(BACKUP_SCRIPT):
        shutil.copy2(TRAIN_SCRIPT, BACKUP_SCRIPT)
        print(f"  Backup saved: {BACKUP_SCRIPT}")
    else:
        print(f"  Backup already exists: {BACKUP_SCRIPT}")

    with open(TRAIN_SCRIPT, 'r') as f:
        code = f.read()

    # Check if already patched
    if 'ShepherdEmbryo' in code:
        print("  Already patched! ShepherdEmbryo found in train_gpt.py")
        return True

    # PATCH 1: Inject Shepherd classes after the existing Block/GPT classes
    # Find the end of the GPT class (before the TRAINING section)
    training_marker = "# -----------------------------\n# TRAINING\n# -----------------------------"
    if training_marker not in code:
        print("  ERROR: Could not find TRAINING section marker")
        return False

    code = code.replace(training_marker, SHEPHERD_IMPORT + "\n" + training_marker)
    print("  Injected ShepherdEmbryo class")

    # PATCH 2: Replace GPT instantiation with ShepherdEmbryo
    # Find the model creation line
    gpt_pattern = r'(raw_model\s*=\s*)GPT\('
    if re.search(gpt_pattern, code):
        code = re.sub(gpt_pattern, r'\1ShepherdEmbryo(', code)
        print("  Replaced GPT() with ShepherdEmbryo()")
    else:
        print("  WARNING: Could not find GPT() instantiation pattern")

    # PATCH 3: Fix optimizer parameter grouping
    # The baseline groups params by name patterns like 'tok_emb', 'lm_head', 'blocks'
    # Shepherd has different module names: 'seed', 'probe_engines', 'core', etc.
    # We need to make sure matrix params go to Muon and scalar params go to Adam
    # The simplest fix: replace the embed parameter detection
    old_embed_check = "if 'tok_emb' in name"
    new_embed_check = "if 'tok_emb' in name or 'embed_low' in name or 'expand' in name"
    if old_embed_check in code:
        code = code.replace(old_embed_check, new_embed_check)
        print("  Patched optimizer embed detection for Shepherd")

    # PATCH 4: Fix the lm_head check for tied embeddings
    # Shepherd uses seed.expand and seed.embed_low for tied output, not tok_emb
    old_head_check = "if 'lm_head' in name"
    new_head_check = "if 'lm_head' in name or ('seed' in name and 'embed' in name)"
    if old_head_check in code:
        # Only replace the first occurrence (in the optimizer setup)
        code = code.replace(old_head_check, new_head_check, 1)
        print("  Patched optimizer head detection for Shepherd")

    # PATCH 5: Reduce model_dim to fit probe overhead in memory
    # The baseline uses 512. Shepherd with 5 probes needs more memory per step.
    # Reduce to 384 to keep memory reasonable on A40.
    old_dim = 'model_dim = int(os.environ.get("MODEL_DIM", 512))'
    new_dim = 'model_dim = int(os.environ.get("MODEL_DIM", 384))'
    if old_dim in code:
        code = code.replace(old_dim, new_dim)
        print("  Reduced default MODEL_DIM to 384 for probe memory overhead")

    # PATCH 6: Reduce num_layers since Shepherd distributes them differently
    old_layers = 'num_layers = int(os.environ.get("NUM_LAYERS", 9))'
    new_layers = 'num_layers = int(os.environ.get("NUM_LAYERS", 6))'
    if old_layers in code:
        code = code.replace(old_layers, new_layers)
        print("  Reduced default NUM_LAYERS to 6 for Shepherd distribution")

    # PATCH 7: Adjust num_heads for 384 dim
    old_heads = 'num_heads = int(os.environ.get("NUM_HEADS", 8))'
    new_heads = 'num_heads = int(os.environ.get("NUM_HEADS", 6))'
    if old_heads in code:
        code = code.replace(old_heads, new_heads)
        print("  Reduced default NUM_HEADS to 6 for 384 dim")

    old_kv = 'num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))'
    new_kv = 'num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 3))'
    if old_kv in code:
        code = code.replace(old_kv, new_kv)
        print("  Reduced default NUM_KV_HEADS to 3 for 384 dim")

    # Write patched file
    with open(TRAIN_SCRIPT, 'w') as f:
        f.write(code)

    print(f"\n  Patch complete! {TRAIN_SCRIPT} now uses ShepherdEmbryo.")
    print(f"  Original saved as {BACKUP_SCRIPT}")
    print(f"\n  To train Shepherd:")
    print(f"    RUN_ID=shepherd_v0002 ITERATIONS=200 VAL_LOSS_EVERY=100 \\")
    print(f"    torchrun --standalone --nproc_per_node=1 train_gpt.py")
    print(f"\n  To restore baseline:")
    print(f"    cp {BACKUP_SCRIPT} {TRAIN_SCRIPT}")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  SHEPHERD EMBRYO — Parameter Golf Integration")
    print("=" * 60)
    print()
    success = patch_train_script()
    if success:
        print("\n  Ready to train Shepherd on FineWeb.")
    else:
        print("\n  Patch failed. Check errors above.")

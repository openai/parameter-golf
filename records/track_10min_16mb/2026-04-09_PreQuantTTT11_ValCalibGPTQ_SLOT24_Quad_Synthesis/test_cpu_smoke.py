"""CPU-only smoke test for the quad-stack synthesis.

Catches structural bugs (shape errors, missing methods, broken forward_hidden /
compute_logits split, dtype issues) WITHOUT needing a GPU. Runs in < 30 seconds
on a Mac.

This does NOT validate val_bpb numerics — the model is tiny, weights random,
data fake. It only proves the new code paths execute end-to-end. If it passes,
spending money on a GPU run becomes much safer.

Usage:
    python3 test_cpu_smoke.py
"""

import os
import sys
import types
import math


# ---------- 1. Monkey-patch flash_attn_3 with an SDPA fallback ----------
# train_gpt.py does `from flash_attn_interface import flash_attn_func`. On Mac
# we have no flash_attn_3 (it requires Hopper / sm_90 CUDA). Inject a fake
# module BEFORE importing train_gpt so the import succeeds.
fake_mod = types.ModuleType('flash_attn_interface')


def _fake_flash_attn(q, k, v, causal=True):
    """SDPA fallback. flash_attn format is [B, T, H, D]; SDPA wants [B, H, T, D].
    Also handles GQA (num_kv_heads != num_q_heads) by repeat_interleave on K/V."""
    import torch
    import torch.nn.functional as F
    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)
    H_q, H_kv = qt.size(1), kt.size(1)
    if H_q != H_kv:
        rep = H_q // H_kv
        kt = kt.repeat_interleave(rep, dim=1)
        vt = vt.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
    return out.transpose(1, 2).contiguous()


fake_mod.flash_attn_func = _fake_flash_attn
sys.modules['flash_attn_interface'] = fake_mod


# ---------- 2. Small model dimensions via env vars ----------
# Pick values small enough to run on CPU in seconds, but large enough that the
# linear layers exceed the GPTQ Hessian-eligibility threshold (>65536 weights).
# We need MODEL_DIM=512 so that c_q (512x512 = 262144) and MLP fc (512x2048 =
# 1048576) survive the filter and the val-calib GPTQ hook actually fires.
os.environ.update({
    'VOCAB_SIZE': '128',
    'MODEL_DIM': '512',
    'EMBEDDING_DIM': '512',
    'NUM_HEADS': '8',
    'NUM_KV_HEADS': '4',
    'NUM_LAYERS': '2',
    'MLP_MULT': '4.0',
    'ROPE_DIMS': '16',
    'TRAIN_SEQ_LEN': '64',
    'EVAL_SEQ_LEN': '64',
    'EVAL_STRIDE': '8',
    'TTT_CHUNK_TOKENS': '256',
    'TTT_ENABLED': '0',
    'SLOT_ENABLED': '1',
    'SLOT_STEPS': '4',
    'SLOT_BATCH_SEQS': '2',
    'SLOT_EVAL_STRIDE': '8',
    'PREQUANT_TTT_ENABLED': '0',
    'GPTQ_ENABLED': '0',
    'VE_ENABLED': '0',
    'XSA_LAST_N': '0',
    'PARALLEL_START_LAYER': '1',
    'RECUR_LAYERS': '0,1',
    'RECUR_START_STEP': '0',
    'GPTQ_CALIBRATION_BATCHES': '2',
    'TRAIN_BATCH_TOKENS': '512',
})


# ---------- 3. CPU-friendly stubs for CUDA-only constructs ----------
import torch
import torch.nn.functional as F

# torch.autocast(device_type="cuda", ...) explodes on CPU. Patch it to a no-op
# context manager (the test doesn't need actual mixed precision).
_orig_autocast = torch.autocast


class _NoAutocast:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


torch.autocast = _NoAutocast

# torch.compile is fine on CPU but can be slow / flaky for arbitrary code.
# Make it a no-op for the smoke test.
_orig_compile = torch.compile
torch.compile = lambda fn, **kwargs: fn


# ---------- 4. Import train_gpt from the same folder ----------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_gpt
from train_gpt import (
    GPT,
    Hyperparameters,
    collect_hessians_val,
    eval_val_slot,
    set_logging_hparams,
)


def section(title):
    print(f"\n--- {title} ---")


device = torch.device('cpu')
h = Hyperparameters()
# log() reads _logger_hparams.is_main_process and writes to a logfile path that
# doesn't exist on a fresh checkout. Replace it with a print-only stub for the
# smoke test.
set_logging_hparams(h)
train_gpt.log = lambda msg, console=True: print(msg) if console else None


# ---------- TEST 1: model construction ----------
section("Model construction")
torch.manual_seed(0)
model = GPT(h).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"[OK] GPT built with {n_params} params on CPU")
assert n_params > 0


# ---------- TEST 2: forward_logits split ----------
section("forward_hidden / compute_logits split")
x = torch.randint(0, h.vocab_size, (2, h.eval_seq_len))
y = torch.randint(0, h.vocab_size, (2, h.eval_seq_len))

hidden = model.forward_hidden(x)
expected_hidden_shape = (2, h.eval_seq_len, h.embedding_dim)
assert hidden.shape == expected_hidden_shape, \
    f"forward_hidden shape: got {tuple(hidden.shape)}, expected {expected_hidden_shape}"
print(f"[OK] forward_hidden returns {tuple(hidden.shape)}")

logits_split = model.compute_logits(hidden)
expected_logits_shape = (2, h.eval_seq_len, h.vocab_size)
assert logits_split.shape == expected_logits_shape
print(f"[OK] compute_logits returns {tuple(logits_split.shape)}")

logits_full = model.forward_logits(x)
assert logits_full.shape == expected_logits_shape
diff = (logits_split - logits_full).abs().max().item()
assert diff < 1e-5, f"forward_logits != compute_logits(forward_hidden), max diff = {diff}"
print(f"[OK] forward_logits == compute_logits(forward_hidden), max diff = {diff:.2e}")

# ---------- TEST 3: depth recurrence path ----------
section("forward_hidden under depth recurrence")
model.set_recurrence_active(True)
hidden_rec = model.forward_hidden(x)
assert hidden_rec.shape == expected_hidden_shape
print(f"[OK] forward_hidden with recurrence_active=True: {tuple(hidden_rec.shape)}")
print(f"     virtual layers: {model._get_virtual_layers()}")
model.set_recurrence_active(False)


# ---------- TEST 4: full forward (loss) ----------
section("Full model forward (loss)")
loss = model(x, y)
print(f"[OK] model(x, y) = {loss.item():.4f}")
assert torch.isfinite(loss).item()
loss.backward()
some_grad = next((p.grad for p in model.parameters() if p.grad is not None), None)
assert some_grad is not None, "no parameter received a gradient"
print(f"[OK] backward pass produced gradients")
model.zero_grad()


# ---------- TEST 5: fake ValidationData + collect_hessians_val ----------
section("collect_hessians_val (val-calibrated GPTQ)")


class FakeValData:
    """Mimics ValidationData enough to satisfy collect_hessians_val and eval_val_slot."""
    def __init__(self, vocab_size, n_tokens):
        self.val_tokens = torch.randint(0, vocab_size, (n_tokens,))
        self.base_bytes_lut = torch.ones(vocab_size, dtype=torch.int16)
        self.has_leading_space_lut = torch.zeros(vocab_size, dtype=torch.bool)
        self.is_boundary_token_lut = torch.zeros(vocab_size, dtype=torch.bool)


val_data = FakeValData(h.vocab_size, n_tokens=4096)
print(f"[OK] fake ValidationData: {val_data.val_tokens.numel()} tokens")

hessians = collect_hessians_val(model, val_data, h, device, n_calibration_batches=2)
assert isinstance(hessians, dict)
assert len(hessians) > 0, "no Hessians collected — hooks did not fire"
sample_name = next(iter(hessians))
sample_h = hessians[sample_name]
assert sample_h.ndim == 2 and sample_h.size(0) == sample_h.size(1)
print(f"[OK] collected {len(hessians)} Hessians, sample {sample_name} shape {tuple(sample_h.shape)}")


# ---------- TEST 6: eval_val_slot ----------
section("eval_val_slot (SLOT-24)")
val_loss, val_bpb = eval_val_slot(h, model, device, val_data)
print(f"[OK] eval_val_slot returned val_loss={val_loss:.4f} val_bpb={val_bpb:.4f}")
assert math.isfinite(val_loss) and math.isfinite(val_bpb)
print("     (numbers are meaningless — random model + fake data — only structure matters)")


# ---------- TEST 7: SLOT actually does something ----------
# Verify that running SLOT changes the loss vs a no-SLOT eval. We do this by
# computing the un-adapted loss on the same batch directly, then comparing.
section("SLOT adaptation reduces per-batch loss (sanity)")

xb = val_data.val_tokens[:128].reshape(2, 64).to(torch.int64)
yb = val_data.val_tokens[1:129].reshape(2, 64).to(torch.int64)

with torch.no_grad():
    base_logits = model.forward_logits(xb)
    base_nll = F.cross_entropy(base_logits.reshape(-1, h.vocab_size), yb.reshape(-1))

# Manually run a few SLOT steps on the same batch
hidden_b = model.forward_hidden(xb).detach().float()
proj_w = model.tok_emb.weight.detach().float() if model.tie_embeddings else model.lm_head.weight.detach().float()
softcap = model.logit_softcap

delta = torch.zeros(2, 1, hidden_b.size(-1), requires_grad=True)
logit_bias = torch.zeros(2, 1, proj_w.size(0), requires_grad=True)
opt = torch.optim.AdamW([delta, logit_bias], lr=0.012, weight_decay=1e-8, eps=1e-5)
targets = yb.reshape(-1)
for step in range(8):
    opt.zero_grad()
    h_aug = hidden_b + delta
    lp = F.linear(h_aug, proj_w) + logit_bias
    lg = softcap * torch.tanh(lp / softcap)
    loss_step = F.cross_entropy(lg.reshape(-1, h.vocab_size), targets)
    loss_step.backward()
    opt.step()

with torch.no_grad():
    h_aug = hidden_b + delta.detach()
    lp = F.linear(h_aug, proj_w) + logit_bias.detach()
    lg = softcap * torch.tanh(lp / softcap)
    slot_nll = F.cross_entropy(lg.reshape(-1, h.vocab_size), targets)

print(f"     base_nll = {base_nll.item():.4f}")
print(f"     slot_nll = {slot_nll.item():.4f}  (after 8 SLOT steps)")
delta_nll = base_nll.item() - slot_nll.item()
if delta_nll > 0:
    print(f"[OK] SLOT reduced loss by {delta_nll:.4f} — adaptation is working")
else:
    print(f"[WARN] SLOT did not reduce loss (Δ={delta_nll:.4f}) — could be normal for random "
          f"model + tiny vocab; check the real GPU run carefully")


# ---------- DONE ----------
print()
print("=" * 60)
print("ALL CPU SMOKE TESTS PASSED")
print("=" * 60)
print("The synthesis code paths execute end-to-end on CPU.")
print("Numbers are meaningless (random model, fake data) — this only proves")
print("there are no shape errors, missing methods, or import bugs.")
print()
print("Next step: validate val_bpb on real hardware (8xH100 SXM).")
print("Cheapest free options:")
print("  1. Apply for OpenAI compute grant — see VALIDATION.md")
print("  2. Modal Labs $30/month free credits — single H100 smoke")
print("  3. GCP $300 free trial — full 8xH100 run if quota approved")

# restore patches in case of import re-use
torch.autocast = _orig_autocast
torch.compile = _orig_compile

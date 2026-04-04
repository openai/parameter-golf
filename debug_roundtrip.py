"""Debug the quantization roundtrip with and without sliding window attention.

Runs on any single GPU (4050/4090/etc). No training, no dataset needed.
Creates a model with random weights, quantizes, dequantizes, and compares
forward pass outputs to isolate where the roundtrip breaks.
"""
import os
import sys
import torch
import torch.nn.functional as F
from torch import Tensor

# Force single GPU, no distributed
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"

# Minimal config for fast testing
os.environ["NUM_LAYERS"] = "4"  # small model, fast
os.environ["MODEL_DIM"] = "512"
os.environ["NUM_HEADS"] = "8"
os.environ["NUM_KV_HEADS"] = "4"
os.environ["MLP_MULT"] = "3"
os.environ["VOCAB_SIZE"] = "1024"
os.environ["TRAIN_SEQ_LEN"] = "256"
os.environ["BIGRAM_VOCAB_SIZE"] = "0"  # disable bigram for simplicity
os.environ["XSA_LAST_N"] = "0"  # disable XSA
os.environ["VE_ENABLED"] = "0"  # disable value embeddings
os.environ["ROPE_DIMS"] = "0"  # full RoPE
os.environ["LN_SCALE"] = "0"  # disable LN scale

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Import the SWA training script's classes
sys.path.insert(0, ".")

# We need to handle the flash_attn import — replace with SDPA for local testing
import importlib
import types

# Patch: create a fake flash_attn_interface module before importing train_gpt_swa
def sdpa_fallback(q, k, v, causal=True, window_size=(-1, -1), **kwargs):
    """SDPA fallback that handles window_size by ignoring it (full attention always)."""
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    y = F.scaled_dot_product_attention(
        q_t, k_t, v_t, attn_mask=None, is_causal=causal,
        enable_gqa=(q.size(2) != k.size(2)),
    )
    return y.transpose(1, 2)

# Create fake module
fake_fa = types.ModuleType("flash_attn_interface")
fake_fa.flash_attn_func = sdpa_fallback
sys.modules["flash_attn_interface"] = fake_fa

# Now import the training script
import train_gpt_swa as tgs

print("=" * 60)
print("TEST 1: Roundtrip WITHOUT sliding window")
print("=" * 60)

# Disable sliding window
os.environ["SWA_FULL_ATTN_LAYERS"] = "4"  # all 4 layers full attention
os.environ["SWA_WINDOW_SIZE"] = "0"

# Reload to pick up new env vars
importlib.reload(tgs)
args = tgs.Hyperparameters()

model_kwargs = dict(
    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
    tie_embeddings=True, tied_embed_init_std=0.005,
    logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
    mtp_num_heads=0, mtp_loss_weight=0.0,
    bigram_vocab_size=0, bigram_dim=128,
    xsa_last_n=0, rope_dims=0, ln_scale=False, dtg=False,
    ve_enabled=False, ve_dim=128, ve_layers="",
    gated_attention=False, value_residual=False,
)

model1 = tgs.GPT(**model_kwargs).to(device).bfloat16()
# Keep banks in float32 like the real training code
model1.qo_bank.data = model1.qo_bank.data.float()
model1.kv_bank.data = model1.kv_bank.data.float()
model1.mlp_up_bank.data = model1.mlp_up_bank.data.float()
model1.mlp_down_bank.data = model1.mlp_down_bank.data.float()
for m in model1.modules():
    if isinstance(m, tgs.CastedLinear):
        m.float()
tgs.restore_low_dim_params_to_fp32(model1)

print(f"  SWA layers: {model1._swa_layers}")
print(f"  Window sizes: {[b.attn.window_size for b in model1.blocks]}")

# Forward pass pre-quant
x = torch.randint(0, 1024, (2, 64), device=device)
y = torch.randint(0, 1024, (2, 64), device=device)
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss_pre = model1(x, y).item()
print(f"  Pre-quant loss: {loss_pre:.4f}")

# Quantize
sd = model1.state_dict()
# Use simple int8 quantization (not GPTQ — we're testing the pipeline, not GPTQ quality)
quant_obj, quant_stats = tgs.quantize_state_dict_int8(sd)
print(f"  Quantized: {quant_stats['num_float_tensors']} float tensors")

# Dequantize
deq_sd = tgs.dequantize_state_dict_int8(quant_obj)

# Load into a new model
model1_eval = tgs.GPT(**model_kwargs).to(device).bfloat16()
model1_eval.qo_bank.data = model1_eval.qo_bank.data.float()
model1_eval.kv_bank.data = model1_eval.kv_bank.data.float()
model1_eval.mlp_up_bank.data = model1_eval.mlp_up_bank.data.float()
model1_eval.mlp_down_bank.data = model1_eval.mlp_down_bank.data.float()
for m in model1_eval.modules():
    if isinstance(m, tgs.CastedLinear):
        m.float()
tgs.restore_low_dim_params_to_fp32(model1_eval)
model1_eval.load_state_dict(deq_sd, strict=True)

print(f"  Eval SWA layers: {model1_eval._swa_layers}")
print(f"  Eval window sizes: {[b.attn.window_size for b in model1_eval.blocks]}")

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss_post = model1_eval(x, y).item()
print(f"  Post-quant loss: {loss_post:.4f}")
print(f"  Gap: {loss_post - loss_pre:.4f}")

del model1, model1_eval
torch.cuda.empty_cache()

print()
print("=" * 60)
print("TEST 2: Roundtrip WITH sliding window (window=128, 2 full layers)")
print("=" * 60)

os.environ["SWA_FULL_ATTN_LAYERS"] = "2"  # layers 2,3 full; layers 0,1 window
os.environ["SWA_WINDOW_SIZE"] = "128"

importlib.reload(tgs)
args2 = tgs.Hyperparameters()

model2 = tgs.GPT(**model_kwargs).to(device).bfloat16()
model2.qo_bank.data = model2.qo_bank.data.float()
model2.kv_bank.data = model2.kv_bank.data.float()
model2.mlp_up_bank.data = model2.mlp_up_bank.data.float()
model2.mlp_down_bank.data = model2.mlp_down_bank.data.float()
for m in model2.modules():
    if isinstance(m, tgs.CastedLinear):
        m.float()
tgs.restore_low_dim_params_to_fp32(model2)

print(f"  SWA layers: {model2._swa_layers}")
print(f"  Window sizes: {[b.attn.window_size for b in model2.blocks]}")

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss_pre2 = model2(x, y).item()
print(f"  Pre-quant loss: {loss_pre2:.4f}")

sd2 = model2.state_dict()
quant_obj2, quant_stats2 = tgs.quantize_state_dict_int8(sd2)
deq_sd2 = tgs.dequantize_state_dict_int8(quant_obj2)

# Load into new model — this will also have sliding window from env vars
model2_eval = tgs.GPT(**model_kwargs).to(device).bfloat16()
model2_eval.qo_bank.data = model2_eval.qo_bank.data.float()
model2_eval.kv_bank.data = model2_eval.kv_bank.data.float()
model2_eval.mlp_up_bank.data = model2_eval.mlp_up_bank.data.float()
model2_eval.mlp_down_bank.data = model2_eval.mlp_down_bank.data.float()
for m in model2_eval.modules():
    if isinstance(m, tgs.CastedLinear):
        m.float()
tgs.restore_low_dim_params_to_fp32(model2_eval)
model2_eval.load_state_dict(deq_sd2, strict=True)

print(f"  Eval SWA layers (same as training): {model2_eval._swa_layers}")
print(f"  Eval window sizes: {[b.attn.window_size for b in model2_eval.blocks]}")

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss_post2 = model2_eval(x, y).item()
print(f"  Post-quant loss: {loss_post2:.4f}")
print(f"  Gap: {loss_post2 - loss_pre2:.4f}")

del model2, model2_eval
torch.cuda.empty_cache()

print()
print("=" * 60)
print("TEST 3: Train WITH sliding window, eval WITHOUT (the bug scenario)")
print("=" * 60)

os.environ["SWA_FULL_ATTN_LAYERS"] = "2"
os.environ["SWA_WINDOW_SIZE"] = "128"

importlib.reload(tgs)

model3 = tgs.GPT(**model_kwargs).to(device).bfloat16()
model3.qo_bank.data = model3.qo_bank.data.float()
model3.kv_bank.data = model3.kv_bank.data.float()
model3.mlp_up_bank.data = model3.mlp_up_bank.data.float()
model3.mlp_down_bank.data = model3.mlp_down_bank.data.float()
for m in model3.modules():
    if isinstance(m, tgs.CastedLinear):
        m.float()
tgs.restore_low_dim_params_to_fp32(model3)

print(f"  Training SWA layers: {model3._swa_layers}")
print(f"  Training window sizes: {[b.attn.window_size for b in model3.blocks]}")

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss_pre3 = model3(x, y).item()
print(f"  Pre-quant loss (with SWA): {loss_pre3:.4f}")

sd3 = model3.state_dict()
quant_obj3, quant_stats3 = tgs.quantize_state_dict_int8(sd3)
deq_sd3 = tgs.dequantize_state_dict_int8(quant_obj3)

# Load into new model — but DISABLE sliding window (like our eval code does)
model3_eval = tgs.GPT(**model_kwargs).to(device).bfloat16()
model3_eval.qo_bank.data = model3_eval.qo_bank.data.float()
model3_eval.kv_bank.data = model3_eval.kv_bank.data.float()
model3_eval.mlp_up_bank.data = model3_eval.mlp_up_bank.data.float()
model3_eval.mlp_down_bank.data = model3_eval.mlp_down_bank.data.float()
for m in model3_eval.modules():
    if isinstance(m, tgs.CastedLinear):
        m.float()
tgs.restore_low_dim_params_to_fp32(model3_eval)
model3_eval.load_state_dict(deq_sd3, strict=True)

# Override: disable sliding window on eval model (THIS IS WHAT WE DO IN THE REAL CODE)
for block in model3_eval.blocks:
    block.attn.window_size = (-1, -1)

print(f"  Eval SWA layers: {model3_eval._swa_layers}")
print(f"  Eval window sizes (after override): {[b.attn.window_size for b in model3_eval.blocks]}")

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss_post3 = model3_eval(x, y).item()
print(f"  Post-quant loss (without SWA): {loss_post3:.4f}")
print(f"  Gap: {loss_post3 - loss_pre3:.4f}")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Test 1 (no SWA → no SWA):     pre={loss_pre:.4f}  post={loss_post:.4f}  gap={loss_post-loss_pre:.4f}")
print(f"  Test 2 (SWA → SWA):            pre={loss_pre2:.4f}  post={loss_post2:.4f}  gap={loss_post2-loss_pre2:.4f}")
print(f"  Test 3 (SWA → no SWA):         pre={loss_pre3:.4f}  post={loss_post3:.4f}  gap={loss_post3-loss_pre3:.4f}")
print()
if abs(loss_post - loss_pre) < 0.5 and abs(loss_post2 - loss_pre2) < 0.5 and abs(loss_post3 - loss_pre3) < 0.5:
    print("  All gaps < 0.5 → Roundtrip pipeline works fine. Bug is elsewhere (torch version? GPTQ? torch.compile?)")
elif abs(loss_post3 - loss_pre3) > 0.5 and abs(loss_post2 - loss_pre2) < 0.5:
    print("  Test 3 broken, Test 2 fine → The SWA→noSWA override is the problem.")
elif abs(loss_post2 - loss_pre2) > 0.5:
    print("  Test 2 broken → Sliding window breaks quantization even without override.")
elif abs(loss_post - loss_pre) > 0.5:
    print("  Test 1 broken → Quantization pipeline itself is broken (not SWA related).")

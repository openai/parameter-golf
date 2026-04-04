"""Debug the full quantization roundtrip with 11-layer banked model.

Tests: parameter banks → unbank → int6 quantize → save → load → dequantize → rebank → eval
Runs on any single GPU. No training, no dataset, no GPTQ Hessians.
Uses GPTQ-lite (percentile search) instead of Full Hessian GPTQ.
"""
import io
import lzma
import os
import sys
import torch
import torch.nn.functional as F
from torch import Tensor
import types

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"

# Match the real #1 submission config
os.environ["NUM_LAYERS"] = "11"
os.environ["MODEL_DIM"] = "512"
os.environ["NUM_HEADS"] = "8"
os.environ["NUM_KV_HEADS"] = "4"
os.environ["MLP_MULT"] = "3"
os.environ["VOCAB_SIZE"] = "1024"
os.environ["TRAIN_SEQ_LEN"] = "256"
os.environ["BIGRAM_VOCAB_SIZE"] = "0"
os.environ["XSA_LAST_N"] = "11"
os.environ["VE_ENABLED"] = "0"
os.environ["ROPE_DIMS"] = "16"
os.environ["LN_SCALE"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}, VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MiB")
print(f"PyTorch: {torch.__version__}")

# SDPA fallback
def sdpa_fallback(q, k, v, causal=True, window_size=(-1, -1), **kwargs):
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    y = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=None, is_causal=causal,
                                        enable_gqa=(q.size(2) != k.size(2)))
    return y.transpose(1, 2)

fake_fa = types.ModuleType("flash_attn_interface")
fake_fa.flash_attn_func = sdpa_fallback
sys.modules["flash_attn_interface"] = fake_fa

sys.path.insert(0, ".")
import train_gpt_swa as tgs

model_kwargs = dict(
    vocab_size=1024, num_layers=11, model_dim=512,
    num_heads=8, num_kv_heads=4, mlp_mult=3.0,
    tie_embeddings=True, tied_embed_init_std=0.005,
    logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
    mtp_num_heads=0, mtp_loss_weight=0.0,
    bigram_vocab_size=0, bigram_dim=128,
    xsa_last_n=11, rope_dims=16, ln_scale=True, dtg=False,
    ve_enabled=False, ve_dim=128, ve_layers="",
    gated_attention=False, value_residual=False,
)


def create_model():
    m = tgs.GPT(**model_kwargs).to(device).bfloat16()
    m.qo_bank.data = m.qo_bank.data.float()
    m.kv_bank.data = m.kv_bank.data.float()
    m.mlp_up_bank.data = m.mlp_up_bank.data.float()
    m.mlp_down_bank.data = m.mlp_down_bank.data.float()
    for mod in m.modules():
        if isinstance(mod, tgs.CastedLinear):
            mod.float()
    tgs.restore_low_dim_params_to_fp32(m)
    return m


def forward_loss(model, x, y):
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return model(x, y).item()


def test_roundtrip(label, swa_window, swa_full_layers, override_eval_window=False):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  SWA_WINDOW_SIZE={swa_window}, SWA_FULL_ATTN_LAYERS={swa_full_layers}")
    if override_eval_window:
        print(f"  Eval: override all window_size to (-1,-1)")
    print(f"{'='*60}")

    os.environ["SWA_WINDOW_SIZE"] = str(swa_window)
    os.environ["SWA_FULL_ATTN_LAYERS"] = str(swa_full_layers)
    import importlib
    importlib.reload(tgs)

    model = create_model()
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  SWA layers: {model._swa_layers}")
    print(f"  Window sizes: {[b.attn.window_size for b in model.blocks]}")
    print(f"  Bank shapes: qo={list(model.qo_bank.shape)} kv={list(model.kv_bank.shape)} "
          f"up={list(model.mlp_up_bank.shape)} down={list(model.mlp_down_bank.shape)}")

    x = torch.randint(0, 1024, (1, 64), device=device)
    y = torch.randint(0, 1024, (1, 64), device=device)

    loss_pre = forward_loss(model, x, y)
    print(f"  Pre-quant loss: {loss_pre:.4f}")

    # Step 1: Get state dict (banked, 3D tensors)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    print(f"  State dict keys: {len(sd)}")
    for k, v in sd.items():
        if 'bank' in k:
            print(f"    {k}: {list(v.shape)} dtype={v.dtype}")

    # Step 2: Unbank (3D → individual 2D)
    sd_unbanked = tgs._unbank_state_dict(sd, model.num_layers)
    print(f"  Unbanked keys: {len(sd_unbanked)}")
    unbanked_sample = [k for k in sd_unbanked if 'blocks.0' in k]
    for k in unbanked_sample[:4]:
        print(f"    {k}: {list(sd_unbanked[k].shape)}")

    # Step 3: Int6 quantize (GPTQ-lite, no Hessians needed)
    int6_cats = {"mlp", "attn"}
    quant_result, quant_meta = tgs.mixed_quantize_int6(sd_unbanked, int6_cats)
    n_int6 = sum(1 for v in quant_meta.values() if isinstance(v, dict) and v.get("type") == "int6")
    n_int8 = sum(1 for v in quant_meta.values() if isinstance(v, dict) and v.get("type") == "int8")
    n_pass = sum(1 for v in quant_meta.values() if isinstance(v, str))
    print(f"  Quantized: {n_int6} int6, {n_int8} int8, {n_pass} passthrough")

    # Step 4: Serialize + compress + decompress + deserialize
    save_obj = {"result": quant_result, "meta": quant_meta}
    buf = io.BytesIO()
    torch.save(save_obj, buf)
    raw = buf.getvalue()
    compressed = lzma.compress(raw, preset=6)
    decompressed = lzma.decompress(compressed)
    loaded_obj = torch.load(io.BytesIO(decompressed), map_location="cpu")
    print(f"  Serialize: raw={len(raw):,} compressed={len(compressed):,} "
          f"ratio={len(raw)/len(compressed):.2f}x")

    # Step 5: Dequantize
    deq_unbanked = tgs.dequantize_mixed_int6(loaded_obj["result"], loaded_obj["meta"], sd_unbanked)
    print(f"  Dequantized keys: {len(deq_unbanked)}")

    # Step 6: Rebank (2D → 3D)
    template_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    deq_rebanked = tgs._rebank_state_dict(deq_unbanked, model.num_layers, template_sd)
    print(f"  Rebanked keys: {len(deq_rebanked)}")
    for k, v in deq_rebanked.items():
        if 'bank' in k:
            print(f"    {k}: {list(v.shape)} dtype={v.dtype}")

    # Step 7: Load into new model
    eval_model = create_model()
    if override_eval_window:
        for block in eval_model.blocks:
            block.attn.window_size = (-1, -1)
        print(f"  Eval window sizes (after override): {[b.attn.window_size for b in eval_model.blocks]}")

    eval_model.load_state_dict(deq_rebanked, strict=True)

    loss_post = forward_loss(eval_model, x, y)
    print(f"  Post-quant loss: {loss_post:.4f}")
    gap = loss_post - loss_pre
    print(f"  Gap: {gap:.4f} {'✓ OK' if abs(gap) < 0.5 else '✗ BROKEN'}")

    # Step 8: Also try with torch.compile
    try:
        compiled = torch.compile(eval_model, dynamic=False, fullgraph=True)
        loss_compiled = forward_loss(compiled, x, y)
        gap_compiled = loss_compiled - loss_pre
        print(f"  Post-quant+compile loss: {loss_compiled:.4f}  gap: {gap_compiled:.4f} "
              f"{'✓ OK' if abs(gap_compiled) < 0.5 else '✗ BROKEN'}")
    except Exception as e:
        print(f"  torch.compile failed: {e}")

    del model, eval_model
    torch.cuda.empty_cache()
    return loss_pre, loss_post, gap


# Run tests
results = []

# Test A: No sliding window (baseline)
r1 = test_roundtrip("A: No SWA (baseline)", swa_window=0, swa_full_layers=11)
results.append(("No SWA → No SWA", *r1))

# Test B: SWA on train and eval (same config)
r2 = test_roundtrip("B: SWA → SWA (same config)", swa_window=256, swa_full_layers=3)
results.append(("SWA → SWA", *r2))

# Test C: SWA on train, no SWA on eval (the real scenario)
r3 = test_roundtrip("C: SWA → No SWA (eval override)", swa_window=256, swa_full_layers=3,
                     override_eval_window=True)
results.append(("SWA → No SWA", *r3))

print("\n" + "=" * 60)
print("SUMMARY (full pipeline: bank → unbank → int6 → lzma → dequant → rebank → eval)")
print("=" * 60)
for name, pre, post, gap in results:
    status = "✓" if abs(gap) < 0.5 else "✗"
    print(f"  {status} {name:25s}  pre={pre:.4f}  post={post:.4f}  gap={gap:+.4f}")

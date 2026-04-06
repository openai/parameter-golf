"""Re-quantize the seq4096-trained model with GPTQ calibration at seq2048, eval at seq2048.

Downloads the raw pre-quant model (trained at seq4096) from HuggingFace,
re-runs Full Hessian GPTQ with AR calibration at seq_len=2048,
evaluates at seq_len=2048 with sliding window stride=64.
No retraining — eval only.
"""
import os
import modal

app = modal.App("parameter-golf-requant2048")

data_vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)
_hf_token = os.environ.get("HF_TOKEN", "")
if not _hf_token:
    _hf_token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(_hf_token_path):
        _hf_token = open(_hf_token_path).read().strip()
hf_secret = modal.Secret.from_dict({"HF_TOKEN": _hf_token})

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install(
        "torch==2.9.1",
        "numpy",
        "sentencepiece",
        "huggingface-hub",
        "datasets",
        "tqdm",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install("psutil", "packaging", "ninja", "wheel", "setuptools")
    .run_commands(
        "pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291",
    )
    .add_local_file("train_gpt_swa.py", "/opt/train_gpt_swa.py")
)


@app.function(
    image=image,
    gpu="H100:8",
    timeout=5400,
    volumes={"/data": data_vol},
    secrets=[hf_secret],
)
def requant_eval():
    import subprocess, os, shutil

    os.makedirs("/workspace/parameter-golf", exist_ok=True)
    os.chdir("/workspace/parameter-golf")
    shutil.copy2("/opt/train_gpt_swa.py", "train_gpt_swa.py")

    # Dataset
    dataset_vol = "/data/fineweb10B_sp1024"
    dataset_local = "./data/datasets/fineweb10B_sp1024"
    tokenizer_vol = "/data/tokenizers"
    tokenizer_local = "./data/tokenizers"
    if not os.path.exists(f"{dataset_vol}/fineweb_train_000000.bin"):
        subprocess.run(["git", "clone", "https://github.com/Itssshikhar/parameter-golf.git", "/tmp/repo"], check=True)
        shutil.copytree("/tmp/repo/data", "./data", dirs_exist_ok=True)
        subprocess.run(["python3", "data/cached_challenge_fineweb.py", "--variant", "sp1024"], check=True)
        os.makedirs(dataset_vol, exist_ok=True)
        os.makedirs(tokenizer_vol, exist_ok=True)
        for f in os.listdir(dataset_local): shutil.copy2(f"{dataset_local}/{f}", f"{dataset_vol}/{f}")
        for f in os.listdir(tokenizer_local): shutil.copy2(f"{tokenizer_local}/{f}", f"{tokenizer_vol}/{f}")
        data_vol.commit()
    else:
        os.makedirs("./data/datasets", exist_ok=True)
        os.makedirs("./data", exist_ok=True)
        if os.path.exists(dataset_local): shutil.rmtree(dataset_local)
        os.symlink(dataset_vol, dataset_local)
        if os.path.exists(tokenizer_local): shutil.rmtree(tokenizer_local)
        os.symlink(tokenizer_vol, tokenizer_local)

    eval_script = '''
import io, lzma, os, sys, time, math, torch, sentencepiece as spm
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

sys.path.insert(0, ".")

# Model config must match TRAINING config (seq4096, SWA w=256, 5 full layers)
os.environ["SWA_WINDOW_SIZE"] = "256"
os.environ["SWA_FULL_ATTN_LAYERS"] = "5"
os.environ["BIGRAM_VOCAB_SIZE"] = "3072"
os.environ["BIGRAM_DIM"] = "112"
os.environ["NUM_LAYERS"] = "11"
os.environ["XSA_LAST_N"] = "11"
os.environ["ROPE_DIMS"] = "16"
os.environ["LN_SCALE"] = "1"
os.environ["VE_ENABLED"] = "1"
os.environ["VE_LAYERS"] = "9,10"
os.environ["TRAIN_SEQ_LEN"] = "4096"  # model was trained at 4096

import train_gpt_swa as tgs

distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
if distributed:
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
master = rank == 0
grad_accum_steps = 8 // world_size

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)

def log0(msg):
    if master: print(msg, flush=True)

args = tgs.Hyperparameters()
sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)

# Load val tokens at BOTH lengths
EVAL_SEQ_LEN = 2048
val_tokens_2048 = tgs.load_validation_tokens(args.val_files, EVAL_SEQ_LEN)
val_tokens_4096 = tgs.load_validation_tokens(args.val_files, 4096)
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = tgs.build_sentencepiece_luts(sp, args.vocab_size, device)

# Download raw pre-quant model (trained at seq4096)
log0("Downloading raw model...")
from huggingface_hub import hf_hub_download
pt_path = hf_hub_download("shikhar007/parameter-golf-gram-ns", "exp2_w256_full5_seed1337.pt")
log0(f"Downloaded: {pt_path}")

# Wait — this is the Exp 2 model (trained at seq2048), not the seq4096 model.
# The seq4096 model was uploaded as final_model.pt. Let me use that.
pt_path_4096 = hf_hub_download("shikhar007/parameter-golf-gram-ns", "final_model.pt")
log0(f"Downloaded seq4096 model: {pt_path_4096}")

model_kwargs = dict(
    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
    mtp_num_heads=0, mtp_loss_weight=0.0,
    bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
    xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale, dtg=args.dtg_enabled,
    ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    gated_attention=args.gated_attention, value_residual=args.value_residual,
)

# Load raw weights into model
base_model = tgs.GPT(**model_kwargs).to(device).bfloat16()
base_model.qo_bank.data = base_model.qo_bank.data.float()
base_model.kv_bank.data = base_model.kv_bank.data.float()
base_model.mlp_up_bank.data = base_model.mlp_up_bank.data.float()
base_model.mlp_down_bank.data = base_model.mlp_down_bank.data.float()
for m in base_model.modules():
    if isinstance(m, tgs.CastedLinear): m.float()
tgs.restore_low_dim_params_to_fp32(base_model)

banked_sd = torch.load(pt_path_4096, map_location="cpu")
base_model.load_state_dict(banked_sd, strict=True)
log0(f"Loaded seq4096 model. Params: {sum(p.numel() for p in base_model.parameters()):,}")

# Pre-quant eval at both lengths
log0("\\nPre-quant eval at 2048...")
compiled_2048 = torch.compile(base_model, dynamic=False, fullgraph=True)
torch.cuda.synchronize(); t0 = time.perf_counter()
pre_loss_2048, pre_bpb_2048 = tgs.eval_val(
    args, compiled_2048, rank, world_size, device, grad_accum_steps,
    val_tokens_2048, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    eval_seq_len=2048,
)
torch.cuda.synchronize()
log0(f"pre_quant@2048: val_loss={pre_loss_2048:.4f} val_bpb={pre_bpb_2048:.4f} time={1000*(time.perf_counter()-t0):.0f}ms")
del compiled_2048

# Unbank for GPTQ
sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
unbanked_sd = tgs._unbank_state_dict(sd_cpu, args.num_layers)

# Build Hessian model (with SWA matching training config)
log0("\\nBuilding Hessian model for GPTQ calibration at seq2048...")
hessian_model = tgs._HessianGPT(
    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings, logit_softcap=args.logit_softcap,
    rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
    bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
    xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
    ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
).to(device).bfloat16()
for m in hessian_model.modules():
    if isinstance(m, tgs.CastedLinear): m.float()
tgs.restore_low_dim_params_to_fp32(hessian_model)
hessian_model.load_state_dict(
    {k: v.to(device) for k, v in unbanked_sd.items() if k in hessian_model.state_dict()},
    strict=False,
)

# AR self-gen calibration at SEQ2048 (not 4096!)
log0("Generating AR calibration at seq_len=2048...")
t_gen = time.perf_counter()
ar_tokens = tgs.generate_autoregressive_calib(
    base_model, device, num_seqs=64, seq_len=2048,  # KEY: calibrate at 2048
    vocab_size=args.vocab_size, temperature=0.8, batch_size=8, seed=1337,
)
log0(f"Generated {len(ar_tokens)} sequences at seq2048 in {time.perf_counter()-t_gen:.1f}s")

# Collect Hessians at seq2048
log0("Collecting Hessians at seq2048...")
hessians = tgs.collect_hessians_from_tokens(hessian_model, ar_tokens, device)
log0(f"Collected hessians for {len(hessians)} layers")
del ar_tokens, hessian_model
torch.cuda.empty_cache()

# GPTQ quantize with 2048-calibrated Hessians
log0("Quantizing with Full Hessian GPTQ (2048-calibrated)...")
quant_result, quant_meta = tgs.mixed_quantize_int6(unbanked_sd, {"mlp", "attn"}, hessians=hessians)

# Compress
quant_buf = io.BytesIO()
torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
quant_raw = quant_buf.getvalue()
quant_blob = lzma.compress(quant_raw, preset=9)
log0(f"Compressed: {len(quant_blob)} bytes ({len(quant_blob)/1024/1024:.2f}MB)")

# Dequantize
loaded_obj = torch.load(io.BytesIO(lzma.decompress(quant_blob)), map_location="cpu")
deq_unbanked = tgs.dequantize_mixed_int6(loaded_obj["w"], loaded_obj["m"], unbanked_sd)
deq_state = tgs._rebank_state_dict(deq_unbanked, args.num_layers, sd_cpu)

# Load into eval model
eval_model = tgs.GPT(**model_kwargs).to(device).bfloat16()
eval_model.qo_bank.data = eval_model.qo_bank.data.float()
eval_model.kv_bank.data = eval_model.kv_bank.data.float()
eval_model.mlp_up_bank.data = eval_model.mlp_up_bank.data.float()
eval_model.mlp_down_bank.data = eval_model.mlp_down_bank.data.float()
for m in eval_model.modules():
    if isinstance(m, tgs.CastedLinear): m.float()
tgs.restore_low_dim_params_to_fp32(eval_model)
eval_model.load_state_dict(deq_state, strict=True)

# Roundtrip eval at 2048
log0("\\nRoundtrip eval at 2048...")
torch._dynamo.reset()
compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)
torch.cuda.synchronize(); t0 = time.perf_counter()
rt_loss, rt_bpb = tgs.eval_val(
    args, compiled_eval, rank, world_size, device, grad_accum_steps,
    val_tokens_2048, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    eval_seq_len=2048,
)
torch.cuda.synchronize()
log0(f"roundtrip@2048 val_loss:{rt_loss:.4f} val_bpb:{rt_bpb:.4f} time:{1000*(time.perf_counter()-t0):.0f}ms")

# Sliding eval at 2048
log0("\\nSliding eval at 2048 stride=64...")
torch.cuda.synchronize(); t0 = time.perf_counter()
sw_loss, sw_bpb = tgs.eval_val_sliding(
    args, eval_model, rank, world_size, device,
    val_tokens_2048, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    stride=64, eval_seq_len=2048,
)
torch.cuda.synchronize()
log0(f"sliding@2048 val_loss:{sw_loss:.4f} val_bpb:{sw_bpb:.4f} time:{1000*(time.perf_counter()-t0):.0f}ms")

log0(f"\\nSUMMARY:")
log0(f"  pre_quant@2048:   {pre_bpb_2048:.4f}")
log0(f"  roundtrip@2048:   {rt_bpb:.4f}  (quant_gap={rt_bpb - pre_bpb_2048:.4f})")
log0(f"  sliding@2048:     {sw_bpb:.4f}  (sliding_benefit={rt_bpb - sw_bpb:.4f})")
log0(f"  Current #1:       1.1147")

if distributed:
    dist.destroy_process_group()
'''

    with open("eval_requant2048.py", "w") as f:
        f.write(eval_script)

    print("=== REQUANT@2048 EVAL START ===", flush=True)
    proc = subprocess.Popen(
        ["torchrun", "--standalone", "--nproc_per_node=8", "eval_requant2048.py"],
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines = []
    for line in proc.stdout:
        line = line.rstrip()
        print(line, flush=True)
        lines.append(line)

    proc.wait()
    print(f"=== REQUANT@2048 EVAL DONE (exit code {proc.returncode}) ===", flush=True)
    return "\n".join(lines)


@app.local_entrypoint()
def main():
    log = requant_eval.remote()
    with open("requant_eval.log", "w") as f:
        f.write(log)
    print(f"\nLog saved to requant_eval.log")

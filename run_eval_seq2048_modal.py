"""Eval-only: load the seq4096-trained quantized model and eval at seq_len=2048.

Downloads final_model.int6.ptz from HuggingFace (seq4096+eval4096 run),
dequantizes, evals at seq_len=2048 with sliding window stride=64.
No retraining, no re-quantization.
"""
import os
import modal

app = modal.App("parameter-golf-eval2048")

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
    timeout=3600,
    volumes={"/data": data_vol},
    secrets=[hf_secret],
)
def eval_2048():
    import subprocess
    import os
    import shutil

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

# Model config must match training: seq4096, SWA w=256, 5 full layers
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

# Load validation tokens at seq_len=2048 (eval length)
EVAL_SEQ_LEN = 2048
val_tokens = tgs.load_validation_tokens(args.val_files, EVAL_SEQ_LEN)
sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = tgs.build_sentencepiece_luts(sp, args.vocab_size, device)

# Download quantized model (from seq4096+eval4096 run)
log0("Downloading quantized model...")
from huggingface_hub import hf_hub_download
ptz_path = hf_hub_download("shikhar007/parameter-golf-gram-ns", "final_model.int6.ptz")
log0(f"Downloaded: {ptz_path}")

# Build model with training config
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

# Dequantize
with open(ptz_path, "rb") as f:
    quant_blob = f.read()
quant_obj = torch.load(io.BytesIO(lzma.decompress(quant_blob)), map_location="cpu")

template_model = tgs.GPT(**model_kwargs)
template_sd = {k: v.detach().cpu() for k, v in template_model.state_dict().items()}
unbanked_template = tgs._unbank_state_dict(template_sd, args.num_layers)
deq_unbanked = tgs.dequantize_mixed_int6(quant_obj["w"], quant_obj["m"], unbanked_template)
deq_state = tgs._rebank_state_dict(deq_unbanked, args.num_layers, template_sd)
del template_model, template_sd, unbanked_template, deq_unbanked

# Load into eval model
eval_model = tgs.GPT(**model_kwargs).to(device).bfloat16()
eval_model.qo_bank.data = eval_model.qo_bank.data.float()
eval_model.kv_bank.data = eval_model.kv_bank.data.float()
eval_model.mlp_up_bank.data = eval_model.mlp_up_bank.data.float()
eval_model.mlp_down_bank.data = eval_model.mlp_down_bank.data.float()
for m in eval_model.modules():
    if isinstance(m, tgs.CastedLinear):
        m.float()
tgs.restore_low_dim_params_to_fp32(eval_model)
eval_model.load_state_dict(deq_state, strict=True)
del deq_state

log0(f"Model loaded. SWA layers: {eval_model._swa_layers}")
log0(f"Window sizes: {[b.attn.window_size for b in eval_model.blocks]}")

# Eval 1: Standard roundtrip at 4096 (verify model works, should match ~1.1306)
log0("\\nEval at seq_len=4096 (should match previous run)...")
val_tokens_4096 = tgs.load_validation_tokens(args.val_files, 4096)
compiled_4096 = torch.compile(eval_model, dynamic=False, fullgraph=True)
torch.cuda.synchronize(); t0 = time.perf_counter()
rt4096_loss, rt4096_bpb = tgs.eval_val(
    args, compiled_4096, rank, world_size, device, grad_accum_steps,
    val_tokens_4096, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    eval_seq_len=4096,
)
torch.cuda.synchronize()
log0(f"roundtrip@4096 val_loss:{rt4096_loss:.4f} val_bpb:{rt4096_bpb:.4f} time:{1000*(time.perf_counter()-t0):.0f}ms")
del compiled_4096

# Eval 2: Standard roundtrip at 2048 (the key test)
log0("\\nEval at seq_len=2048...")
torch._dynamo.reset()  # clear compiled graph for new shapes
compiled_2048 = torch.compile(eval_model, dynamic=False, fullgraph=True)
torch.cuda.synchronize(); t0 = time.perf_counter()
rt2048_loss, rt2048_bpb = tgs.eval_val(
    args, compiled_2048, rank, world_size, device, grad_accum_steps,
    val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    eval_seq_len=EVAL_SEQ_LEN,
)
torch.cuda.synchronize()
log0(f"roundtrip@2048 val_loss:{rt2048_loss:.4f} val_bpb:{rt2048_bpb:.4f} time:{1000*(time.perf_counter()-t0):.0f}ms")
del compiled_2048

# Eval 3: Sliding window at 2048 stride=64 (the money number)
log0("\\nSliding eval at seq_len=2048 stride=64...")
torch.cuda.synchronize(); t0 = time.perf_counter()
sw_loss, sw_bpb = tgs.eval_val_sliding(
    args, eval_model, rank, world_size, device,
    val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    stride=64, eval_seq_len=EVAL_SEQ_LEN,
)
torch.cuda.synchronize()
log0(f"sliding@2048 val_loss:{sw_loss:.4f} val_bpb:{sw_bpb:.4f} time:{1000*(time.perf_counter()-t0):.0f}ms")

# Eval 4: Sliding window at 4096 stride=64 (for comparison)
log0("\\nSliding eval at seq_len=4096 stride=64...")
torch.cuda.synchronize(); t0 = time.perf_counter()
sw4096_loss, sw4096_bpb = tgs.eval_val_sliding(
    args, eval_model, rank, world_size, device,
    val_tokens_4096, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    stride=64, eval_seq_len=4096,
)
torch.cuda.synchronize()
log0(f"sliding@4096 val_loss:{sw4096_loss:.4f} val_bpb:{sw4096_bpb:.4f} time:{1000*(time.perf_counter()-t0):.0f}ms")

log0(f"\\nSUMMARY:")
log0(f"  roundtrip@4096:  {rt4096_bpb:.4f}")
log0(f"  roundtrip@2048:  {rt2048_bpb:.4f}")
log0(f"  sliding@2048:    {sw_bpb:.4f}")
log0(f"  sliding@4096:    {sw4096_bpb:.4f}")
log0(f"  Current #1:      1.1147")

if distributed:
    dist.destroy_process_group()
'''

    with open("eval_seq2048.py", "w") as f:
        f.write(eval_script)

    print("=== EVAL START ===", flush=True)
    proc = subprocess.Popen(
        ["torchrun", "--standalone", "--nproc_per_node=8", "eval_seq2048.py"],
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
    print(f"=== EVAL DONE (exit code {proc.returncode}) ===", flush=True)
    return "\n".join(lines)


@app.local_entrypoint()
def main():
    log = eval_2048.remote()
    with open("eval_seq2048.log", "w") as f:
        f.write(log)
    print(f"\nLog saved to eval_seq2048.log")

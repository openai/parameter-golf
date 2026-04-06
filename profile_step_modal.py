"""Profile a single training step on 8xH100 to find bottlenecks.

Runs 50 steps with torch.profiler, outputs a breakdown of where time goes.
"""
import os
import modal

app = modal.App("parameter-golf-profile")

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
    timeout=1800,
    volumes={"/data": data_vol},
    secrets=[hf_secret],
)
def profile():
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

    profile_script = '''
import os, sys, time, torch
import torch.distributed as dist
sys.path.insert(0, ".")

os.environ["SWA_WINDOW_SIZE"] = "256"
os.environ["SWA_FULL_ATTN_LAYERS"] = "5"
os.environ["BIGRAM_VOCAB_SIZE"] = "3072"
os.environ["BIGRAM_DIM"] = "112"
os.environ["WARMDOWN_ITERS"] = "4000"

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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)

def log0(msg):
    if master: print(msg, flush=True)

args = tgs.Hyperparameters()

# Build model (same as training)
base_model = tgs.GPT(
    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
    mtp_num_heads=0, mtp_loss_weight=0.0,
    bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
    xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
    dtg=args.dtg_enabled, ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    gated_attention=args.gated_attention, value_residual=args.value_residual,
).to(device).bfloat16()
base_model.qo_bank.data = base_model.qo_bank.data.float()
base_model.kv_bank.data = base_model.kv_bank.data.float()
base_model.mlp_up_bank.data = base_model.mlp_up_bank.data.float()
base_model.mlp_down_bank.data = base_model.mlp_down_bank.data.float()
for m in base_model.modules():
    if isinstance(m, tgs.CastedLinear): m.float()
tgs.restore_low_dim_params_to_fp32(base_model)

compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

# Optimizers (same as training)
block_named = list(base_model.blocks.named_parameters())
bank_params = [base_model.qo_bank, base_model.kv_bank, base_model.mlp_up_bank, base_model.mlp_down_bank]

optimizer_muon = tgs.Muon(bank_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                           backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd)
for g in optimizer_muon.param_groups: g["base_lr"] = args.matrix_lr

scalar_params = [p for n, p in block_named if p.ndim < 2 or any(pat in n for pat in tgs.CONTROL_TENSOR_NAME_PATTERNS)]
if hasattr(base_model, 'skip_weights'): scalar_params.append(base_model.skip_weights)
if hasattr(base_model, 'smear'): scalar_params.extend(base_model.smear.parameters())
if base_model.bigram is not None:
    scalar_params.append(base_model.bigram.scale)
    if base_model.bigram.proj is not None:
        optimizer_muon.param_groups[0]["params"].append(base_model.bigram.proj.weight)

embed_params = [base_model.tok_emb.weight]
if base_model.bigram is not None: embed_params.append(base_model.bigram.embed.weight)

optimizer_tok = torch.optim.AdamW([{"params": embed_params, "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr}],
                                   betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
optimizer_scalar = torch.optim.AdamW([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
                                      betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]

train_loader = tgs.DistributedTokenLoader(args.train_files, rank, world_size, device)
grad_accum_steps = 8 // world_size
grad_scale = 1.0 / grad_accum_steps

# Warmup (prime torch.compile)
log0("Warming up torch.compile...")
model.train()
for i in range(20):
    for opt in optimizers: opt.zero_grad(set_to_none=True)
    x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(x, y)
    (loss * grad_scale).backward()
    if hasattr(optimizer_muon, 'launch_reduce_scatters'):
        optimizer_muon.launch_reduce_scatters()
    for opt in optimizers: opt.step()
    for opt in optimizers: opt.zero_grad(set_to_none=True)
log0("Warmup done.")

# Reset
train_loader = tgs.DistributedTokenLoader(args.train_files, rank, world_size, device)

# Profile 30 steps (skip first 5 for warmth)
log0("\\nProfiling 30 training steps...")

torch.cuda.synchronize()
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=2, warmup=3, active=25),
    record_shapes=True,
    with_stack=False,
) as prof:
    for step in range(30):
        for opt in optimizers: opt.zero_grad(set_to_none=True)

        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            (loss * grad_scale).backward()

        if hasattr(optimizer_muon, 'launch_reduce_scatters'):
            optimizer_muon.launch_reduce_scatters()
        for opt in optimizers: opt.step()
        for opt in optimizers: opt.zero_grad(set_to_none=True)

        prof.step()

torch.cuda.synchronize()

# Print results (rank 0 only)
if master:
    log0("\\n" + "=" * 80)
    log0("CUDA TIME BREAKDOWN (top 30 ops by CUDA time)")
    log0("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30), flush=True)

    log0("\\n" + "=" * 80)
    log0("CPU TIME BREAKDOWN (top 30 ops by CPU time)")
    log0("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30), flush=True)

    log0("\\n" + "=" * 80)
    log0("SELF CUDA TIME (top 30 individual kernels)")
    log0("=" * 80)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30), flush=True)

if distributed:
    dist.destroy_process_group()
'''

    with open("profile_step.py", "w") as f:
        f.write(profile_script)

    print("=== PROFILE START ===", flush=True)
    proc = subprocess.Popen(
        ["torchrun", "--standalone", "--nproc_per_node=8", "profile_step.py"],
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
    print(f"=== PROFILE DONE (exit code {proc.returncode}) ===", flush=True)
    return "\n".join(lines)


@app.local_entrypoint()
def main():
    log = profile.remote()
    with open("profile_step.log", "w") as f:
        f.write(log)
    print(f"\nLog saved to profile_step.log")

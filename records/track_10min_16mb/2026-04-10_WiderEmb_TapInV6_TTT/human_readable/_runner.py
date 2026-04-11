import subprocess, sys, os, time
import torch, torch.distributed as dist, torch.nn.functional as F

# Build cutlass extension before importing train_gpt (which does `import cutlass_evt_fusion` at module level)
for p in sys.path:
    _cd = os.path.join(p, "cutlass_evt_fusion", "csrc")
    if os.path.isdir(_cd):
        subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"],
                               cwd=os.path.join(p, "cutlass_evt_fusion"),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        break

# V6 + TTT env defaults
for k, v in [("SLIDING_WINDOW_ENABLED","1"),("TAPIN_CPP","1"),("TAPIN_V4_ENABLED","1"),
             ("TAPIN_V6_CROSS","1"),("TAPIN_V6_CROSS_W","0.06"),("TAPIN_V4_ENTROPY_MIN","0.0"),
             ("TAPIN_V4_BOOST_PER_MATCH","0.02"),("TAPIN_V4_MIN_MATCH","3"),
             ("TAPIN_V4_MAX_MATCH","100"),("TAPIN_V4_TOP_K","1000"),
             ("TTT_ENABLED","1"),("TTT_LR","0.005"),("TTT_EPOCHS","3"),
             ("TTT_CHUNK_TOKENS","32768"),("TTT_FREEZE_BLOCKS","0")]:
    os.environ.setdefault(k, v)

import train_gpt
from tapin_cpp import make_fast_apply_tapin_rule_v6

world_size = int(os.environ.get("WORLD_SIZE", "1"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
rank = int(os.environ.get("RANK", "0"))
distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
if distributed:
    dist.init_process_group(backend="nccl")
    dist.barrier()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
enable_cudnn_sdp(False)
enable_flash_sdp(True)
enable_mem_efficient_sdp(False)
enable_math_sdp(False)
torch._dynamo.config.optimize_ddp = False

h = train_gpt.Hyperparameters()
os.makedirs("logs", exist_ok=True)
train_gpt.set_logging_hparams(h)

# Phase 1: Train + Quant + Raw SW eval
train_gpt.train_and_eval(h, device)

# Phase 2: V6 + TTT on fresh model load
def _unfused(self, x):
    return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())
train_gpt.MLP.forward = _unfused

def _tilt_ref():
    return train_gpt._ngram_tilt_hints, train_gpt._ngram_tilt_betas
train_gpt.apply_tapin_rule_v5 = make_fast_apply_tapin_rule_v6(
    train_gpt._v5_stats, _tilt_ref)

h.tapin_v4_enabled = True
h.ttt_enabled = True
val_data = train_gpt.ValidationData(h, device)
model = train_gpt.deserialize(h, device)
if h.num_loops > 0:
    model.looping_active = True
model.eval()
train_gpt._ngram_tilt_hints = None
train_gpt._ngram_tilt_betas = None

if rank == 0:
    print(f"\n=== V6 + TTT ===")
    print(f"  TTT: lr={h.ttt_lr} epochs={h.ttt_epochs} chunk={h.ttt_chunk_tokens} freeze={h.ttt_freeze_blocks}")
t0 = time.time()
ttt_loss, ttt_bpb = train_gpt.eval_val_sliding_ttt(h, device, val_data, model)
if rank == 0:
    print(f"  val_loss: {ttt_loss:.6f}  val_bpb: {ttt_bpb:.6f}  time: {time.time()-t0:.1f}s")

if distributed:
    dist.destroy_process_group()

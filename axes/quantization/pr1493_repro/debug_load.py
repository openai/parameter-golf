"""Debug: compare different weight loading methods on H200."""
import os, sys, torch, numpy as np
from pathlib import Path

os.environ.setdefault("VOCAB_SIZE", "8192")
os.environ.setdefault("DATA_DIR", "/home/willzhao/parameter-golf/data")
os.environ.setdefault("BUNDLE_DIR", "./bundle")
os.environ.setdefault("TTT_ENABLED", "0")
os.environ.setdefault("VAL_LOSS_EVERY", "0")

from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)

sys.path.insert(0, str(Path(__file__).parent))
import train_save_bundle as tsb

h = tsb.Hyperparameters()
device = torch.device("cuda", 0)
torch.backends.cuda.matmul.allow_tf32 = True

ema = torch.load("bundle/ema_weights.pt", map_location="cpu")

# Load val batch
val_file = Path("/home/willzhao/parameter-golf/data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin")
header = np.fromfile(val_file, dtype="<i4", count=256)
tokens = np.fromfile(val_file, dtype="<u2", count=int(header[2]), offset=256*4)
tokens = torch.from_numpy(tokens.astype(np.int64)).to(device)
x = tokens[:2048].unsqueeze(0)
y = tokens[1:2049].unsqueeze(0)

def test_model(label, model):
    if h.num_loops > 0:
        model.looping_active = True
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
    print(f"{label}: loss={loss.item():.6f}")

# Method A: standard path (bf16 + restore_fp32 + load)
mA = tsb.GPT(h).to(device).bfloat16()
tsb.restore_fp32_params(mA)
sdA = mA.state_dict()
mA.load_state_dict({k: v.to(sdA[k].dtype) for k, v in ema.items()}, strict=True)
test_model("A: bf16+restore+load (our path)", mA)

# Method B: all bf16, no restore_fp32_params
mB = tsb.GPT(h).to(device).bfloat16()
mB.load_state_dict({k: v.to(torch.bfloat16) for k, v in ema.items()}, strict=True)
test_model("B: all bf16 no restore", mB)

# Method C: construct fp32, load fp32, then to(device).bfloat16()
mC = tsb.GPT(h)  # fp32 by default
mC.load_state_dict(ema, strict=True)
mC = mC.to(device).bfloat16()
tsb.restore_fp32_params(mC)
test_model("C: fp32 construct+load, then bf16+restore", mC)

# Method D: same as A but with torch.compile
mD = tsb.GPT(h).to(device).bfloat16()
tsb.restore_fp32_params(mD)
sdD = mD.state_dict()
mD.load_state_dict({k: v.to(sdD[k].dtype) for k, v in ema.items()}, strict=True)
if h.num_loops > 0: mD.looping_active = True
mD.eval()
compiled_mD = torch.compile(mD, dynamic=False, fullgraph=True)
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        lossD = compiled_mD(x, y)
print(f"D: bf16+restore+load+COMPILE: loss={lossD.item():.6f}")

# Method E: check INIT loss for reference
mE = tsb.GPT(h).to(device).bfloat16()
tsb.restore_fp32_params(mE)
test_model("E: INIT weights (no load)", mE)

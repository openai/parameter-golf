"""Quick PyTorch profiler to identify bottlenecks in the training step."""
import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
import os, sys, time

# Minimal setup - import model from train_gpt
os.environ.setdefault("ITERATIONS", "50")
os.environ.setdefault("WARMUP_STEPS", "0")
os.environ.setdefault("VAL_LOSS_EVERY", "0")
os.environ.setdefault("TRAIN_LOG_EVERY", "10")

# Import model classes
sys.path.insert(0, ".")
from train_gpt import GPT, CastedLinear, restore_low_dim_params_to_fp32, CONTROL_TENSOR_NAME_PATTERNS, Muon

def profile_model(use_swiglu=False):
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    model = GPT(
        vocab_size=1024, num_layers=9, model_dim=512, num_heads=8,
        num_kv_heads=4, mlp_mult=2, tie_embeddings=True,
        tied_embed_init_std=0.005, logit_softcap=30.0, rope_base=10000.0,
        qk_gain_init=1.5, use_swiglu=use_swiglu, swiglu_hidden=672,
    ).to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(model)
    model = torch.compile(model, dynamic=False, fullgraph=True)

    # Dummy data
    x = torch.randint(0, 1024, (64, 1024), device=device)
    y = torch.randint(0, 1024, (64, 1024), device=device)

    # Warmup
    for _ in range(5):
        loss = model(x, y)
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()

    name = "SwiGLU" if use_swiglu else "ReLU²"
    print(f"\n{'='*60}")
    print(f"Profiling {name} model")
    print(f"{'='*60}")

    # Time raw speed
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        loss = model(x, y)
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    avg_ms = 1000 * (time.perf_counter() - t0) / 20
    print(f"Average step time: {avg_ms:.1f}ms")

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(5):
            with record_function("forward"):
                loss = model(x, y)
            with record_function("backward"):
                loss.backward()
            with record_function("zero_grad"):
                model.zero_grad()

    # Print summary
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    # Export chrome trace
    trace_file = f"/home/ubuntu/profile_{name.lower().replace('²','2')}.json"
    prof.export_chrome_trace(trace_file)
    print(f"Chrome trace saved to {trace_file}")

if __name__ == "__main__":
    profile_model(use_swiglu=False)
    profile_model(use_swiglu=True)

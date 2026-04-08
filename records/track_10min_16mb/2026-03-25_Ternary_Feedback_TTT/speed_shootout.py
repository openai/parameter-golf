import mlx.core as mx
import time
import numpy as np
from train_gpt_mlx import GPT, Hyperparameters

def benchmark(args, name):
    args.iterations = 10
    model = GPT(args)
    # Warmup
    x = mx.random.randint(0, 8192, (1, 1024))
    y = mx.random.randint(0, 8192, (1, 1024))
    
    # Force compile
    def train_step(x, y):
        return model.loss(x, y)
    
    compiled_step = mx.compile(train_step)
    
    # Warmup
    loss = compiled_step(x, y)
    mx.eval(loss)
    
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        loss = compiled_step(x, y)
        mx.eval(loss)
        times.append(time.perf_counter() - t0)
    
    avg_time = np.mean(times) * 1000
    print(f"{name}: {avg_time:.2f} ms/step")
    return avg_time

if __name__ == "__main__":
    base_args = Hyperparameters()
    base_args.num_layers = 12
    base_args.model_dim = 512
    base_args.num_heads = 8
    base_args.feedback_enabled = False
    base_args.capsule_enabled = False
    base_args.bigram_hash_enabled = False
    
    print("--- Speed Shootout (1024 seq_len, 512 dim, 12 layers) ---")
    
    # 1. Simple Transformer Baseline
    base_args.architecture = "transformer"
    t_time = benchmark(base_args, "Simple Transformer")
    
    # 2. Optimized Hybrid TKA-H
    base_args.architecture = "hybrid"
    base_args.feedback_enabled = True
    base_args.capsule_enabled = True
    base_args.bigram_hash_enabled = True
    base_args.koopman_enabled = True
    h_time = benchmark(base_args, "Optimized Hybrid TKA-H")
    
    improvement = (t_time - h_time) / t_time * 100
    if h_time < t_time:
        print(f"SUCCESS: Hybrid is {improvement:.1f}% FASTER than Transformer baseline.")
    else:
        print(f"FAILURE: Hybrid is {abs(improvement):.1f}% SLOWER than Transformer baseline.")

import mlx.core as mx
import time
from train_gpt_mlx import GPT, Hyperparameters

args = Hyperparameters()
args.num_layers = 8
args.model_dim = 256
args.num_heads = 4
args.feedback_enabled = True
args.ttt_enabled = False

model = GPT(args)
x = mx.random.randint(0, 8192, (1, 1024))
y = mx.random.randint(0, 8192, (1, 1024))
mx.eval(model.parameters())

print("Init done")
t0 = time.perf_counter()
logits = model(x)
mx.eval(logits)
print(f"Time: {time.perf_counter() - t0:.4f}s")

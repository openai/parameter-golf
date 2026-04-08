import mlx.core as mx
import time
from train_gpt_mlx import GPT, Hyperparameters, ce_from_logits

args = Hyperparameters()
args.num_layers = 12
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
logits, _ = model.forward_logits_with_carry(x, temperature=1.0)
loss = ce_from_logits(logits, y, reduction="mean").astype(mx.float32)
mx.eval(loss)
print(f"Time: {time.perf_counter() - t0:.4f}s")

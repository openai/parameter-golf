import mlx.core as mx
import time
from train_gpt_mlx import GPT, Hyperparameters, SplitOptimizers

args = Hyperparameters()
args.num_layers = 8
args.model_dim = 256
args.num_heads = 4
args.feedback_enabled = True
args.ttt_enabled = False

model = GPT(args)
mx.eval(model.parameters())

func = mx.compile(nn.value_and_grad(model, lambda x, y: model.loss(x, y)))

x = mx.random.randint(0, 8192, (16384 // 2, 1024))
y = mx.random.randint(0, 8192, (16384 // 2, 1024))

print("Init done, starting trace/compile")
t0 = time.perf_counter()
loss, grads = func(x, y)
mx.eval(loss, grads)
t_compile = time.perf_counter() - t0
print(f"Compile pass: {t_compile:.4f}s")

t0 = time.perf_counter()
loss, grads = func(x, y)
mx.eval(loss, grads)
t_run = time.perf_counter() - t0
print(f"Run pass: {t_run:.4f}s")

import mlx.core as mx
import mlx.nn as nn
from functools import partial

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
    def __call__(self, x):
        return self.fc(x)

class DenseTernaryMoE(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = [MLP(dim) for _ in range(num_experts)]

    def __call__(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)
        router_logits = self.router(x_flat)
        
        density = mx.mean(mx.softmax(router_logits, axis=1), axis=0) # shape (E,)
        routing_weights = mx.softmax(router_logits.astype(mx.float32), axis=1)
        topk_vals = mx.topk(routing_weights, self.top_k, axis=-1)
        threshold = mx.min(topk_vals, axis=-1, keepdims=True)
        mask = routing_weights >= threshold
        
        fraction_routed = mx.mean(mask.astype(mx.float32), axis=0)
        aux_loss = mx.mean(density * fraction_routed) * self.num_experts
        
        active_weights = mx.where(mask, routing_weights, mx.zeros_like(routing_weights))
        active_weights = active_weights / mx.sum(active_weights, axis=-1, keepdims=True)
        active_weights = active_weights.astype(x.dtype)
        
        final_output = mx.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            weight = active_weights[:, i:i+1] # (N, 1)
            out = expert(x_flat)
            final_output = final_output + out * weight
            
        return final_output.reshape(B, T, D), aux_loss

def loss_fn(model, x):
    out, aux = model(x)
    return mx.mean(out) + aux

model = DenseTernaryMoE(32, 4, 2)
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
c_loss = mx.compile(loss_and_grad_fn)
x = mx.random.normal((2, 10, 32))
v, g = c_loss(model, x)
print("Dense MOE Compile SUCCESS!", v)

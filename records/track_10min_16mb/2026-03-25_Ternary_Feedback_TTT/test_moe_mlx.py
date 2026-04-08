import mlx.core as mx
import mlx.nn as nn

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
    def __call__(self, x):
        return self.fc(x)

class TernaryMoE(nn.Module):
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
        routing_weights = mx.softmax(router_logits.astype(mx.float32), axis=1)
        
        # MLX topk returns (values, indices) maybe? Let's check docs or use topk
        topk_vals, selected_experts = mx.topk(routing_weights, self.top_k, axis=-1)
        
        routing_weights = topk_vals / mx.sum(topk_vals, axis=-1, keepdims=True)
        routing_weights = routing_weights.astype(x.dtype)
        
        final_output = mx.zeros_like(x_flat)
        aux_loss = mx.array(0.0)
        
        density = mx.mean(mx.softmax(router_logits, axis=1), axis=0) # shape (E,)
        
        for i, expert in enumerate(self.experts):
            expert_mask = (selected_experts == i) # shape (N, K)
            
            # Since a token can't select same expert twice, token_indices is unique
            # any() over K axis tells us if the token selected expert i
            token_mask = mx.any(expert_mask, axis=1) # shape (N,)
            
            # Since MLX doesn't do boolean indexing gracefully with dynamic shapes inside evaluate / compile, 
            # maybe we use where? But where processes all elements. 
            # If we don't compile, boolean indexing works.
            # But the model uses __call__ which might be mx.compiled
            # In MLX, dynamic shapes are fine IF we don't mx.compile.
            # BUT train_gpt_mlx.py uses mx.compile(loss_fn). Thus dynamic shapes (boolean indexing) will crash mx.compile!
            
            # Wait, PyTorch model compiles MoE using what? PyTorch 2.0 compiler handles dynamic shapes loosely.
            # In MLX, boolean shapes cause errors with compile. 
            # Workaround for mx.compile: compute expert for all tokens, then mask? That's dense! It negates MoE speed up.
            # Wait, MLX just added scatter. Let's print out what we can do.
            pass
            
        return final_output, aux_loss

print("Test script written")

import re
with open('train_gpt_mlx.py', 'r') as f:
    content = f.read()

# 1. Hyperparameters
params = """
    # MoE
    moe_enabled = _e("MOE_ENABLED", 0, bool)
    moe_num_experts = _e("MOE_NUM_EXPERTS", 8, int)
    moe_top_k = _e("MOE_TOP_K", 2, int)
    moe_router_aux_loss_coef = _e("MOE_ROUTER_AUX_LOSS_COEF", 0.01, float)

    # Feedback"""
content = content.replace("    # Feedback", params)

# 2. TernaryMoE class
moe_class = """
class DenseTernaryMoE(nn.Module):
    \"\"\"Sparse Ternary Mixture of Experts for MLX. 
    Computed densely to avoid dynamic shape recompilation issues,
    but evaluates fast on sparse weights by masking.\"\"\"
    def __init__(self, dim, mlp_mult, num_experts, top_k, group_size=128, activation="lrelu2", leaky_relu_slope=0.5):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = [MLP(dim, mlp_mult, group_size, activation, leaky_relu_slope) for _ in range(num_experts)]

    def __call__(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)
        router_logits = self.router(x_flat)
        
        density = mx.mean(mx.softmax(router_logits, axis=1), axis=0)
        routing_weights = mx.softmax(router_logits.astype(mx.float32), axis=1)
        
        topk_vals = mx.topk(routing_weights, self.top_k, axis=-1)
        threshold = mx.min(topk_vals, axis=-1, keepdims=True)
        mask = routing_weights >= threshold
        
        fraction_routed = mx.mean(mask.astype(mx.float32), axis=0)
        aux_loss = mx.mean(density * fraction_routed) * self.num_experts
        
        active_weights = mx.where(mask, routing_weights, mx.zeros_like(routing_weights))
        active_weights = active_weights / mx.maximum(mx.sum(active_weights, axis=-1, keepdims=True), 1e-9)
        active_weights = active_weights.astype(x.dtype)
        
        final_output = mx.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            weight = active_weights[:, i:i+1]
            out = expert(x_flat)
            final_output = final_output + out * weight
            
        return final_output.reshape(B, T, D), aux_loss

# ---------------------------------------------------------------------------
# Koopman State Space Model (Path 2: Attention-free architecture)
"""
content = content.replace("# ---------------------------------------------------------------------------\n# Koopman State Space Model (Path 2: Attention-free architecture)", moe_class)

# 3. Modify Block and KoopmanBlock __init__ and return signatures
# Block
content = re.sub(
    r"def __init__\(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,\s*group_size=128, activation=\"lrelu2\", leaky_relu_slope=0.5,\s*partial_rope_dims=0, vrl_enabled=False, ln_scale_factor=1.0, xsa=False\):",
    "def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,\n                 group_size=128, activation=\"lrelu2\", leaky_relu_slope=0.5,\n                 partial_rope_dims=0, vrl_enabled=False, ln_scale_factor=1.0, xsa=False, moe_enabled=False, moe_num_experts=8, moe_top_k=2):",
    content
)
content = content.replace("self.mlp = MLP(dim, mlp_mult, group_size=group_size, activation=activation,\n                       leaky_relu_slope=leaky_relu_slope)", 
"if moe_enabled:\n            self.mlp = DenseTernaryMoE(dim, mlp_mult, moe_num_experts, moe_top_k, group_size, activation, leaky_relu_slope)\n        else:\n            self.mlp = MLP(dim, mlp_mult, group_size=group_size, activation=activation, leaky_relu_slope=leaky_relu_slope)\n        self.moe_enabled = moe_enabled")

# KoopmanBlock
content = re.sub(
    r"def __init__\(self, dim, state_dim, mlp_mult, mixer_rank=4, conv_kernel=4,\s*decay_window=32, group_size=128, activation=\"lrelu2\",\s*leaky_relu_slope=0.5, ln_scale_factor=1.0\):",
    "def __init__(self, dim, state_dim, mlp_mult, mixer_rank=4, conv_kernel=4,\n                 decay_window=32, group_size=128, activation=\"lrelu2\",\n                 leaky_relu_slope=0.5, ln_scale_factor=1.0, moe_enabled=False, moe_num_experts=8, moe_top_k=2):",
    content
)

# 4. Modify Block and KoopmanBlock __call__ to return x, v_out, aux_loss / x, None, aux_loss
block_call = """    def __call__(self, x, x0, v0=None):
        aux_loss = mx.array(0.0)
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        normed = self.attn_norm(x)
        attn_out, v_out = self.attn(normed, v0=v0)
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        
        mlp_in = self.mlp_norm(x)
        if hasattr(self, 'moe_enabled') and self.moe_enabled:
            mlp_out, aux_loss = self.mlp(mlp_in)
        else:
            mlp_out = self.mlp(mlp_in)
            
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * mlp_out
        return x, v_out, aux_loss"""
content = re.sub(r"    def __call__\(self, x, x0, v0=None\):.*?return x, v_out", block_call, content, flags=re.DOTALL)

# Fix shared run block which calls self.mlp directly
orig_run_block = r"""            normed_mlp = rms_norm\(x\) \* ln_sf
            x = x \+ self.per_layer_mlp_scales\[layer_idx\].astype\(x.dtype\)\[None, None, :\] \* block.mlp\(normed_mlp\)
            return x, None"""
new_run_block = """            normed_mlp = rms_norm(x) * ln_sf
            if hasattr(block, 'moe_enabled') and block.moe_enabled:
                mlp_out, aux_loss = block.mlp(normed_mlp)
            else:
                mlp_out = block.mlp(normed_mlp)
                aux_loss = mx.array(0.0)
            x = x + self.per_layer_mlp_scales[layer_idx].astype(x.dtype)[None, None, :] * mlp_out
            return x, None, aux_loss"""
content = re.sub(orig_run_block, new_run_block, content, count=1)

orig_run_block_attn = r"""            normed_mlp = rms_norm\(x\) \* ln_sf
            x = x \+ self.per_layer_mlp_scales\[layer_idx\].astype\(x.dtype\)\[None, None, :\] \* block.mlp\(normed_mlp\)
            return x, v_out"""
new_run_block_attn = """            normed_mlp = rms_norm(x) * ln_sf
            if hasattr(block, 'moe_enabled') and block.moe_enabled:
                mlp_out, aux_loss = block.mlp(normed_mlp)
            else:
                mlp_out = block.mlp(normed_mlp)
                aux_loss = mx.array(0.0)
            x = x + self.per_layer_mlp_scales[layer_idx].astype(x.dtype)[None, None, :] * mlp_out
            return x, v_out, aux_loss"""
content = re.sub(orig_run_block_attn, new_run_block_attn, content, count=1)

# Modify GPT.__init__ make_block
content = content.replace("ln_scale_factor=ln_sf,\n                )", "ln_scale_factor=ln_sf,\n                    moe_enabled=args.moe_enabled, moe_num_experts=args.moe_num_experts, moe_top_k=args.moe_top_k\n                )")

content = content.replace("ln_scale_factor=ln_sf, xsa=layer_xsa,\n                )", "ln_scale_factor=ln_sf, xsa=layer_xsa,\n                    moe_enabled=args.moe_enabled, moe_num_experts=args.moe_num_experts, moe_top_k=args.moe_top_k\n                )")

# 5. Fix _decoder_pass
orig_decoder_pass = """            x, _ = self._run_block(bi, x, x0, v0=v0)"""
new_decoder_pass = """            x, _, aux_loss = self._run_block(bi, x, x0, v0=v0)
            if aux_loss is not None:
                moe_losses.append(aux_loss)"""
content = content.replace(orig_decoder_pass, new_decoder_pass)
content = content.replace("def _decoder_pass(self, x, x0, skips, sketch, v0):", "def _decoder_pass(self, x, x0, skips, sketch, v0, moe_losses):")

# 6. Fix _koopman_ssm_forward
content = content.replace("x = block(x, x0)", "if hasattr(block, 'moe_enabled') and block.moe_enabled:\n                x, _, aux = block(x, x0)\n                moe_losses.append(aux)\n            else:\n                x, _ = block(x, x0)")
content = content.replace("for i, block in enumerate(self.koopman_blocks):", "moe_losses = []\n        for i, block in enumerate(self.koopman_blocks):")
content = content.replace("return self.final_norm(x), None, [], []", "return self.final_norm(x), None, [], [], moe_losses")

# 7. __call__
content = content.replace("skips = []", "skips = []\n        moe_losses = []")
orig_call_encoder = """x, v_out = self._run_block(i, x, x0, v0=v0)"""
new_call_encoder = """x, v_out, aux_loss = self._run_block(i, x, x0, v0=v0)\n            moe_losses.append(aux_loss)"""
content = content.replace(orig_call_encoder, new_call_encoder)
content = content.replace("x = self._decoder_pass(encoded, x0, skips, sketch=sketch, v0=v0)", "x = self._decoder_pass(encoded, x0, skips, sketch=sketch, v0=v0, moe_losses=moe_losses)")
content = content.replace("return self.final_norm(x), capsule_state, consistency_losses, jepa_loss", "return self.final_norm(x), capsule_state, consistency_losses, jepa_loss, moe_losses")

with open('train_gpt_mlx.py', 'w') as f:
    f.write(content)
print("Patched successfully")

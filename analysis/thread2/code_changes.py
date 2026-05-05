"""
Concrete speed optimization code changes for parameter-golf train_gpt.py.
Each change is independent and can be tested separately.
"""

# =============================================================================
# CHANGE 1: torch.compile with reduce-overhead (CUDA graphs + autotune)
# Expected: 10-15% speedup
# Risk: Low - well-tested in PyTorch 2.7+
# Location: line 865 of train_gpt.py
# =============================================================================

# BEFORE:
# compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)

# AFTER:
compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True, mode='reduce-overhead')
# 'reduce-overhead' enables CUDA graph capture + autotuning
# This reduces kernel launch overhead which is ~30% of step time


# =============================================================================
# CHANGE 2: DDP with gradient_as_bucket_view (avoid gradient copy)
# Expected: 1-3% speedup
# Risk: Very low
# Location: line 866 of train_gpt.py
# =============================================================================

# BEFORE:
# model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)

# AFTER:
model = DDP(
    compiled_model,
    device_ids=[local_rank],
    broadcast_buffers=False,
    gradient_as_bucket_view=True,  # Avoid copying gradients to buckets
    static_graph=True,             # Enable optimizations for static computation graph
)
# static_graph=True is safe because our model shape never changes


# =============================================================================
# CHANGE 3: Muon Newton-Schulz steps 5 → 3
# Expected: Saves ~1ms/step from Muon overhead
# Risk: Low - 3 steps still provides good orthogonalization
# Location: Hyperparameters class or environment variable
# =============================================================================

# BEFORE:
# muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))

# AFTER:
muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 3))
# Or pass MUON_BACKEND_STEPS=3 as env var


# =============================================================================
# CHANGE 4: Async data prefetch with background thread
# Expected: 1-3ms/step saved (eliminates CPU-GPU pipeline stalls)
# Risk: Low
# Location: Replace DistributedTokenLoader usage in training loop
# =============================================================================

import threading
import queue

class AsyncBatchPrefetcher:
    """Prefetches next batch in a background thread while GPU computes."""
    
    def __init__(self, loader, global_tokens, seq_len, grad_accum_steps, prefetch=2):
        self.loader = loader
        self.global_tokens = global_tokens
        self.seq_len = seq_len
        self.grad_accum_steps = grad_accum_steps
        self.queue = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def _worker(self):
        while not self._stop.is_set():
            try:
                x, y = self.loader.next_batch(
                    self.global_tokens, self.seq_len, self.grad_accum_steps
                )
                self.queue.put((x, y))
            except Exception as e:
                self.queue.put(e)
                break
    
    def next_batch(self):
        result = self.queue.get()
        if isinstance(result, Exception):
            raise result
        return result
    
    def stop(self):
        self._stop.set()

# Usage in training loop:
# prefetcher = AsyncBatchPrefetcher(train_loader, args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
# x, y = prefetcher.next_batch()  # instead of train_loader.next_batch(...)


# =============================================================================
# CHANGE 5: Compiled autograd (compile backward pass too)
# Expected: 5-10% speedup
# Risk: Moderate - experimental, test carefully
# Location: Before training loop
# =============================================================================

# Add near the top of main():
import torch._dynamo
torch._dynamo.config.compiled_autograd = True

# This makes the backward pass also go through torch.compile,
# enabling the same fusion and launch overhead reduction as forward.
# May conflict with DDP - test carefully.


# =============================================================================
# CHANGE 6: Reduce validation frequency
# Expected: Saves ~2-3 seconds per val run (matters for final BPB)
# Risk: None - just less monitoring
# Location: Hyperparameters class
# =============================================================================

# BEFORE:
# val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))

# AFTER:
val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 5000))
# Or even higher - validation takes ~2s each time
# With 13780 steps, default 1000 means 14 validations = ~28s lost
# At 5000, only 3 validations = ~6s lost
# Saves ~22s = ~500 extra training steps!


# =============================================================================
# CHANGE 7: Overlap backward with optimizer via hooks
# Expected: 3-5% speedup
# Risk: High - complex implementation, subtle bugs possible
# Location: Custom Muon optimizer with backward hooks
# =============================================================================

class MuonWithOverlap(torch.optim.Optimizer):
    """Muon optimizer that starts Newton-Schulz as gradients arrive."""
    
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )
        self._ns_stream = None  # Will be created on first use
        self._processed = {}    # Track which params have been NS-processed
    
    def register_hooks(self, world_size, rank):
        """Register backward hooks that start NS computation early."""
        self._ns_stream = torch.cuda.Stream()
        
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if i % world_size == rank:
                    p.register_post_accumulate_grad_hook(
                        self._make_hook(p, group, i, world_size, rank)
                    )
    
    def _make_hook(self, param, group, idx, world_size, rank):
        def hook(p):
            # This runs as soon as this param's gradient is ready
            with torch.cuda.stream(self._ns_stream):
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=group["momentum"])
                processed = zeropower_via_newtonschulz5(g, steps=group["backend_steps"])
                processed *= max(1, g.size(0) / g.size(1)) ** 0.5
                self._processed[id(p)] = processed
        return hook
    
    @torch.no_grad()
    def step(self, closure=None):
        # Wait for all NS computations to finish
        if self._ns_stream is not None:
            self._ns_stream.synchronize()
        
        # Proceed with all_reduce and weight update as normal
        # ... (rest of original Muon.step but using self._processed)


# =============================================================================
# CHANGE 8: FP8 matmul for MLP (only beneficial for dim >= 640)
# Expected: 5-8% speedup for wider models
# Risk: Moderate - may affect convergence
# Location: MLP class forward method
# =============================================================================

class MLP_FP8(nn.Module):
    """MLP with FP8 matmul for compute-bound operations."""
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x):
        # Use FP8 for the large matmuls if dim >= 640
        if x.size(-1) >= 640:
            # Cast weights to fp8 for faster matmul
            fc_w = self.fc.weight.to(torch.float8_e4m3fn)
            h = torch.relu(F.linear(x.to(torch.float8_e4m3fn), fc_w).to(x.dtype))
            proj_w = self.proj.weight.to(torch.float8_e4m3fn)
            return F.linear((h * h).to(torch.float8_e4m3fn), proj_w).to(x.dtype)
        else:
            h = torch.relu(self.fc(x))
            return self.proj(h.square())


# =============================================================================
# PRIORITY ORDER (impact vs effort):
# 1. torch.compile mode='reduce-overhead'  [EASY,  10-15%]
# 2. VAL_LOSS_EVERY=5000                   [TRIVIAL, ~500 extra steps]
# 3. MUON_BACKEND_STEPS=3                  [TRIVIAL, ~1ms]
# 4. DDP static_graph + gradient_as_bucket [EASY,  1-3%]
# 5. compiled_autograd                     [MODERATE, 5-10%]
# 6. Async data prefetch                   [EASY,  1-3ms]
# 7. FP8 MLP (dim>=640 only)              [MODERATE, 5-8%]
# 8. Backward-optimizer overlap            [HARD,  3-5%]
# =============================================================================

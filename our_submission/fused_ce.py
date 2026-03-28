# fused_cross_entropy_triton_op.py
# Parameter Golf - V=1024 fused CE with softcap
# Based on Grok's implementation, fixed for proper autograd integration

import torch
import triton
import triton.language as tl


# ============================ TRITON KERNELS ============================

@triton.jit
def _fused_ce_fwd_kernel(
    logits_ptr, targets_ptr, loss_ptr, logsumexp_ptr,
    V: tl.constexpr, softcap: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Forward: softcap + online softmax + CE loss. One row per program."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < V

    # Load raw logits
    raw = tl.load(logits_ptr + row * V + offs, mask=mask, other=-float("inf"))

    # Apply softcap: x = softcap * tanh(x / softcap)
    if softcap > 0.0:
        sc = softcap * tl.math.tanh(raw / softcap)
    else:
        sc = raw

    # Numerically stable log-sum-exp
    max_val = tl.max(sc, axis=0)
    exp_val = tl.exp(sc - max_val)
    sum_exp = tl.sum(exp_val, axis=0)
    lse = max_val + tl.log(sum_exp)

    # Target logit (after softcap)
    target_idx = tl.load(targets_ptr + row)
    target_raw = tl.load(logits_ptr + row * V + target_idx)
    if softcap > 0.0:
        target_sc = softcap * tl.math.tanh(target_raw / softcap)
    else:
        target_sc = target_raw

    # Loss = lse - target_logit
    loss = lse - target_sc
    tl.store(loss_ptr + row, loss)
    tl.store(logsumexp_ptr + row, lse)


@triton.jit
def _fused_ce_bwd_kernel(
    logits_ptr, targets_ptr, logsumexp_ptr, grad_output_ptr, grad_logits_ptr,
    V: tl.constexpr, softcap: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward: recompute softmax from saved lse, chain rule through tanh."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < V

    # Load raw logits
    raw = tl.load(logits_ptr + row * V + offs, mask=mask, other=0.0)

    # Recompute softcapped logits
    if softcap > 0.0:
        sc = softcap * tl.math.tanh(raw / softcap)
    else:
        sc = raw

    # Recompute softmax from saved logsumexp
    lse = tl.load(logsumexp_ptr + row)
    probs = tl.exp(sc - lse)

    # grad_ce = probs - one_hot(target)
    target_idx = tl.load(targets_ptr + row)
    grad_ce = tl.where(offs == target_idx, probs - 1.0, probs)

    # Chain rule through softcap: d/dx [s*tanh(x/s)] = 1 - tanh²(x/s)
    if softcap > 0.0:
        tanh_val = tl.math.tanh(raw / softcap)
        grad_ce = grad_ce * (1.0 - tanh_val * tanh_val)

    # Scale by upstream gradient (for reduction='mean', this is 1/BT)
    grad_scale = tl.load(grad_output_ptr + row)
    grad_ce = grad_ce * grad_scale

    tl.store(grad_logits_ptr + row * V + offs, grad_ce, mask=mask)


# ============================ AUTOGRAD FUNCTION ============================
# Note: We use torch.autograd.Function here because triton_op with proper
# backward support requires PyTorch 2.8+. For 2.7, autograd.Function works
# BUT we need torch.compile to handle it. Setting allow_in_graph=True
# tells the compiler this function is safe to include in the graph.

class FusedCrossEntropyWithSoftcap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, softcap):
        BT, V = logits.shape
        loss = torch.empty(BT, dtype=torch.float32, device=logits.device)
        logsumexp = torch.empty(BT, dtype=torch.float32, device=logits.device)

        BLOCK_SIZE = triton.next_power_of_2(V)
        grid = (BT,)

        _fused_ce_fwd_kernel[grid](
            logits, targets, loss, logsumexp,
            V=V, softcap=softcap, BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(logits, targets, logsumexp)
        ctx.softcap = softcap
        ctx.V = V
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, logsumexp = ctx.saved_tensors
        BT = logits.shape[0]
        grad_logits = torch.empty_like(logits)

        BLOCK_SIZE = triton.next_power_of_2(ctx.V)
        grid = (BT,)

        _fused_ce_bwd_kernel[grid](
            logits, targets, logsumexp, grad_output, grad_logits,
            V=ctx.V, softcap=ctx.softcap, BLOCK_SIZE=BLOCK_SIZE,
        )

        return grad_logits, None, None


# Allow torch.compile to see through this function
FusedCrossEntropyWithSoftcap = torch.compiler.allow_in_graph(FusedCrossEntropyWithSoftcap)


# ============================ DROP-IN REPLACEMENT ============================

def fused_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    softcap: float = 30.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Drop-in replacement for softcap + F.cross_entropy.

    Replaces:
        logits = softcap * torch.tanh(logits / softcap)
        loss = F.cross_entropy(logits.float(), targets, reduction=reduction)

    With a single fused Triton kernel (no intermediate HBM write).
    """
    # Ensure float32 for loss computation
    loss = FusedCrossEntropyWithSoftcap.apply(logits.float(), targets, softcap)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


# ============================ TEST ============================

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"
    BT = 256
    V = 1024
    softcap = 30.0

    # Create test data
    raw_logits = torch.randn(BT, V, device=device, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, V, (BT,), device=device, dtype=torch.int64)

    # --- Baseline (what the SOTA does) ---
    raw_logits_ref = raw_logits.detach().clone().requires_grad_(True)
    softcapped_ref = softcap * torch.tanh(raw_logits_ref / softcap)
    loss_ref = torch.nn.functional.cross_entropy(softcapped_ref, targets, reduction="mean")
    loss_ref.backward()
    grad_ref = raw_logits_ref.grad.clone()

    # --- Our fused kernel ---
    raw_logits_test = raw_logits.detach().clone().requires_grad_(True)
    loss_test = fused_cross_entropy(raw_logits_test, targets, softcap=softcap, reduction="mean")
    loss_test.backward()
    grad_test = raw_logits_test.grad.clone()

    # Compare
    loss_diff = (loss_ref - loss_test).abs().item()
    grad_diff = (grad_ref - grad_test).abs().max().item()

    print(f"Baseline loss: {loss_ref.item():.8f}")
    print(f"Fused loss:    {loss_test.item():.8f}")
    print(f"Loss diff:     {loss_diff:.2e}")
    print(f"Grad max diff: {grad_diff:.2e}")

    if loss_diff < 1e-4 and grad_diff < 1e-3:
        print("PASS — Fused kernel matches baseline")
    else:
        print("FAIL — check implementation")

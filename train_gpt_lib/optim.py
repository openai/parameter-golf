from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor

from .config import CONTROL_TENSOR_NAME_PATTERNS, Hyperparameters
from .model import GPT


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        params = list(params)
        if not params:
            # Null / no-op Muon: satisfies the interface without requiring real params.
            # param_groups stays empty; step() is a no-op.
            self.defaults = dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov)
            self.state: dict = {}
            self.param_groups: list = []
            return
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# Supported optimizer strategy names.
OPTIMIZER_REGISTRY: tuple[str, ...] = (
    "muon_adam",       # Muon for matrices + Adam for rest  [default]
    "muon_adamw",      # Muon for matrices + AdamW for rest
    "muon_steps3",     # Muon (3 Newton-Schulz iters) + Adam
    "muon_steps10",    # Muon (10 Newton-Schulz iters) + Adam
    "muon_mom90",      # Muon momentum=0.90 + Adam
    "muon_mom98",      # Muon momentum=0.98 + Adam
    "adam",            # Adam everywhere (no Muon)
    "adamw",           # AdamW everywhere (weight_decay via args.adamw_weight_decay)
)


def build_optimizers(
    args: Hyperparameters,
    base_model: GPT,
) -> tuple[list[torch.optim.Optimizer], Muon]:
    opt_name = args.optimizer
    if opt_name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer={opt_name!r}. Available: {list(OPTIMIZER_REGISTRY)}")

    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    # Per-strategy Muon hyperparameter overrides.
    muon_backend_steps = args.muon_backend_steps
    muon_momentum = args.muon_momentum
    if opt_name == "muon_steps3":
        muon_backend_steps = 3
    elif opt_name == "muon_steps10":
        muon_backend_steps = 10
    elif opt_name == "muon_mom90":
        muon_momentum = 0.90
    elif opt_name == "muon_mom98":
        muon_momentum = 0.98

    use_muon = opt_name not in ("adam", "adamw")
    use_adamw_for_rest = opt_name in ("adamw", "muon_adamw")

    adam_cls_rest = torch.optim.AdamW if use_adamw_for_rest else torch.optim.Adam
    wd_rest = args.adamw_weight_decay if use_adamw_for_rest else 0.0

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr

    optimizer_tok = adam_cls_rest(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=wd_rest,
        fused=True,
    )

    if use_muon:
        optimizer_muon = Muon(
            matrix_params,
            lr=args.matrix_lr,
            momentum=muon_momentum,
            backend_steps=muon_backend_steps,
        )
        for group in optimizer_muon.param_groups:
            group["base_lr"] = args.matrix_lr
        matrix_opt: torch.optim.Optimizer = optimizer_muon
    else:
        # No Muon: put matrix params into Adam/AdamW; keep a null Muon for interface compatibility.
        optimizer_muon = Muon([], lr=0.0, momentum=0.0, backend_steps=1)
        adam_cls_mat = torch.optim.AdamW if opt_name == "adamw" else torch.optim.Adam
        wd_mat = args.adamw_weight_decay if opt_name == "adamw" else 0.0
        matrix_opt = adam_cls_mat(
            [{"params": matrix_params, "lr": args.matrix_lr, "base_lr": args.matrix_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=wd_mat,
            fused=True,
        )

    optimizer_scalar = adam_cls_rest(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=wd_rest,
        fused=True,
    )

    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, matrix_opt, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = adam_cls_rest(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=wd_rest,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    if base_model.mtp_heads is not None:
        mtp_params = [head.weight for head in base_model.mtp_heads]
        optimizer_mtp = adam_cls_rest(
            [{"params": mtp_params, "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=wd_rest,
            fused=True,
        )
        optimizers.append(optimizer_mtp)

    return optimizers, optimizer_muon

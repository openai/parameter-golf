from __future__ import annotations

import torch
from torch import Tensor


def zeropower_via_newtonschulz5(g: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    x /= x.norm() + eps
    transposed = g.size(0) > g.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        a_t = x @ x.T
        b_t = b * a_t + c * a_t @ a_t
        x = a * x + b_t @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, weight_decay: float = 0.0, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, weight_decay=weight_decay, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size() if distributed else 1
        rank = torch.distributed.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            updates_flat = torch.zeros(sum(int(p.numel()) for p in params), device=params[0].device, dtype=torch.bfloat16)
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
                torch.distributed.all_reduce(updates_flat, op=torch.distributed.ReduceOp.SUM)
            curr = 0
            for p in params:
                if weight_decay > 0:
                    p.mul_(1.0 - lr * weight_decay)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


class NorMuon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int,
        beta2: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        nesterov: bool = True,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                beta2=beta2,
                eps=eps,
                weight_decay=weight_decay,
                nesterov=nesterov,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size() if distributed else 1
        rank = torch.distributed.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            updates_flat = torch.zeros(sum(int(p.numel()) for p in params), device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    if "second_moment" not in state:
                        state["second_moment"] = torch.zeros_like(g, dtype=torch.float32)
                    buf = state["momentum_buffer"]
                    second = state["second_moment"]
                    second.mul_(beta2).addcmul_(g.float(), g.float(), value=1 - beta2)
                    normed = g / second.sqrt().add_(eps).to(dtype=g.dtype)
                    buf.mul_(momentum).add_(normed)
                    if nesterov:
                        normed = normed.add(buf, alpha=momentum)
                    normed = zeropower_via_newtonschulz5(normed, steps=backend_steps)
                    normed *= max(1, normed.size(0) / normed.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = normed.reshape(-1)
                curr += p.numel()
            if distributed:
                torch.distributed.all_reduce(updates_flat, op=torch.distributed.ReduceOp.SUM)
            curr = 0
            for p in params:
                if weight_decay > 0:
                    p.mul_(1.0 - lr * weight_decay)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

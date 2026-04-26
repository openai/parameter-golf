# Vendored from state-spaces/mamba
# Source: https://github.com/state-spaces/mamba/blob/74729d0f6d9c2096407eda5562393ab70960a3f6/mamba_ssm/ops/selective_scan_interface.py
# Function vendored: selective_scan_ref (lines 115-181 of source file)
# Commit: 74729d0f6d9c2096407eda5562393ab70960a3f6
# Fetched: 2026-04-26
# License: Apache-2.0
#   See https://github.com/state-spaces/mamba/blob/74729d0f6d9c2096407eda5562393ab70960a3f6/LICENSE
#   No NOTICE file present at upstream root (verified 2026-04-26).
# Copyright (c) 2023, Tri Dao, Albert Gu.
#
# Local modifications from upstream (this worktree has no `einops` installed,
# and the rest of the upstream file imports CUDA-only deps that don't apply
# on MPS):
#   - Vendored only the `selective_scan_ref` function. Dropped imports of
#     `mamba_ssm.utils.torch.{custom_bwd, custom_fwd}` (decorators only used
#     by the CUDA autograd Functions) and the `causal_conv1d` try/except
#     block (CUDA kernel).
#   - Replaced `from einops import rearrange, repeat` with native torch ops:
#     * `rearrange(B.float(), "... (L two) -> ... L two", two=2)`
#         → `B.float().unflatten(-1, (-1, 2))`
#     * `repeat(B, "B G N L -> B (G H) N L", H=H)`
#         → `B.repeat_interleave(H, dim=1)`
#     * `rearrange(D, "d -> d 1")`
#         → `D.unsqueeze(-1)`
# Functionally equivalent to upstream for the single-state case used in
# this worktree (Mamba-1, non-complex A, B.dim() in {2,3}, C.dim() in {2,3}).
#
# **Use this file as the correctness oracle for any custom selective-scan**
# you write in an experiment. See ../references/INDEX.md for the protocol.

import torch
import torch.nn.functional as F


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            # einops "... (L two) -> ... L two" with two=2  ⇔  unflatten last dim into (-1, 2)
            B = torch.view_as_complex(B.float().unflatten(-1, (-1, 2)))
        if is_variable_C:
            C = torch.view_as_complex(C.float().unflatten(-1, (-1, 2)))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            # einops "B G N L -> B (G H) N L" with H = dim // G  ⇔  repeat_interleave on dim 1
            B = B.repeat_interleave(dim // B.shape[1], dim=1)
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = C.repeat_interleave(dim // C.shape[1], dim=1)
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (batch dim L)
    # einops 'd -> d 1'  ⇔  D.unsqueeze(-1)
    out = y if D is None else y + u * D.unsqueeze(-1)
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)

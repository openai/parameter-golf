from __future__ import annotations
from typing import Callable

from functools import partial
from random import randrange

import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, Sequential
from torch.utils._pytree import tree_flatten, tree_unflatten

from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange, Reduce
import itertools

"""
ein notation:
b - batch
d - feature dimension
s - residual streams
t - residual streams + num branch inputs
f - number of fractions (division of feature dimension space)
v - number of views for branch input
"""

# helper functions

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def add(x, y):
    return x + y

# sinkhorn

def l1norm(t, dim):
    return F.normalize(t, p = 1, dim = dim)


def get_all_permutations(n: int):
    """
    生成所有 n × n 的排列矩阵，并按 (n!, n, n) 的形状返回
    """

    assert n >= 1, "n 必须为正整数"

    perms = list(itertools.permutations(range(n)))
    index = torch.tensor(perms, dtype=torch.long)

    eye = torch.eye(n, dtype=torch.float32)
    perm_mats = eye[index]  # (n!, n, n)

    return perm_mats

# main functions

def get_expand_reduce_stream_functions(
    num_streams,
    add_stream_embed = False,
    dim = None,
    disable = False
):
    if num_streams == 1 or disable:
        return (nn.Identity(), nn.Identity())

    if add_stream_embed:
        assert exists(dim), '`dim` must be passed into get_init_and_expand_reduce_stream_functions for returning an expansion function with stream embeddings added'

        expand_fn = StreamEmbed(num_streams, dim, expand_to_streams = True)
    else:
        expand_fn = Reduce(pattern = 'b ... -> (b s) ...', reduction = 'repeat', s = num_streams)

    reduce_fn = Reduce(pattern = '(b s) ... -> b ...', reduction = 'sum', s = num_streams)

    return expand_fn, reduce_fn

def get_init_and_expand_reduce_stream_functions(
    num_streams,
    num_fracs = 1,
    dim = None,
    add_stream_embed = False,
    disable = None,
    **kwargs
):
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = MHCLite if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams, num_fracs = num_fracs, **kwargs)
    expand_reduce_fns = get_expand_reduce_stream_functions(num_streams, add_stream_embed = add_stream_embed, dim = dim, disable = disable)

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim = dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)

# norms

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * (self.gamma + 1)

# main classes

# residual base class

class Residual(Module):
    def __init__(
        self,
        *args,
        branch: Module | None = None,
        residual_transform: Module | None = None,
        **kwargs
    ):
        super().__init__()
        self.branch = branch
        self.residual_transform = default(residual_transform, nn.Identity())

    def width_connection(
        self,
        residuals
    ):
        return residuals, residuals, dict()

    def depth_connection(
        self,
        branch_output,
        residuals,

    ):
        return branch_output + self.residual_transform(residuals)

    def decorate_branch(
        self,
        branch: Callable
    ):
        assert not exists(self.branch), 'branch was already wrapped on init'

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(
        self,
        residuals,
        *branch_args,
        **branch_kwargs
    ):

        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)

# hyper connection residual streams

perm_mats = {}

class MHCLite(Module):
    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        branch: Module | None = None,
        layer_index = None,
        channel_first = False,
        dropout = 0.,
        residual_transform: Module | None = None, # to support resnet blocks where dimension in not equal to dimension out - usually a residual conv
        add_branch_out_to_residual = True,  # will disable depth connections (weighted residual sum with beta) if set False
        num_input_views = 1,                # allow for the branch module to receive multiple input views, dimension placed on the very left (before batch)
        depth_residual_fn = add,
        num_fracs = 1,                      # https://arxiv.org/abs/2503.14125
    ):
        """
        Appendix J, Algorithm2 in - https://arxiv.org/abs/2409.19606
        """
        super().__init__()

        self.branch = branch

        # frac-connections paper - num_fracs > 1 will be the `m` in their paper https://arxiv.org/abs/2503.14125

        assert num_fracs >= 1

        self.num_fracs = num_fracs
        self.has_fracs = num_fracs > 1

        self.split_fracs = Rearrange('b ... (f d) -> b ... f d', f = num_fracs)
        self.merge_fracs = Rearrange('b ... f d -> b ... (f d)')

        assert divisible_by(dim, num_fracs), f'feature dimension ({dim}) must be divisible by the `num_fracs` ({num_fracs})'

        dim //= num_fracs # effective dim handled in dimension is feature dimension divided by num fractions

        # they used layernorm in paper, but rmsnorm is fine given what we know now

        # self.norm = RMSNorm(dim * num_residual_streams * num_fracs)

        assert num_residual_streams > 0, '`num_residual_streams` must be greater than 0'

        self.num_residual_streams = num_residual_streams
        init_residual_index = default(layer_index, randrange(num_residual_streams)) % num_residual_streams # just choose one random residual stream if layer index not given

        # handle the parameter dimensions, which may require (num_residuals x num_fractions) - generalizing hyper + frac connections

        num_residual_streams_fracs = num_residual_streams * num_fracs
        num_input_views_fracs = num_input_views * num_fracs

        self.num_fracs = num_fracs

        # width num residual streams
        self.norm = RMSNorm(dim * num_residual_streams_fracs)

        assert num_input_views >= 1
        self.num_input_views = num_input_views

        # width connection
        # ------
        # XXX MHC Lite impl. 
        # H_res is from nC to n!
        # ------
        if (num_residual_streams, "cpu") not in perm_mats:
            _perm_mats = get_all_permutations(num_residual_streams).to("cpu")
            perm_mats[(num_residual_streams, "cpu")] = _perm_mats
        perms = perm_mats[(num_residual_streams, "cpu")]

        init_alpha0 = torch.ones((num_residual_streams_fracs, num_input_views_fracs)) * -1
        init_alpha0[init_residual_index, :] = 1.
        init_alpha1 = torch.ones(len(perms) * num_fracs) * -8
        init_alpha1[0] = 0.

        # (s*v + s!)
        self.static_alpha = nn.Parameter(cat([
            init_alpha0.view(-1) , 
            init_alpha1
        ], dim = -1))


        self.dynamic_alpha_fn = nn.Parameter(
            torch.zeros(
                dim * num_residual_streams, 
                num_fracs * ( len(perms) + num_residual_streams * num_input_views )
            )
        )

        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)
        self.residual_scale = nn.Parameter(torch.ones(1) * 1e-2)

        # depth connection related (beta)

        self.add_branch_out_to_residual = add_branch_out_to_residual

        if add_branch_out_to_residual:
            beta_init = torch.ones(num_residual_streams_fracs) * -1.
            beta_init[init_residual_index] = 1.
            self.static_beta = nn.Parameter(beta_init)

            # ------ 
            # XXX
            # same here
            # ------
            dynamic_beta_shape = (
                dim * num_residual_streams,
                num_fracs * num_residual_streams
            ) # preserve backwards compat
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dynamic_beta_shape))

            self.h_post_scale = nn.Parameter(torch.ones(()) * 1e-2)

        # dropouts

        self.dropout = nn.Dropout(dropout)

        # channel first option

        self.channel_first = channel_first

        # maybe residual transform

        self.residual_transform = default(residual_transform, nn.Identity())

        # maybe custom depth connection residual function
        # this is to prepare for gating the addition of the branch outputs to the residual streams
        # needed for memory lanes a la RMT / LMM

        self.depth_residual_fn = depth_residual_fn

    def width_connection(
        self,
        residuals
    ):
        streams = self.num_residual_streams

        maybe_transformed_residuals = self.residual_transform(residuals)

        # width connection

        # handle channel first

        if self.channel_first:
            residuals = rearrange(residuals, 'b d ... -> b ... d')

        # split out fractions

        residuals = self.split_fracs(residuals)

        # split out streams

        residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s = streams)

        # norm
        normed = rearrange(residuals, 'b ... s d -> b ... (s d)', s = streams)
        # normed = F.normalize(normed, dim = -1)
        normed = self.norm(normed)

        # alpha for weighted sum of residuals going into branch
        wc_weight = normed @ self.dynamic_alpha_fn # ... f (s*v + s!)
        psize = self.num_input_views * streams
        dynamic_pre, dynamic_residual = wc_weight[..., :psize], wc_weight[..., psize:]
        static_pre  , static_residual   = self.static_alpha[:psize], self.static_alpha[psize:]

        dev = str(wc_weight.device)
        if (streams, dev) not in perm_mats:
            _perm_mats = get_all_permutations(streams).to(dev)
            perm_mats[(streams, dev)] = _perm_mats
        perms = perm_mats[(streams, dev)]
        res_coeff = self.residual_scale * dynamic_residual + static_residual
        res_coeff = torch.softmax(res_coeff, dim = -1)
        alpha_residual = einsum(res_coeff, perms, '... r, r i j-> ... i j') # (..., s, s)
        alpha_residual = self.split_fracs(alpha_residual) # (..., f, s, f, s)
        
        # (..., f, s, f, v)
        alpha_pre = self.pre_branch_scale * dynamic_pre + static_pre  # (..., s*v)
        alpha_pre = rearrange(alpha_pre, '... (f s v) -> ... s f v', v = self.num_input_views, f = self.num_fracs)
        alpha_pre = alpha_pre.sigmoid()

        # the alpha is now split and "manifold constrained" with sinkhorn and sigmoid

        alpha = cat((alpha_pre, alpha_residual), dim = -1) # (..., f, s, f, s+v)

        # beta for weights from branch output back to residual streams

        beta = None
        if self.add_branch_out_to_residual:
            dc_weight = normed @ self.dynamic_beta_fn # ... (s f)
            dc_weight = rearrange(dc_weight, '... (s f) -> ... s f', s = streams)

            dynamic_beta = dc_weight * self.h_post_scale

            static_beta = rearrange(self.static_beta, '... (s f) -> ... s f', s = streams)

            beta = dynamic_beta + static_beta
            beta = beta.sigmoid() * 2 # sigmoid * 2 for "H_post"

        mix_h = einsum(alpha, residuals, '... f1 s f2 t, ... f1 s d -> ... f2 t d')

        if self.num_input_views == 1:
            branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]
        else:
            branch_input, residuals = mix_h[..., :self.num_input_views, :], mix_h[..., self.num_input_views:, :]
            branch_input = rearrange(branch_input, 'b ... v d -> v b ... d')

        if self.channel_first:
            branch_input = rearrange(branch_input, 'b ... d -> b d ...')

        # maybe merge fractions back

        branch_input = self.merge_fracs(branch_input)
        
        residuals = rearrange(residuals, 'b ... s d -> (b s) ... d')
        if self.channel_first:
            residuals = rearrange(residuals, 'b ... d -> b d ...')
        residuals = self.merge_fracs(residuals)
        return branch_input, residuals, dict(beta = beta)

    def depth_connection(
        self,
        branch_output,
        residuals,
        *,
        beta
    ):
        assert self.add_branch_out_to_residual

        # maybe split fractions

        branch_output = self.split_fracs(branch_output)

        # 'depth' connection

        if self.channel_first:
            branch_output = rearrange(branch_output, 'b d ... -> b ... d')

        output = einsum(branch_output, beta, 'b ... f1 d, b ... f1 s f2 -> b ... f2 s d')

        output = rearrange(output, 'b ... s d -> (b s) ... d')

        # merge merge back fractions

        output = self.merge_fracs(output)

        # channel first

        if self.channel_first:
            output = rearrange(output, 'b ... d -> b d ...')

        residuals = self.depth_residual_fn(output, residuals)

        return self.dropout(residuals)

    def decorate_branch(
        self,
        branch: Callable
    ):
        assert not exists(self.branch), 'branch was already wrapped on init'

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(
        self,
        residuals,
        *branch_args,
        **branch_kwargs
    ):

        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):

            if not self.add_branch_out_to_residual:
                return branch_out

            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)

MHCLite.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
MHCLite.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)

# stream embed

class StreamEmbed(Module):
    def __init__(
        self,
        num_streams,
        dim,
        channel_first = False,
        expand_to_streams = False
    ):
        super().__init__()
        self.channel_first = channel_first
        self.num_streams = num_streams

        self.expand_to_streams = expand_to_streams
        self.stream_embed = nn.Parameter(torch.zeros(num_streams, dim))

    def forward(self, residuals):

        if self.expand_to_streams:
            residuals = repeat(residuals, 'b ... -> (b s) ...', s = self.num_streams)

        if self.channel_first:
            residuals = rearrange(residuals, '(b s) d ... -> b ... s d', s = self.num_streams)
        else:
            residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s = self.num_streams)

        residuals = residuals + self.stream_embed

        if self.channel_first:
            residuals = rearrange(residuals, 'b ... s d -> (b s) d ...', s = self.num_streams)
        else:
            residuals = rearrange(residuals, 'b ... s d -> (b s) ... d', s = self.num_streams)

        return residuals

# attention pool - taken from Enformer https://www.nature.com/articles/s41592-021-01252-x , in turn taken from somewhere else

class AttentionPoolReduceStream(Module):
    def __init__(
        self,
        num_streams,
        dim,
        channel_first = False
    ):
        super().__init__()
        self.num_streams = num_streams
        self.channel_first = channel_first

        self.to_attn_logits = nn.Linear(dim, dim, bias = False)
        self.to_attn_logits.weight.data.copy_(torch.eye(dim))

    def forward(self, residuals):

        if self.channel_first:
            residuals = rearrange(residuals, '(b s) d ... -> b ... s d', s = self.num_streams)
        else:
            residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s = self.num_streams)

        attn_logits = self.to_attn_logits(residuals)
        attn = attn_logits.softmax(dim = -2)

        residuals = reduce(residuals * attn, 'b ... s d -> b ... d', 'sum')

        if self.channel_first:
            residuals = rearrange(residuals, 'b ... d -> b d ...')

        return residuals

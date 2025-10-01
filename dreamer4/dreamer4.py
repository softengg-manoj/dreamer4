from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Parameter, Sequential, Linear, RMSNorm, Identity
from torch import cat, stack, tensor, Tensor, is_tensor

# ein related

import einx
from einops import einsum, rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# constants

LinearNoBias = partial(Linear, bias = False)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def softclamp(t, value = 50.):
    return (t / value).tanh() * value

# multi-head rmsnorm

class MultiHeadRMSNorm(Module):
    def __init__(
        self,
        dim_head,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** 0.5
        self.gamma = Parameter(torch.zeros(heads, dim_head)) # weight decay friendly

    def forward(
        self,
        x
    ):
        normed = l2norm(x)
        scale = (self.gamma + 1.) * self.scale
        return einx.multiply('... h n d, h d', normed, scale)

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        softclamp_value = 50.,
        pre_rmsnorm = True
    ):
        super().__init__()
        self.norm = RMSNorm(dim) if pre_rmsnorm else Identity()

        self.scale = dim_head ** -0.5
        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        dim_inner = dim_head * heads
        self.to_q = LinearNoBias(dim, dim_inner)
        self.to_kv = LinearNoBias(dim, dim_inner * 2)
        self.to_out = LinearNoBias(dim_inner, dim)

        # stability related

        self.q_heads_rmsnorm = MultiHeadRMSNorm(dim_head, heads = heads)
        self.k_heads_rmsnorm = MultiHeadRMSNorm(dim_head, heads = heads)

        self.softclamp_value = softclamp_value

    def forward(
        self,
        tokens,
        kv_cache = None,
        return_kv_cache = False
    ):
        tokens = self.norm(tokens)

        q, k, v = (self.to_q(tokens), *self.to_kv(tokens).chunk(2, dim = -1))

        # split heads

        q, k, v = map(self.split_heads, (q, k, v))

        # qk rmsnorm

        q = self.q_heads_rmsnorm(q)
        k = self.k_heads_rmsnorm(k)

        # caching

        if exists(kv_cache):
            ck, cv = kv_cache
            k = cat((ck, k), dim = -2)
            v = cat((cv, v), dim = -2)

        # similarity

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        # softclamping a la gemma 3

        if exists(self.softclamp_value):
            sim = softclamp(sim, self.softclamp_value)

        # scale and attention

        sim = sim * self.scale

        attn = sim.softmax(dim = -1)

        # aggregate

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # merge heads

        out = self.merge_heads(out)

        # combine heads

        out = self.to_out(out)

        if not return_kv_cache:
            return out

        return out, stack((k, v))

# feedforward

class SwiGLUFeedforward(Module):
    def __init__(
        self,
        dim,
        expansion_factor = 4,
        pre_rmsnorm = True
    ):
        super().__init__()
        self.norm = RMSNorm(dim) if pre_rmsnorm else Identity()

        dim_inner = int(dim * expansion_factor * 2 / 3)

        self.proj_in = Linear(dim, dim_inner * 2)
        self.proj_out = Linear(dim_inner, dim)

    def forward(self, x):
        x = self.norm(x)

        x, gates = self.proj_in(x).chunk(2, dim = -1)
        x = x * F.gelu(gates)

        return self.proj_out(x)

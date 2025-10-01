from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Parameter, Sequential, Linear, RMSNorm, Identity
from torch import cat, stack, arange, tensor, Tensor, is_tensor

# ein related

# b - batch
# n - sequence
# h - attention heads
# d - feature dimension
# f - frequencies (rotary)
# p - positions (3 for spacetime in this work)

import einx
from einops import einsum, rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# flex attention - but will make sure it works if it is not available
# may also end up crafting own custom flash attention kernel for this work

flex_attention = None

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass

# constants

LinearNoBias = partial(Linear, bias = False)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def softclamp(t, value = 50.):
    return (t / value).tanh() * value

# golden gate rotary - Jerry Xiong, PhD student at UIUC
# https://jerryxio.ng/posts/nd-rope/

def _phi(m):
    x = 2.
    for _ in range(10):
        x = (1. + x) ** (1. / (m + 1.))
    return x

def make_directions(n, d):
    g = _phi(d)
    alpha = (1.0 / g) ** arange(1, d + 1, dtype = torch.float64)
    i = arange(1, n + 1, dtype = torch.float64).unsqueeze(1)
    z = torch.fmod(i * alpha, 1.0)
    directions = torch.erfinv(2.0 * z - 1.0)
    directions = l2norm(directions)
    return directions.float()

class GoldenGateRoPENd(Module):
    def __init__(
        self,
        dim_pos,
        heads,
        dim_head,
        rope_min_freq = 1.,
        rope_max_freq = 10000.,
        rope_p_zero_freqs = 0., # proportion of frequencies set to 0
    ):
        super().__init__()
        assert divisible_by(dim_head, 2)

        n_freqs = dim_head // 2
        n_zero_freqs = round(rope_p_zero_freqs * n_freqs)

        omega = cat((
            torch.zeros(n_zero_freqs),
            rope_min_freq * (rope_max_freq / rope_min_freq) ** torch.linspace(0, 1, n_freqs - n_zero_freqs),
        ))

        directions = make_directions(heads * n_freqs, dim_pos)
        directions = rearrange(directions, '(h f) p -> h f p', h = heads)

        omega_expanded = rearrange(omega, 'f -> f 1')
        self.register_buffer('freqs', directions * omega_expanded)  # shape: (h, f, p)

    def forward(
        self,
        x,    # (b h n d)
        pos   # (b n p)
    ):
        dtype = x

        x, y = x.float().chunk(2, dim = -1) # (b, h, n, f)

        freqs = rearrange(self.freqs, 'h f p -> 1 h 1 f p')
        positions = rearrange(pos.float(), 'b n p -> b 1 n 1 p')

        # thetas for freqs and positions (batch, head, seq, freq)

        theta = reduce(freqs * positions, 'b h n f p -> b h n f', 'sum')

        # apply rotations

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        x_out = x * cos_theta - y * sin_theta
        y_out = x * sin_theta + y * cos_theta

        out = cat((x_out, y_out), dim = -1)
        return out.type_as(dtype)

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
        x # (b h n d)
    ):
        normed = l2norm(x)
        scale = (self.gamma + 1.) * self.scale
        return einx.multiply('... h n d, h d', normed, scale)

# masking related
# block causal mask (space fully attends within each block, while time is causal)

def flex_block_causal_mask(seq_len, block_size):

    def create_mask(_, __, qi, ki):
        q_block_index = qi // block_size
        k_block_index = ki // block_size
        return q_block_index >= k_block_index

    block_mask = create_block_mask(create_mask, B = None, H = None, Q_LEN = seq_len, KV_LEN = seq_len, _compile = True)
    return block_mask

def nonflex_block_causal_mask(seq_len, block_size, device = None):
    blocks = ceil(seq_len / block_size)

    causal_mask = torch.ones((blocks, blocks), device = device, dtype = torch.bool).tril()
    block_causal_mask = repeat(causal_mask, 'i j -> (i block_size1) (j block_size2)', block_size1 = block_size, block_size2 = block_size)

    return block_causal_mask[:seq_len, :seq_len]

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
        tokens, # (b n d)
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

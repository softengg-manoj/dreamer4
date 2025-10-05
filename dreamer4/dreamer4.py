from __future__ import annotations

import math
from math import ceil, log2
from random import random
from collections import namedtuple
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Parameter, Sequential, Linear, RMSNorm, Identity
from torch import cat, stack, arange, tensor, Tensor, is_tensor

import torchvision
from torchvision.models import VGG16_Weights

from x_mlps_pytorch import create_mlp
from x_mlps_pytorch.ensemble import Ensemble

from assoc_scan import AssocScan

from accelerate import Accelerator

# ein related

# b - batch
# n - sequence
# h - attention heads
# d - feature dimension
# f - frequencies (rotary)
# p - positions (3 for spacetime in this work)
# t - time
# g - groups of query heads to key heads (gqa)
# vc - video channels
# vh, vw - video height and width

import einx
from einops import einsum, rearrange, repeat, reduce, pack, unpack
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

TokenizerLosses = namedtuple('TokenizerLosses', ('recon', 'lpips'))

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(arr):
    return arr[0]

def divisible_by(num, den):
    return (num % den) == 0

def is_power_two(num):
    return log2(num).is_integer()

# tensor helpers

def pack_one(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return first(unpack(out, packed_shape, inv_pattern))

    return packed, inverse

def pad_at_dim(
    t,
    pad: tuple[int, int],
    dim = -1,
    value = 0.
):
    if pad == (0, 0):
        return t

    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def softclamp(t, value = 50.):
    return (t / value).tanh() * value

# loss related

class LPIPSLoss(Module):
    def __init__(
        self,
        vgg: Module | None = None,
        vgg_weights: VGG16_Weights = VGG16_Weights.DEFAULT,
        sampled_frames = 1
    ):
        super().__init__()

        if not exists(vgg):
            vgg = torchvision.models.vgg16(weights = vgg_weights)
            vgg.classifier = Sequential(*vgg.classifier[:-2])

        self.vgg = [vgg]
        self.sampled_frames = sampled_frames

    def forward(
        self,
        pred,
        data,
    ):
        batch, device, is_video = pred.shape[0], pred.device, pred.ndim == 5

        vgg, = self.vgg
        vgg = vgg.to(data.device)

        # take care of sampling random frames of the video

        if is_video:
            pred, data = tuple(rearrange(t, 'b c t ... -> b t c ...') for t in (pred, data))

            # batch randperm

            batch_randperm = torch.randn(pred.shape[:2], device = pred.device).argsort(dim = -1)
            rand_frames = batch_randperm[..., :self.sampled_frames]

            batch_arange = arange(batch, device = device)
            batch_arange = rearrange(batch_arange, '... -> ... 1')

            pred, data = tuple(t[batch_arange, rand_frames] for t in (pred, data))

            # fold sampled frames into batch

            pred, data = tuple(rearrange(t, 'b t c ... -> (b t) c ...') for t in (pred, data))

        pred_embed, embed = tuple(vgg(t) for t in (pred, data))

        return F.mse_loss(embed, pred_embed)

def ramp_weight(times, slope = 0.9, intercept = 0.1):
    # equation (8) paper, their "ramp" loss weighting
    return slope * times + intercept

# reinforcement learning related

# rewards

class SymExpTwoHot(Module):
    def __init__(
        self,
        range = (-20., 20.),
        bins = 255
    ):
        super().__init__()

        min_value, max_value = range
        values = torch.linspace(min_value, max_value, bins)
        values = values.sign() * (torch.exp(values.abs()) - 1.)

        self.num_bins = bins
        self.register_buffer('bin_values', values)

    @property
    def device(self):
        return self.bin_values.device

    def logits_to_scalar_value(
        self,
        logits # (... l)
    ):
        return einsum(logits, self.bin_values, '... l, l -> ...')

    def forward(
        self,
        values
    ):
        bin_values = self.bin_values
        min_bin_value, max_bin_value = self.bin_values[0], self.bin_values[-1]

        values, inverse_pack = pack_one(values, '*')
        num_values = values.shape[0]

        values = values.clamp(min = min_bin_value, max = max_bin_value)

        indices = torch.searchsorted(self.bin_values, values)

        # fetch the closest two indices (two-hot encoding)

        left_indices = (indices - 1).clamp(min = 0)
        right_indices = left_indices + 1

        left_indices, right_indices = tuple(rearrange(t, '... -> ... 1') for t in (left_indices, right_indices))

        # fetch the left and right values for the consecutive indices

        left_values = self.bin_values[left_indices]
        right_values = self.bin_values[right_indices]

        # calculate the left and right values by the distance to the left and right

        values = rearrange(values, '... -> ... 1')
        total_distance = right_values - left_values

        left_logit_value = (right_values - values) / total_distance
        right_logit_value = 1. - left_logit_value

        # set the left and right values (two-hot)

        encoded = torch.zeros((num_values, self.num_bins), device = self.device)

        encoded.scatter_(-1, left_indices, left_logit_value)
        encoded.scatter_(-1, right_indices, right_logit_value)

        return inverse_pack(encoded, '* l')

# generalized advantage estimate

@torch.no_grad()
def calc_gae(
    rewards,
    values,
    masks,
    gamma = 0.99,
    lam = 0.95,
    use_accelerated = None
):
    assert values.shape[-1] == rewards.shape[-1]
    use_accelerated = default(use_accelerated, rewards.is_cuda)

    values = F.pad(values, (0, 1), value = 0.)
    values, values_next = values[..., :-1], values[..., 1:]

    delta = rewards + gamma * values_next * masks - values
    gates = gamma * lam * masks

    scan = AssocScan(reverse = True, use_accelerated = use_accelerated)

    gae = scan(gates, delta)

    returns = gae + values

    return returns

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
        pos   # (b n p)
    ):

        freqs = rearrange(self.freqs, 'h f p -> h 1 f p')
        positions = rearrange(pos.float(), 'n p -> 1 n 1 p')

        # thetas for freqs and positions (batch, head, seq, freq)

        theta = reduce(freqs * positions, 'h n f p -> h n f', 'sum')

        return cat((theta, theta), dim = -1)

class Rotary1D(Module):
    def __init__(
        self,
        dim_head,
        theta = 10000.
    ):
        super().__init__()
        inv_freq = 1.0 / (theta ** (arange(0, dim_head, 2).float() / dim_head))
        self.register_buffer('inv_freq', inv_freq)

    def forward(
        self,
        seq_len
    ):
        device, dtype = self.inv_freq.device, self.inv_freq.dtype

        t = torch.arange(seq_len, device = device).type(dtype)
        freqs = einsum(t, self.inv_freq, 'i, j -> i j')

        return cat((freqs, freqs), dim = -1)


def apply_rotations(
    rotations, # (h n d) | (n d)
    t          # (b h n d)
):
    heads, dtype = t.shape[1], t.dtype
    t = t.float()

    # handle gqa for rotary

    if rotations.ndim == 3 and rotations.shape[0] < heads:
        rotary_heads = rotations.shape[0]

        assert divisible_by(heads, rotary_heads)
        groups = heads // rotary_heads
        rotations = repeat(rotations, 'h ... -> (h g) ...', g = groups)

    x1, x2 = t.chunk(2, dim = -1)
    rotated_half_t = cat((-x2, x1), dim = -1)

    # rotate in the positions

    rotated = t * rotations.cos() + rotated_half_t * rotations.sin()
    return rotated.type(dtype)

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

# naive attend

def naive_attend(
    q, k, v,
    softclamp_value = None,
    scale = None,
    causal = False,
    causal_block_size = 1,
    mask = None
):

    if not exists(scale):
        scale = q.shape[-1] ** -0.5

    # grouped query attention

    groups = q.shape[1] // k.shape[1]

    q = rearrange(q, 'b (h g) ... -> b h g ...', g = groups)

    # similarity

    sim = einsum(q, k, 'b h g i d, b h j d -> b h g i j')

    # scale and attention

    sim = sim * scale

    # softclamping a la gemma 3

    if exists(softclamp_value):
        sim = softclamp(sim, softclamp_value)

    # masking

    mask_value = -torch.finfo(sim.dtype).max

    if exists(mask):
        sim = sim.masked_fill(~mask, mask_value)

    if causal:
        is_blocked_causal = causal_block_size > 1
        i, j = sim.shape[-2:]

        if is_blocked_causal:
          i = ceil(i / causal_block_size)
          j = ceil(j / causal_block_size)

        causal_mask = torch.ones((i, j), dtype = torch.bool, device = sim.device).triu(j - i + 1)

        if causal_block_size > 1:
            causal_mask = repeat(causal_mask, 'i j -> (i b1) (j b2)', b1 = causal_block_size, b2 = causal_block_size)
            causal_mask = causal_mask[:sim.shape[-2], :sim.shape[-1]]

        sim = sim.masked_fill(causal_mask, mask_value)

    # attend

    attn = sim.softmax(dim = -1)

    # aggregate

    out = einsum(attn, v, 'b h g i j, b h j d -> b h g i d')

    # merge the groups

    return rearrange(out, 'b h g i d -> b (h g) i d')

# flex attention related and factory function for attend depending on whether on cuda + flex attention available

def block_mask_causal(block_size):

    def inner(b, h, q, k):
        bq = q // block_size
        bk = k // block_size
        return bq >= bk

    return inner

def special_token_mask(q, k, seq_len, num_tokens, special_attend_only_itself = False):
    bq = q % seq_len
    bk = k % seq_len

    is_special_start_index = seq_len - num_tokens

    q_is_special = q >= is_special_start_index
    k_is_special = k >= is_special_start_index

    if special_attend_only_itself:
        out = ~(q_is_special & ~k_is_special) # modality attends to everything, but latent can only attend to itself (proposed attention pattern for encoder of video tokenizer)
    else:
        out = ~(~q_is_special & k_is_special) # modality cannot attend to agent tokens

    return out

def block_mask_special_tokens_right(
    seq_len,
    num_tokens
):
    def inner(b, h, q, k):
        return special_token_mask(q, k, seq_len, num_tokens)
    return inner

def compose_mask(mask1, mask2):
    def inner(b, h, q, k):
        return mask1(b, h, q, k) & mask2(b, h, q, k)

    return inner

def block_mask_noop(b, h, q, k):
    return b >= 0

def score_mod_softclamp(value):
    def inner(sim, b, h, q, k):
        if not exists(value):
           return sim

        sim = sim / value
        sim = torch.tanh(sim)
        sim = sim * value
        return sim

    return inner

# factory for attend function

def get_attend_fn(
    use_flex,
    seq_len,
    k_seq_len,
    causal = False,
    causal_block_size = 1,
    softclamp_value = 50.,
    num_special_tokens = 0,             # special tokens are latents / agents
    block_size_per_special = None,      # defaults to k_seq_len
    special_attend_only_itself = False, # by default, modality only attends to itself while special sees everything, but if turned True, will be the inverse - special can only attend to itself but modality can attend everything
    device = None
):
    block_size_per_special = default(block_size_per_special, k_seq_len)

    if use_flex:
        # flex pathway

        block_mask_fn = block_mask_causal(causal_block_size) if causal else block_mask_noop

        if num_special_tokens > 0:
            special_block_mask = block_mask_special_tokens_right(block_size_per_special, num_special_tokens, special_attend_only_itself)
            block_mask_fn = compose_mask(block_mask_fn, special_block_mask)

        block_mask = create_block_mask(block_mask_fn, B = None, H = None, Q_LEN = seq_len, KV_LEN = k_seq_len)

        score_mod = score_mod_softclamp(softclamp_value)
        attend_fn = partial(flex_attention, block_mask = block_mask, score_mod = score_mod, enable_gqa = True)
    else:
        # naive pathway

        mask = None
        if num_special_tokens > 0:
            q_seq = torch.arange(seq_len, device = device)[:, None]
            k_seq = torch.arange(k_seq_len, device = device)[None, :]

            mask = special_token_mask(q_seq, k_seq, block_size_per_special, num_special_tokens, special_attend_only_itself)

        attend_fn = partial(naive_attend, causal = causal, causal_block_size = causal_block_size, mask = mask, softclamp_value = softclamp_value)

    return attend_fn

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        query_heads = None,
        heads = 8,
        pre_rmsnorm = True,
    ):
        super().__init__()
        self.norm = RMSNorm(dim) if pre_rmsnorm else Identity()

        # setup grouped query attention

        query_heads = default(query_heads, heads)
        assert query_heads >= heads and divisible_by(query_heads, heads)

        # scaling, splitting and merging of heads

        self.split_heads = Rearrange('b n (h d) -> b h n d', d = dim_head)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        dim_q_inner = dim_head * query_heads
        dim_kv_inner = dim_head * heads

        self.to_q = LinearNoBias(dim, dim_q_inner)
        self.to_kv = LinearNoBias(dim, dim_kv_inner * 2)
        self.to_out = LinearNoBias(dim_q_inner, dim)

        # stability related

        self.q_heads_rmsnorm = MultiHeadRMSNorm(dim_head, heads = query_heads)
        self.k_heads_rmsnorm = MultiHeadRMSNorm(dim_head, heads = heads)

    def forward(
        self,
        tokens, # (b n d)
        kv_cache = None,
        return_kv_cache = False,
        rotary_pos_emb = None,
        attend_fn: Callable | None = None
    ):
        tokens, inverse_packed_batch = pack_one(tokens, '* n d')

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

        # rotary

        if exists(rotary_pos_emb):
            q = apply_rotations(rotary_pos_emb, q)
            k = apply_rotations(rotary_pos_emb, k)

        # attention

        attend_fn = default(attend_fn, naive_attend)

        out = attend_fn(q, k, v)

        # merge heads

        out = self.merge_heads(out)

        # combine heads

        out = self.to_out(out)

        out = inverse_packed_batch(out)

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

# video tokenizer

class VideoTokenizer(Module):
    def __init__(
        self,
        dim,
        dim_latent,
        patch_size,
        encoder_depth = 4,
        decoder_depth = 4,
        attn_kwargs: dict = dict(),
        attn_dim_head = 64,
        attn_heads = 8,
        attn_softclamp_value = 50.,
        ff_kwargs: dict = dict(),
        decoder_pos_mlp_depth = 2,
        channels = 3,
        per_image_patch_mask_prob = (0., 0.9), # probability of patch masking appears to be per image probabilities drawn uniformly between 0. and 0.9 - if you are a phd student and think i'm mistakened, please open an issue
        lpips_loss_network: Module | None = None,
        lpips_loss_weight = 0.2,
        nd_rotary_kwargs: dict = dict(
            rope_min_freq = 1.,
            rope_max_freq = 10000.,
            rope_p_zero_freqs = 0.
        )
    ):
        super().__init__()

        self.patch_size = patch_size

        # special tokens

        self.latent_token = Parameter(torch.randn(dim) * 1e-2)

        # mae masking - Kaiming He paper from long ago

        self.per_image_patch_mask_prob = per_image_patch_mask_prob
        self.mask_token = Parameter(torch.randn(dim) * 1e-2)

        # patch and unpatch

        dim_patch = channels * patch_size ** 2

        self.patch_to_tokens = Sequential(
            Rearrange('b c t (h p1) (w p2) -> b t h w (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            Linear(dim_patch, dim)
        )

        self.tokens_to_patch = Sequential(
            Linear(dim, dim_patch),
            Rearrange('b t h w (p1 p2 c) -> b c t (h p1) (w p2)', p1 = patch_size, p2 = patch_size),
        )

        # 3d rotations

        self.spacetime_rotary = GoldenGateRoPENd(
            dim_pos = 3,
            heads = attn_heads,
            dim_head = attn_dim_head,
            **nd_rotary_kwargs
        )

        # attention related

        self.attn_softclamp_value = attn_softclamp_value

        # encoder

        encoder_layers = []

        for _ in range(encoder_depth):
            encoder_layers.append(ModuleList([
                Attention(dim = dim, heads = attn_heads, dim_head = attn_dim_head, **attn_kwargs),
                SwiGLUFeedforward(dim = dim, **ff_kwargs)
            ]))

        self.encoder_layers = ModuleList(encoder_layers)
        self.encoder_norm = RMSNorm(dim)

        # latents

        self.encoded_to_latents = Sequential(
            LinearNoBias(dim, dim_latent),
            nn.Tanh(),
        )

        self.latents_to_decoder = LinearNoBias(dim_latent, dim)

        # decoder

        # parameterize the decoder positional embeddings for MAE style training so it can be resolution agnostic

        self.to_decoder_pos_emb = create_mlp(
            dim_in = 2,
            dim = dim * 2,
            dim_out = dim,
            depth = decoder_pos_mlp_depth,
        )

        decoder_layers = []

        for _ in range(decoder_depth):
            decoder_layers.append(ModuleList([
                Attention(dim = dim, heads = attn_heads, dim_head = attn_dim_head, **attn_kwargs),
                SwiGLUFeedforward(dim = dim, **ff_kwargs)
            ]))

        self.decoder_layers = ModuleList(decoder_layers)
        self.decoder_norm = RMSNorm(dim)

        # loss related

        self.register_buffer('zero', tensor(0.), persistent = False)

        self.has_lpips_loss = lpips_loss_weight > 0.
        self.lpips_loss_weight = lpips_loss_weight

        if self.has_lpips_loss:
            self.lpips = LPIPSLoss(lpips_loss_network)

    @torch.no_grad()
    def tokenize(
        self,
        video
    ):
        self.eval()
        return self.forward(video, return_latents = True)

    def forward(
        self,
        video, # (b c t h w) 
        return_latents = False,
        mask_patches = None,
        return_all_losses = False
    ):
        batch, _, time, height, width = video.shape
        patch_size, device = self.patch_size, video.device

        assert divisible_by(height, patch_size) and divisible_by(width, patch_size)

        # to tokens

        tokens = self.patch_to_tokens(video)

        # get some dimensions

        num_patch_height, num_patch_width, _ = tokens.shape[-3:]

        # rotary positions

        positions = stack(torch.meshgrid(
            arange(time, device = device),
            arange(num_patch_height, device = device),
            arange(num_patch_width, device = device)
        ), dim = -1)

        positions = rearrange(positions, 't h w p -> t (h w) p')

        # give the latents an out of bounds position and assume the network will figure it out

        positions = pad_at_dim(positions, (0, 1), dim = -2, value = -1) # todo - make this value configurable, and ultimately craft own flash attention function where certain positions can be unrotated

        positions = rearrange(positions, 't hw p -> (t hw) p')

        rotary_pos_emb = self.spacetime_rotary(positions)

        # masking

        mask_patches = default(mask_patches, self.training)

        if mask_patches:
            min_mask_prob, max_mask_prob = self.per_image_patch_mask_prob

            mask_prob = torch.empty(tokens.shape[:2], device = tokens.device).uniform_(min_mask_prob, max_mask_prob) # (b t)

            mask_prob = repeat(mask_prob, 'b t -> b t vh vw', vh = tokens.shape[2], vw = tokens.shape[3])
            mask_patch = torch.bernoulli(mask_prob) == 1.

            tokens = einx.where('..., d, ... d', mask_patch, self.mask_token, tokens)

        # pack space

        tokens, inverse_pack_space = pack_one(tokens, 'b t * d')

        # add the latent

        latents = repeat(self.latent_token, 'd -> b t d', b = tokens.shape[0], t = tokens.shape[1])

        tokens, packed_latent_shape = pack((tokens, latents), 'b t * d')

        space_seq_len = tokens.shape[-2]

        # pack time

        tokens, inverse_pack_time = pack_one(tokens, 'b * d')

        seq_len = tokens.shape[1]

        # attend hyper parameters

        attend_kwargs = dict(
            causal = True,
            causal_block_size = space_seq_len,
            softclamp_value = self.attn_softclamp_value,
            block_size_per_special = space_seq_len,
            num_special_tokens = 1
        )

        use_flex = tokens.is_cuda and exists(flex_attention)

        # encoder attend

        # modality can only attend to itself while latents can attend to everything
        # similar to agent token in dynamics model

        encoder_attend_fn = get_attend_fn(use_flex, seq_len, seq_len, special_attend_only_itself = True)

        # encoder

        for attn, ff in self.encoder_layers:
            tokens = attn(tokens, rotary_pos_emb = rotary_pos_emb, attend_fn = encoder_attend_fn) + tokens
            tokens = ff(tokens) + tokens

        tokens = self.encoder_norm(tokens)

        # latent bottleneck

        tokens = inverse_pack_time(tokens)

        tokens, latents = unpack(tokens, packed_latent_shape, 'b t * d')

        latents = self.encoded_to_latents(latents)

        if return_latents:
            return latents

        latent_tokens = self.latents_to_decoder(latents)

        # generate decoder positional embedding and concat the latent token

        spatial_pos_height = torch.linspace(-1., 1., num_patch_height, device = device)
        spatial_pos_width = torch.linspace(-1., 1., num_patch_width, device = device)

        space_height_width_coor = stack(torch.meshgrid(spatial_pos_height, spatial_pos_width, indexing = 'ij'), dim = -1)

        decoder_pos_emb = self.to_decoder_pos_emb(space_height_width_coor)
        decoder_pos_emb = repeat(decoder_pos_emb, '... -> b t ...', b = batch, t = time)

        tokens, _ = pack((decoder_pos_emb, latent_tokens), 'b * d')

        # decoder attend

        decoder_attend_fn = get_attend_fn(use_flex, seq_len, seq_len)

        # decoder attention

        for attn, ff in self.decoder_layers:
            tokens = attn(tokens, rotary_pos_emb = rotary_pos_emb, attend_fn = decoder_attend_fn) + tokens

            tokens = ff(tokens) + tokens

        tokens = self.decoder_norm(tokens)

        # unpack time

        tokens = inverse_pack_time(tokens)

        # excise latents

        tokens = tokens[..., :-1, :]

        # unpack space

        tokens = inverse_pack_space(tokens)

        # project back to patches

        recon_video = self.tokens_to_patch(tokens)

        # losses

        recon_loss = F.mse_loss(video, recon_video)

        lpips_loss = self.zero

        if self.has_lpips_loss:
            lpips_loss = self.lpips(video, recon_video)

        # losses

        total_loss = (
            recon_loss +
            lpips_loss * self.lpips_loss_weight
        )

        if not return_all_losses:
            return total_loss

        losses = (recon_loss, lpips_loss)

        return total_loss, TokenizerLosses(losses)

# dynamics model, axial space-time transformer

class DynamicsModel(Module):
    def __init__(
        self,
        dim,
        dim_latent,
        video_tokenizer: VideoTokenizer | None = None,
        max_steps = 64,            # K_max in paper
        num_spatial_tokens = 32,   # latents were projected into spatial tokens, and presumably pooled back for the final prediction (or one special one does the x-prediction)
        num_register_tokens = 8,   # they claim register tokens led to better temporal consistency
        num_tasks = 0,
        depth = 4,
        pred_orig_latent = True,   # directly predicting the original x0 data yield better results, rather than velocity (x-space vs v-space)
        time_block_every = 4,      # every 4th block is time
        attn_kwargs: dict = dict(
            heads = 8,
        ),
        attn_dim_head = 64,
        attn_softclamp_value = 50.,
        ff_kwargs: dict = dict(),
        loss_weight_fn: Callable = ramp_weight,
        num_future_predictions = 8,         # they do multi-token prediction of 8 steps forward
        prob_no_shortcut_train = None       # probability of no shortcut training, defaults to 1 / num_step_sizes
    ):
        super().__init__()

        # can accept raw video if tokenizer is passed in

        self.video_tokenizer = video_tokenizer

        # spatial and register tokens

        self.latents_to_spatial_tokens = Sequential(
            Linear(dim_latent, dim * num_spatial_tokens),
            Rearrange('... (tokens d) -> ... tokens d', tokens = num_spatial_tokens)
        )

        self.register_tokens = Parameter(torch.randn(num_register_tokens, dim) * 1e-2)

        # signal and step sizes

        assert divisible_by(dim, 2)
        dim_half = dim // 2

        assert is_power_two(max_steps), '`max_steps` must be a power of 2'
        self.max_steps = max_steps
        self.num_step_sizes_log2 = int(log2(max_steps))

        self.signal_levels_embed = nn.Embedding(max_steps, dim_half)
        self.step_size_embed = nn.Embedding(self.num_step_sizes_log2, dim_half) # power of 2, so 1/1, 1/2, 1/4, 1/8 ... 1/Kmax

        self.prob_no_shortcut_train = default(prob_no_shortcut_train, self.num_step_sizes_log2 ** -1.)

        # loss related

        self.pred_orig_latent = pred_orig_latent # x-space or v-space
        self.loss_weight_fn = loss_weight_fn

        # reinforcement related

        # they sum all the actions into a single token

        self.action_learned_embed = Parameter(torch.randn(dim) * 1e-2)

        self.num_tasks = num_tasks
        self.task_embed = nn.Embedding(num_tasks, dim)

        # calculate "space" seq len

        self.space_seq_len = (
            1    # action / agent token
            + 1  # signal + step
            + num_register_tokens
            + num_spatial_tokens
        )

        # attention

        self.attn_softclamp_value = attn_softclamp_value

        # time rotary embedding

        self.time_rotary = Rotary1D(attn_dim_head)

        # transformer

        layers = []
        is_time = []

        for i in range(depth):
            layer_index = i + 1

            is_time_block = divisible_by(layer_index, time_block_every)
            is_time.append(is_time_block)

            rearrange_to_attend = Rearrange('b t s d -> b s t d') if is_time_block else Identity()
            rearrange_from_attend = Rearrange('b s t d -> b t s d') if is_time_block else Identity()

            layers.append(ModuleList([
                rearrange_to_attend,
                rearrange_from_attend,
                Attention(dim = dim, dim_head = attn_dim_head, **attn_kwargs),
                SwiGLUFeedforward(dim = dim, **ff_kwargs)
            ]))

        self.layers = ModuleList(layers)
        self.is_time = is_time

        # to prediction

        self.to_pred = Sequential(
            RMSNorm(dim),
            Linear(dim, dim_latent)
        )

    def parameter(self):
        params = super().parameters()

        if not exists(self.video_tokenizer):
            return params

        return list(set(params) - set(self.video_tokenizer.parameters()))

    def forward(
        self,
        *,
        video = None,
        latents = None,             # (b t d)
        signal_levels = None,       # (b t)
        step_sizes_log2 = None,     # (b)
        tasks = None,               # (b)
        return_pred_only = False
    ):
        # handle video or latents

        assert exists(video) ^ exists(latents)

        if exists(video):
            assert exists(self.video_tokenizer), 'video_tokenizer must be passed in if training from raw video on dynamics model'

            latents = self.video_tokenizer.tokenize(video)

        batch, time, device = *latents.shape[:2], latents.device

        # flow related

        assert not (exists(signal_levels) ^ exists(step_sizes_log2))

        # if neither signal levels or step sizes passed in
        # generate them randomly for training

        no_shortcut_train = random() < self.prob_no_shortcut_train

        if no_shortcut_train:
            # if no shortcut training, step sizes are just 1 and noising is all steps, where each step is 1 / d_min
            # in original shortcut paper, they actually set d = 0 for some reason, look into that later, as there is no mention in the dreamer paper of doing this

            step_sizes_log2 = torch.zeros((batch,), device = device).long() # zero because zero is equivalent to step size of 1
            signal_levels = torch.randint(0, self.max_steps, (batch, time), device = device)
        else:

            # now we follow eq (4)

            step_sizes_log2 = torch.randint(1, self.num_step_sizes_log2, (batch,), device = device)
            num_step_sizes = 2 ** step_sizes_log2

            signal_levels = torch.randint(0, self.max_steps, (batch, time)) // num_step_sizes[:, None] * num_step_sizes[:, None] # times are discretized to step sizes

        # get the noise

        noise = torch.randn_like(latents)

        # times is from 0 to 1

        times = rearrange(signal_levels.float() / self.max_steps, 'b t -> b t 1')

        # noise from 0 as noise to 1 as data

        noised_latents = noise.lerp(latents, times)

        # reinforcementnet learning related

        agent_tokens = repeat(self.action_learned_embed, 'd -> b d', b = batch)

        if exists(tasks):
            assert self.num_tasks > 0

            task_embeds = self.task_embed(tasks)
            agent_tokens = agent_tokens + task_embeds

        # main function, needs to be defined as such for shortcut training - additional calls for consistency loss

        def get_prediction(noised_latents, signal_levels, step_sizes_log2, agent_tokens):
            # latents to spatial tokens

            space_tokens = self.latents_to_spatial_tokens(noised_latents)

            # pack to tokens
            # [signal + step size embed] [latent space tokens] [register] [actions / agent]

            registers = repeat(self.register_tokens, 's d -> b t s d', b = batch, t = time)

            # determine signal + step size embed for their diffusion forcing + shortcut

            signal_embed = self.signal_levels_embed(signal_levels)

            step_size_embed = self.step_size_embed(step_sizes_log2)
            step_size_embed = repeat(step_size_embed, 'b ... -> b t ...', t = time)

            flow_token = cat((signal_embed, step_size_embed), dim = -1)
            flow_token = rearrange(flow_token, 'b t d -> b t d')

            # handle agent tokens w/ actions and task embeds

            agent_tokens = repeat(agent_tokens, 'b d -> b t d', t = time)

            # pack to tokens for attending

            tokens, packed_tokens_shape = pack([flow_token, space_tokens, registers, agent_tokens], 'b t * d')

            # attend functions for space and time

            seq_len = tokens.shape[1]

            use_flex = exists(flex_attention) and tokens.is_cuda

            attend_kwargs = dict(use_flex = use_flex, softclamp_value = self.attn_softclamp_value, device = device)

            space_attend = get_attend_fn(causal = False, seq_len = self.space_seq_len, k_seq_len = self.space_seq_len, num_special_tokens = 1, **attend_kwargs) # space has an agent token on the right-hand side for reinforcement learning - cannot be attended to by modality

            time_attend = get_attend_fn(causal = True, seq_len = time, k_seq_len = time, **attend_kwargs)

            # rotary

            rotary_pos_emb = self.time_rotary(time)

            # attention

            for (pre_attn_rearrange, post_attn_rearrange, attn, ff), layer_is_time in zip(self.layers, self.is_time):

                tokens = pre_attn_rearrange(tokens)

                # when is a axial time attention block, should be causal

                attend_fn = time_attend if layer_is_time else space_attend

                layer_rotary_pos_emb = rotary_pos_emb if layer_is_time else None

                # attention layer

                tokens = attn(tokens, rotary_pos_emb = layer_rotary_pos_emb, attend_fn = attend_fn) + tokens

                tokens = post_attn_rearrange(tokens)

                # feedforward layer

                tokens = ff(tokens) + tokens

            # unpack

            flow_token, space_tokens, register_tokens, agent_tokens = unpack(tokens, packed_tokens_shape, 'b t * d')

            # pooling

            pooled = reduce(space_tokens, 'b t s d -> b t d', 'mean')

            pred = self.to_pred(pooled)

            return pred

        # forward the network

        pred = get_prediction(noised_latents, signal_levels, step_sizes_log2, agent_tokens)

        if return_pred_only:
            return pred

        # determine the target for the loss

        pred_target = None

        is_x_space = self.pred_orig_latent
        is_v_space_pred = not self.pred_orig_latent

        maybe_shortcut_loss_weight = 1.

        if no_shortcut_train:

            # allow for original velocity pred
            # x-space as in paper is in else clause

            if is_v_space_pred:
                pred_target = flow = latents - noise
            else:
                pred_target = latents
        else:
            # shortcut training - Frans et al. https://arxiv.org/abs/2410.12557

            # basically a consistency loss where you ensure quantity of two half steps equals one step
            # dreamer then makes it works for x-space with some math

            get_prediction_no_grad = torch.no_grad()(get_prediction)

            step_sizes_log2_minus_one = step_sizes_log2 - 1 # which equals d / 2
            half_step_size = 2 ** step_sizes_log2_minus_one

            first_step_pred = get_prediction_no_grad(noised_latents, signal_levels, step_sizes_log2_minus_one, agent_tokens)

            # first derive b'

            if is_v_space_pred:
                first_step_pred_flow = first_step_pred
            else:
                first_times = signal_levels[..., None].float() / self.max_steps
                first_step_pred_flow = (first_step_pred - noised_latents) / (1. - first_times)

            # take a half step

            denoised_latent = noised_latents + first_step_pred_flow * (half_step_size[:, None, None] / self.max_steps)

            # get second prediction for b''

            second_step_pred = get_prediction_no_grad(denoised_latent, signal_levels + half_step_size[:, None], step_sizes_log2_minus_one, agent_tokens)

            if is_v_space_pred:
                second_step_pred_flow = second_step_pred
            else:
                second_times = signal_levels[..., None].float() / self.max_steps
                second_step_pred_flow = (second_step_pred - denoised_latent) / (1. - second_times)

            # pred target is sg(b' + b'') / 2

            pred_target = (first_step_pred_flow + second_step_pred_flow).detach() / 2

            # need to convert x-space to v-space

            if is_x_space:
                pred = (pred - noised_latents) / (1. - first_times)
                maybe_shortcut_loss_weight = (1. - first_times) ** 2

        # mse loss

        losses = F.mse_loss(pred, pred_target, reduction = 'none')

        losses = losses * maybe_shortcut_loss_weight # handle the (1-t)^2 in eq(7)

        # loss weighting with their ramp function

        if exists(self.loss_weight_fn):
            loss_weight = self.loss_weight_fn(times)
            losses = losses * loss_weight

        return losses.mean()

# dreamer

class Dreamer(Module):
    def __init__(
        self,
        video_tokenizer: VideoTokenizer,
        dynamics_model: DynamicsModel,
        discount_factor = 0.9995
    ):
        super().__init__()

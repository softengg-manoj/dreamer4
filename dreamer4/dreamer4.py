from __future__ import annotations

import math
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Parameter, Sequential, Linear, RMSNorm, Identity
from torch import cat, stack, arange, tensor, Tensor, is_tensor

from x_mlps_pytorch import create_mlp

from accelerate import Accelerator

# ein related

# b - batch
# n - sequence
# h - attention heads
# d - feature dimension
# f - frequencies (rotary)
# p - positions (3 for spacetime in this work)
# t - time
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

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(arr):
    return arr[0]

def divisible_by(num, den):
    return (num % den) == 0

def pack_one(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return first(unpack(out, packed_shape, inv_pattern))

    return packed, inverse

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
        pos   # (b n p)
    ):

        freqs = rearrange(self.freqs, 'h f p -> 1 h 1 f p')
        positions = rearrange(pos.float(), 'b n p -> b 1 n 1 p')

        # thetas for freqs and positions (batch, head, seq, freq)

        theta = reduce(freqs * positions, 'b h n f p -> b h n f', 'sum')

        return theta

def apply_rotations(
    theta # (b h n f)
):
    dtype = x

    x, y = rearrange(x.float(), '... (split d) -> split ... d', split = 2) # (b, h, n, f)

    # apply rotations

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    x_out = x * cos_theta - y * sin_theta
    y_out = x * sin_theta + y * cos_theta

    out = rearrange([x_out, y_out], 'split ... d -> ... (split d)')
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

def flex_block_causal_mask(
    seq_len,
    block_size,
    num_special_tokens = 0,
    prevent_modality_to_special = False, # encoder of tokenizer as well as (perhaps crucially) the dynamics model
    prevent_special_to_modality = False  # decoder of tokenizer
):
    assert num_special_tokens <= block_size

    # assume special tokens (either latent or agent tokens) are placed at the right hand side
    # so [modality] [latents | agent]

    def create_mask(_, __, qi, ki):
        q_block_index = qi // block_size
        k_block_index = ki // block_size

        special_token_index_start = block_size - num_special_tokens

        q_is_special = (qi % block_size) >= special_token_index_start
        k_is_special = (ki % block_size) >= special_token_index_start

        causal_mask = q_block_index >= k_block_index

        if prevent_modality_to_special:
            causal_mask = causal_mask & ~(q_is_special & ~k_is_special)

        if prevent_special_to_modality:
            causal_mask = causal_mask & ~(~q_is_special & k_is_special)

        return causal_mask

    block_mask = create_block_mask(create_mask, B = None, H = None, Q_LEN = seq_len, KV_LEN = seq_len, _compile = True)
    return block_mask

# for softclamping with flex attention

def softclamp_score_mod(value):

    def inner(attn_logits, b, h, qi, ki):
        attn_logits = attn_logits / value
        attn_logits = torch.tanh(attn_logits)
        attn_logits = attn_logits * value
        return attn_logits

    return inner

# todo - reuse the inner function from flex attn above with broadcasting

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
        attn_kwargs: dict = dict(
            dim_head = 64,
            heads = 8,
        ),
        ff_kwargs: dict = dict(),
        channels = 3,
        per_image_patch_mask_prob = (0., 0.9), # probability of patch masking appears to be per image probabilities drawn uniformly between 0. and 0.9 - if you are a phd student and think i'm mistakened, please open an issue
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

        # encoder

        encoder_layers = []

        for _ in range(encoder_depth):
            encoder_layers.append(ModuleList([
                Attention(dim = dim, **attn_kwargs),
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

        decoder_layers = []

        for _ in range(decoder_depth):
            decoder_layers.append(ModuleList([
                Attention(dim = dim, **attn_kwargs),
                SwiGLUFeedforward(dim = dim, **ff_kwargs)
            ]))

        self.decoder_layers = ModuleList(decoder_layers)
        self.decoder_norm = RMSNorm(dim)

    def forward(
        self,
        video, # (b c t h w) 
        return_latents = False,
        mask_patches = None
    ):
        patch_size = self.patch_size

        *_, height, width = video.shape

        assert divisible_by(height, patch_size) and divisible_by(width, patch_size)

        # to tokens

        tokens = self.patch_to_tokens(video)

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

        latents = repeat(self.latent_token, 'd -> b t 1 d', b = tokens.shape[0], t = tokens.shape[1])

        tokens = cat((tokens, latents), dim = -2)

        # pack time

        tokens, inverse_pack_time = pack_one(tokens, 'b * d')

        # encoder

        for attn, ff in self.encoder_layers:
            tokens = attn(tokens) + tokens
            tokens = ff(tokens) + tokens

        tokens = self.encoder_norm(tokens)

        # latent bottleneck

        latents = self.encoded_to_latents(tokens)

        if return_latents:
            latents = inverse_pack_time(latents)
            return latents[..., -1, :]

        tokens = self.latents_to_decoder(latents)

        # decoder

        for attn, ff in self.decoder_layers:
            tokens = attn(tokens) + tokens
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

        return recon_loss

# dynamics model, axial space-time transformer

class DynamicsModel(Module):
    def __init__(
        self,
        dim,
        dim_latent,
        num_signal_levels = 500,
        num_step_sizes = 32,
        num_spatial_tokens = 32,        # latents were projected into spatial tokens, and presumably pooled back for the final prediction (or one special one does the x-prediction)
        num_register_tokens = 8,        # they claim register tokens led to better temporal consistency
        depth = 4,
        pred_is_clean_latents = True,   # directly predicting the original x0 data yield better results, rather than velocity (x-space vs v-space)
        time_block_every = 4,           # every 4th block is time
        attn_kwargs: dict = dict(
            dim_head = 64,
            heads = 8,
        ),
        ff_kwargs: dict = dict()
    ):
        super().__init__()

        # spatial and register tokens

        self.latents_to_spatial_tokens = Sequential(
            Linear(dim_latent, dim * num_spatial_tokens),
            Rearrange('... (tokens d) -> ... tokens d', tokens = num_spatial_tokens)
        )

        self.register_tokens = Parameter(torch.randn(num_register_tokens, dim) * 1e-2)

        # signal and step sizes

        assert divisible_by(dim, 2)
        dim_half = dim // 2

        self.num_signal_levels = num_signal_levels
        self.num_step_sizes = num_step_sizes

        self.signal_levels_embed = nn.Embedding(num_signal_levels, dim_half)
        self.step_sizes_embed = nn.Embedding(num_step_sizes, dim_half)

        self.pred_is_clean_latents = pred_is_clean_latents

        # they sum all the actions into a single token

        self.action_learned_embed = Parameter(torch.randn(dim) * 1e-2)

        # transformer

        layers = []

        for i in range(depth):
            layer_index = i + 1
            is_time_block = divisible_by(layer_index, time_block_every)

            rearrange_to_attend = Rearrange('b t s d -> b s t d') if is_time_block else Identity()
            rearrange_from_attend = Rearrange('b s t d -> b t s d') if is_time_block else Identity()

            layers.append(ModuleList([
                rearrange_to_attend,
                rearrange_from_attend,
                Attention(dim = dim, **attn_kwargs),
                SwiGLUFeedforward(dim = dim, **ff_kwargs)
            ]))

        self.layers = ModuleList(layers)

        # to prediction

        self.to_pred = Sequential(
            RMSNorm(dim),
            Linear(dim, dim_latent)
        )

    def forward(
        self,
        latents,                    # (b t d)
        signal_levels = None,       # (b t)
        step_sizes = None           # (b t)
    ):

        assert not (exists(signal_levels) ^ exists(step_sizes))

        flow_matching = exists(signal_levels)

        # flow matching if `signal_levels` passed in

        if flow_matching:

            noise = torch.randn_like(latents)

            times = signal_levels.float() / self.num_signal_levels

            times = rearrange(times, 'b t -> b t 1')

            flow = latents - noise

            latents = noise.lerp(latents, times)

            noised_latents = latents

        # latents to spatial tokens

        space_tokens = self.latents_to_spatial_tokens(latents)

        # pack to tokens
        # [signal + step size embed] [latent space tokens] [register] [actions / agent]

        registers = repeat(self.register_tokens, 's d -> b t s d', b = latents.shape[0], t = latents.shape[1])

        agent_token = repeat(self.action_learned_embed, 'd -> b t d', b = latents.shape[0], t = latents.shape[1])

        # determine signal + step size embed for their diffusion forcing + shortcut

        if exists(signal_levels):
            signal_embed = self.signal_levels_embed(signal_levels)
            step_size_embed = self.step_sizes_embed(step_sizes)

            flow_token = cat((signal_embed, step_size_embed), dim = -1)
            flow_token = rearrange(flow_token, 'b t d -> b t d')

        else:
            flow_token = registers[..., 0:0, :]

        # pack to tokens for attending

        tokens, packed_tokens_shape = pack([flow_token, space_tokens, registers, agent_token], 'b t * d')

        # attention

        for pre_attn_rearrange, post_attn_rearrange, attn, ff in self.layers:

            tokens = pre_attn_rearrange(tokens)

            tokens = attn(tokens) + tokens

            tokens = post_attn_rearrange(tokens)

            tokens = ff(tokens) + tokens

        # unpack

        flow_token, space_tokens, register_tokens, agent_token = unpack(tokens, packed_tokens_shape, 'b t * d')

        # pooling

        pooled = reduce(space_tokens, 'b t s d -> b t d', 'mean')

        pred = self.to_pred(pooled)

        if not flow_matching:
            return pred

        # x-space vs v-space

        if self.pred_is_clean_latents:
            denoised_latent = pred
            pred_flow = (denoised_latent - noised_latents) / (1. - times)
        else:
            pred_flow = pred

        return F.mse_loss(pred_flow, flow)

# dreamer

class Dreamer(Module):
    def __init__(
        self,
        video_tokenizer: VideoTokenizer,
        dynamics_model: DynamicsModel
    ):
        super().__init__()

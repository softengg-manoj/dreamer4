import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, RMSNorm, Identity
from torch import cat, stack, tensor, Tensor, is_tensor

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
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

    def forward(
        self,
        tokens,
        kv_cache = None,
        return_kv_cache = False
    ):
        tokens = self.norm(tokens)

        q, k, v = (self.to_q(tokens), *self.to_kv(tokens).chunk(2, dim = -1))

        q, k, v = map(self.split_heads, (q, k, v))

        if exists(kv_cache):
            ck, cv = kv_cache
            k = cat((ck, k), dim = -2)
            v = cat((cv, v), dim = -2)

        q = q * self.scale

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)

        out = self.to_out(out)

        if not return_kv_cache:
            return out

        return out, stack((k, v))

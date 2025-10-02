import pytest
import torch

def test_attn():
    from dreamer4.dreamer4 import Attention

    x = torch.randn(1, 1024, 512)
    attn = Attention(512)

    assert attn(x).shape == x.shape

def test_ff():
    from dreamer4.dreamer4 import SwiGLUFeedforward
    x = torch.randn(1, 1024, 512)
    ff = SwiGLUFeedforward(512)

    assert ff(x).shape == x.shape

def test_tokenizer():
    from dreamer4.dreamer4 import VideoTokenizer

    tokenizer = VideoTokenizer(512, dim_latent = 32, patch_size = 16)
    x = torch.randn(1, 3, 16, 256, 256)

    loss = tokenizer(x)
    assert loss.numel() == 1

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

import pytest
import torch

def test_e2e():
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsModel

    tokenizer = VideoTokenizer(512, dim_latent = 32, patch_size = 32)
    x = torch.randn(1, 3, 4, 256, 256)

    loss = tokenizer(x)
    assert loss.numel() == 1

    latents = tokenizer(x, return_latents = True)
    assert latents.shape[-1] == 32

    dynamics = DynamicsModel(512, dim_latent = 32)
    pred = dynamics(latents)
    assert pred.shape == latents.shape

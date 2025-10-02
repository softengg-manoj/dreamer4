import pytest
import torch

def test_e2e():
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsModel

    tokenizer = VideoTokenizer(512, dim_latent = 32, patch_size = 32)
    x = torch.randn(2, 3, 4, 256, 256)

    loss = tokenizer(x)
    assert loss.numel() == 1

    latents = tokenizer(x, return_latents = True)
    assert latents.shape[-1] == 32

    dynamics = DynamicsModel(512, dim_latent = 32, num_signal_levels = 500, num_step_sizes = 32)

    signal_levels = torch.randint(0, 500, (2, 4))
    step_sizes = torch.randint(0, 32, (2, 4))

    pred = dynamics(latents, signal_levels = signal_levels, step_sizes = step_sizes)
    assert pred.shape == latents.shape

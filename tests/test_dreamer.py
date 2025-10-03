import pytest
param = pytest.mark.parametrize
import torch

@param('pred_orig_latent', (False, True))
def test_e2e(
    pred_orig_latent
):
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsModel

    tokenizer = VideoTokenizer(512, dim_latent = 32, patch_size = 32)
    x = torch.randn(2, 3, 4, 256, 256)

    loss = tokenizer(x)
    assert loss.numel() == 1

    latents = tokenizer(x, return_latents = True)
    assert latents.shape[-1] == 32

    dynamics = DynamicsModel(512, dim_latent = 32, num_signal_levels = 500, num_step_sizes = 32, pred_orig_latent = pred_orig_latent)

    signal_levels = torch.randint(0, 500, (2, 4))
    step_sizes = torch.randint(0, 32, (2, 4))

    flow_loss = dynamics(latents, signal_levels = signal_levels, step_sizes = step_sizes)
    assert flow_loss.numel() == 1

def test_symexp_two_hot():
    import torch
    from dreamer4.dreamer4 import SymExpTwoHot

    two_hot_encoder = SymExpTwoHot((-3., 3.), 20)
    values = torch.randn((10))

    encoded = two_hot_encoder(values)
    recon_values = two_hot_encoder.logits_to_scalar_value(encoded)

    assert torch.allclose(recon_values, values, atol = 1e-6)

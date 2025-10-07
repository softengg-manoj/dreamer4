import pytest
param = pytest.mark.parametrize
import torch

@param('pred_orig_latent', (False, True))
@param('grouped_query_attn', (False, True))
@param('dynamics_with_video_input', (False, True))
@param('prob_no_shortcut_train', (None, 0., 1.))
@param('add_task_embeds', (False, True))
@param('num_spatial_tokens', (2, 8))
@param('signal_and_step_passed_in', (False, True))
@param('add_reward_embed_to_agent_token', (False, True))
def test_e2e(
    pred_orig_latent,
    grouped_query_attn,
    dynamics_with_video_input,
    prob_no_shortcut_train,
    add_task_embeds,
    num_spatial_tokens,
    signal_and_step_passed_in,
    add_reward_embed_to_agent_token
):
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsModel

    tokenizer = VideoTokenizer(
        16,
        encoder_depth = 1,
        decoder_depth = 1,
        dim_latent = 16,
        patch_size = 32,
        attn_dim_head = 16,
        num_latent_tokens = 4
    )

    video = torch.randn(2, 3, 4, 256, 256)

    loss = tokenizer(video)
    assert loss.numel() == 1

    latents = tokenizer(video, return_latents = True)
    assert latents.shape[-1] == 16

    recon = tokenizer.decode(latents, 256, 256)
    assert recon.shape == video.shape

    query_heads, heads = (16, 4) if grouped_query_attn else (8, 8)

    dynamics = DynamicsModel(
        dim = 16,
        video_tokenizer = tokenizer,
        dim_latent = 16,
        max_steps = 64,
        num_tasks = 4,
        num_latent_tokens = 4,
        depth = 4,
        num_spatial_tokens = num_spatial_tokens,
        pred_orig_latent = pred_orig_latent,
        attn_dim_head = 16,
        attn_kwargs = dict(
            heads = heads,
            query_heads = query_heads,
        ),
        prob_no_shortcut_train = prob_no_shortcut_train,
        add_reward_embed_to_agent_token = add_reward_embed_to_agent_token
    )

    signal_levels = step_sizes_log2 = None

    if signal_and_step_passed_in:
        signal_levels = torch.randint(0, 32, (2, 4))
        step_sizes_log2 = torch.randint(1, 5, (2,))

    if dynamics_with_video_input:
        dynamics_input = dict(video = video)
    else:
        dynamics_input = dict(latents = latents)

    tasks = None
    if add_task_embeds:
        tasks = torch.randint(0, 4, (2,))

    flow_loss = dynamics(
        **dynamics_input,
        tasks = tasks,
        signal_levels = signal_levels,
        step_sizes_log2 = step_sizes_log2
    )

    assert flow_loss.numel() == 1

    # generating

    generated_video = dynamics.generate(
        time_steps = 10,
        image_height = 128,
        image_width = 128,
        batch_size = 2
    )

    assert generated_video.shape == (2, 3, 10, 128, 128)

    # rl

    rewards = torch.randn((2, 4)) * 100.

    flow_loss = dynamics(
        **dynamics_input,
        tasks = tasks,
        rewards = rewards
    )

def test_symexp_two_hot():
    import torch
    from dreamer4.dreamer4 import SymExpTwoHot

    two_hot_encoder = SymExpTwoHot(
        (-3., 3.),
        num_bins = 20,
        learned_embedding = True,
        dim_embed = 512
    )

    values = torch.randn((10))

    two_hot_encoded = two_hot_encoder(values)
    recon_values = two_hot_encoder.bins_to_scalar_value(two_hot_encoded)

    assert torch.allclose(recon_values, values, atol = 1e-6)

    reward_embeds = two_hot_encoder.embed(two_hot_encoded)
    assert reward_embeds.shape == (10, 512)

@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'no cuda')
@param('causal', (False, True))
@param('softclamp_value', (50., None))
@param('num_agent_tokens', (0, 1))
@param('causal_block_size', (1, 8))
@param('block_size_per_special', (1, 8))
@param('special_attend_only_itself', (False, True))
def test_attend_factory(
    causal,
    softclamp_value,
    num_agent_tokens,
    causal_block_size,
    block_size_per_special,
    special_attend_only_itself
):

    from dreamer4.dreamer4 import get_attend_fn

    q = torch.randn(1, 8, 1024, 512).cuda()
    k = torch.randn(1, 4, 1024, 512).cuda()
    v = torch.randn(1, 4, 1024, 512).cuda()

    attend_kwargs = dict(
        seq_len = 1024,
        k_seq_len = 1024,
        causal = causal,
        causal_block_size = causal_block_size,
        softclamp_value = softclamp_value,
        device = q.device,
        num_agent_tokens = num_agent_tokens,
        block_size_per_special = block_size_per_special,
        special_attend_only_itself = special_attend_only_itself
    )

    attend = get_attend_fn(True, **attend_kwargs)
    flex_out = attend(q, k, v)

    attend = get_attend_fn(False, **attend_kwargs)
    out = attend(q, k, v)

    assert torch.allclose(flex_out, out, atol = 1e-6)

import torch

from latent_meanflow.trainers.discrete_mask_prior_trainer import DiscreteMaskPriorTrainer


def _make_trainer():
    return DiscreteMaskPriorTrainer(
        backbone_config={
            "target": "latent_meanflow.models.backbones.latent_interval_sit.LatentIntervalSiT",
            "params": {
                "input_size": [16, 16],
                "patch_size": 4,
                "hidden_size": 64,
                "depth": 2,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "time_conditioning": "t",
                "time_embed_dim": 64,
            },
        },
        objective_config={
            "target": "latent_meanflow.objectives.discrete_mask_diffusion.DiscreteMaskDiffusionObjective",
            "params": {
                "time_eps": 1.0e-3,
                "loss_type": "cross_entropy",
                "mask_schedule": "linear",
                "min_mask_ratio": 0.0,
                "max_mask_ratio": 1.0,
            },
        },
        sampler_config={
            "target": "latent_meanflow.samplers.discrete_mask_diffusion.SeededDiscreteMaskDiffusionSampler",
            "params": {
                "default_nfe": 4,
                "mask_schedule": "linear",
                "min_mask_ratio": 0.0,
                "max_mask_ratio": 1.0,
                "reveal_noise_scale": 0.2,
                "sample_temperature": 1.0,
            },
        },
        mask_num_classes=4,
        mask_spatial_shape=(16, 16),
        log_sample_nfe=4,
    )


def _make_batch():
    mask_index = torch.zeros((2, 16, 16), dtype=torch.long)
    mask_index[:, :8, :8] = 1
    mask_index[:, :8, 8:] = 2
    mask_index[:, 8:, :] = 3
    return {"mask_index": mask_index}


def test_discrete_mask_prior_smoke():
    torch.manual_seed(7)
    trainer = _make_trainer()
    batch = _make_batch()

    outputs = trainer(batch)
    assert outputs["loss"].ndim == 0
    assert outputs["pred_field"].shape == (2, 4, 16, 16)
    assert outputs["z_t"].shape == (2, 16, 16)
    assert outputs["z_t"].dtype == torch.long
    assert torch.all(outputs["z_t"] <= trainer.mask_token_id)

    noise = torch.randn((2, trainer.latent_channels, *trainer.latent_spatial_shape))
    samples_a = trainer.sample_latents(batch_size=2, nfe=4, noise=noise)
    samples_b = trainer.sample_latents(batch_size=2, nfe=4, noise=noise.clone())
    samples_c = trainer.sample_latents(batch_size=2, nfe=4, noise=torch.randn_like(noise))

    assert samples_a.shape == (2, 16, 16)
    assert samples_a.dtype == torch.long
    assert int(samples_a.min().item()) >= 0
    assert int(samples_a.max().item()) < trainer.num_classes
    assert torch.equal(samples_a, samples_b)
    assert not torch.equal(samples_a, samples_c)

    for nfe in (1, 2, 4):
        sample = trainer.sample_latents(batch_size=2, nfe=nfe, noise=torch.randn_like(noise))
        assert sample.shape == (2, 16, 16)
        assert sample.dtype == torch.long
        assert int(sample.min().item()) >= 0
        assert int(sample.max().item()) < trainer.num_classes

    history = trainer.sampler.sample_with_history(
        model_fn=trainer.predict_field,
        batch_size=1,
        latent_shape=(trainer.latent_channels, *trainer.latent_spatial_shape),
        device=trainer.device,
        noise=torch.randn((1, trainer.latent_channels, *trainer.latent_spatial_shape)),
        nfe=4,
    )
    for prev_state, next_state in zip(history[:-1], history[1:]):
        revealed_mask = prev_state != trainer.mask_token_id
        assert torch.equal(next_state[revealed_mask], prev_state[revealed_mask])

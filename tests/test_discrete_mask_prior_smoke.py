import unittest

import torch
import torch.nn.functional as F

from latent_meanflow.objectives.discrete_mask_diffusion import DiscreteMaskDiffusionObjective
from latent_meanflow.samplers.discrete_mask_diffusion import SeededDiscreteMaskDiffusionSampler
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


def _make_matching_onehot(mask_index, *, num_classes):
    return F.one_hot(mask_index.clamp_min(0), num_classes=num_classes).permute(0, 3, 1, 2).float()


class DiscreteMaskPriorSmokeTest(unittest.TestCase):
    def test_discrete_mask_prior_smoke(self):
        torch.manual_seed(7)
        trainer = _make_trainer()
        batch = _make_batch()

        outputs = trainer(batch)
        self.assertEqual(outputs["loss"].ndim, 0)
        self.assertEqual(tuple(outputs["pred_field"].shape), (2, 4, 16, 16))
        self.assertEqual(tuple(outputs["z_t"].shape), (2, 16, 16))
        self.assertEqual(outputs["z_t"].dtype, torch.long)
        self.assertTrue(torch.all(outputs["z_t"] <= trainer.mask_token_id))

        noise = torch.randn((2, trainer.latent_channels, *trainer.latent_spatial_shape))
        samples_a = trainer.sample_latents(batch_size=2, nfe=4, noise=noise)
        samples_b = trainer.sample_latents(batch_size=2, nfe=4, noise=noise.clone())
        samples_c = trainer.sample_latents(batch_size=2, nfe=4, noise=torch.randn_like(noise))

        self.assertEqual(tuple(samples_a.shape), (2, 16, 16))
        self.assertEqual(samples_a.dtype, torch.long)
        self.assertGreaterEqual(int(samples_a.min().item()), 0)
        self.assertLess(int(samples_a.max().item()), trainer.num_classes)
        self.assertTrue(torch.equal(samples_a, samples_b))
        self.assertFalse(torch.equal(samples_a, samples_c))

        for nfe in (1, 2, 4):
            sample = trainer.sample_latents(batch_size=2, nfe=nfe, noise=torch.randn_like(noise))
            self.assertEqual(tuple(sample.shape), (2, 16, 16))
            self.assertEqual(sample.dtype, torch.long)
            self.assertGreaterEqual(int(sample.min().item()), 0)
            self.assertLess(int(sample.max().item()), trainer.num_classes)

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
            self.assertTrue(torch.equal(next_state[revealed_mask], prev_state[revealed_mask]))

    def test_missing_mask_index_should_fail(self):
        trainer = _make_trainer()
        batch = _make_batch()
        mask_index = batch["mask_index"]
        mask_onehot = _make_matching_onehot(mask_index, num_classes=trainer.num_classes)
        batch_without_index = {"mask_onehot": mask_onehot}

        with self.assertRaisesRegex(KeyError, "requires 'mask_index'"):
            trainer(batch_without_index)

    def test_matching_mask_index_and_mask_onehot_should_pass(self):
        trainer = _make_trainer()
        batch = _make_batch()
        mask_index = batch["mask_index"]
        batch["mask_onehot"] = _make_matching_onehot(mask_index, num_classes=trainer.num_classes)

        outputs = trainer(batch)
        self.assertEqual(tuple(outputs["mask_onehot"].shape), (2, trainer.num_classes, 16, 16))
        self.assertTrue(torch.equal(outputs["mask_index"], mask_index))

    def test_inconsistent_mask_index_and_mask_onehot_should_fail(self):
        trainer = _make_trainer()
        batch = _make_batch()
        mask_index = batch["mask_index"]
        mask_onehot = _make_matching_onehot(mask_index, num_classes=trainer.num_classes)
        mask_onehot[0, :, 0, 0] = 0.0
        mask_onehot[0, 3, 0, 0] = 1.0
        batch["mask_onehot"] = mask_onehot

        with self.assertRaisesRegex(ValueError, "mask_index.*mask_onehot|mask_onehot.*mask_index"):
            trainer(batch)

    def test_ignore_index_zero_auxiliary_onehot_should_pass(self):
        trainer = _make_trainer()
        trainer.ignore_index = -1
        trainer.objective.ignore_index = -1

        batch = _make_batch()
        mask_index = batch["mask_index"].clone()
        mask_index[:, 0, 0] = -1
        mask_onehot = _make_matching_onehot(mask_index, num_classes=trainer.num_classes)
        mask_onehot[:, :, 0, 0] = 0.0
        batch = {
            "mask_index": mask_index,
            "mask_onehot": mask_onehot,
        }

        outputs = trainer(batch)
        self.assertEqual(int(outputs["mask_index"][0, 0, 0].item()), -1)
        self.assertEqual(float(outputs["mask_onehot"][0, :, 0, 0].sum().item()), 0.0)

    def test_objective_non_full_range_schedule_should_fail(self):
        with self.assertRaisesRegex(ValueError, "full-range absorbing schedule"):
            DiscreteMaskDiffusionObjective(
                min_mask_ratio=0.25,
                max_mask_ratio=1.0,
            )
        with self.assertRaisesRegex(ValueError, "min_mask_ratio=0.0, max_mask_ratio=1.0"):
            DiscreteMaskDiffusionObjective(
                min_mask_ratio=0.0,
                max_mask_ratio=0.75,
            )

    def test_sampler_non_full_range_schedule_should_fail(self):
        with self.assertRaisesRegex(ValueError, "full-range absorbing schedule"):
            SeededDiscreteMaskDiffusionSampler(
                min_mask_ratio=0.25,
                max_mask_ratio=1.0,
            )
        with self.assertRaisesRegex(ValueError, "min_mask_ratio=0.0, max_mask_ratio=1.0"):
            SeededDiscreteMaskDiffusionSampler(
                min_mask_ratio=0.0,
                max_mask_ratio=0.75,
            )

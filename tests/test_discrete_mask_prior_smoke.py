import unittest
from types import SimpleNamespace
from pathlib import Path

import torch
import torch.nn.functional as F
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

from latent_meanflow.objectives.discrete_mask_diffusion import DiscreteMaskDiffusionObjective
from latent_meanflow.samplers.discrete_mask_diffusion import SeededDiscreteMaskDiffusionSampler
from latent_meanflow.trainers.discrete_mask_prior_trainer import DiscreteMaskPriorTrainer


def _make_trainer(objective_params=None, sampler_params=None):
    objective_params = dict(objective_params or {})
    sampler_params = dict(sampler_params or {})
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
                **objective_params,
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
                **sampler_params,
            },
        },
        mask_num_classes=4,
        mask_spatial_shape=(16, 16),
        log_sample_nfe=4,
    )


def _make_sampler(**overrides):
    sampler = SeededDiscreteMaskDiffusionSampler(
        default_nfe=4,
        mask_schedule="linear",
        min_mask_ratio=0.0,
        max_mask_ratio=1.0,
        reveal_noise_scale=0.2,
        sample_temperature=1.0,
        refinement_mode="remask_low_confidence",
        min_keep_fraction=0.15,
        lock_noise_scale=0.1,
        **overrides,
    )
    sampler.configure_discrete_state(num_classes=4, mask_token_id=4)
    return sampler


def _make_batch():
    mask_index = torch.zeros((2, 16, 16), dtype=torch.long)
    mask_index[:, :8, :8] = 1
    mask_index[:, :8, 8:] = 2
    mask_index[:, 8:, :] = 3
    return {"mask_index": mask_index}


def _make_matching_onehot(mask_index, *, num_classes):
    return F.one_hot(mask_index.clamp_min(0), num_classes=num_classes).permute(0, 3, 1, 2).float()


def _make_contextual_model_fn(*, num_classes, mask_token_id):
    def model_fn(state, t=None, condition=None):
        del condition
        batch_size, height, width = state.shape
        device = state.device
        y = torch.linspace(-1.0, 1.0, steps=height, device=device).view(1, height, 1).expand(batch_size, height, width)
        x = torch.linspace(-1.0, 1.0, steps=width, device=device).view(1, 1, width).expand(batch_size, height, width)
        visible = (state != mask_token_id).float()
        state_safe = state.clamp(min=0, max=num_classes - 1).float() / float(max(num_classes - 1, 1))
        neighborhood_visible = F.avg_pool2d(visible.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        neighborhood_state = F.avg_pool2d(state_safe.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        if t is None:
            t_term = torch.zeros((batch_size, 1, 1), device=device, dtype=torch.float32)
        else:
            t_term = t.view(batch_size, 1, 1).to(device=device, dtype=torch.float32)
        logits = torch.stack(
            [
                1.2 - (x + 0.75).abs() + 0.7 * neighborhood_visible + 0.2 * t_term,
                1.2 - (y + 0.75).abs() + 0.8 * neighborhood_state - 0.1 * t_term,
                1.2 - (x - 0.75).abs() - 0.5 * neighborhood_visible + 0.3 * t_term,
                1.2 - (y - 0.75).abs() - 0.7 * neighborhood_state - 0.2 * t_term,
            ],
            dim=1,
        )
        return logits

    return model_fn


class _ToyMaskDataset:
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        mask_index = _make_batch()["mask_index"][idx]
        return {"mask_index": mask_index}


REPO_ROOT = Path(__file__).resolve().parents[1]


class DiscreteMaskPriorSmokeTest(unittest.TestCase):
    def test_discrete_mask_prior_smoke(self):
        torch.manual_seed(7)
        trainer = _make_trainer(
            objective_params={
                "corruption_mode": "exact_count",
                "full_mask_batch_fraction": 0.25,
                "high_mask_batch_fraction": 0.25,
                "high_mask_min_ratio": 0.85,
                "class_balance_mode": "inverse_sqrt_frequency",
                "class_counts": [256, 128, 64, 32],
            },
            sampler_params={
                "refinement_mode": "remask_low_confidence",
                "min_keep_fraction": 0.15,
                "lock_noise_scale": 0.1,
            },
        )
        batch = _make_batch()

        outputs = trainer(batch)
        self.assertEqual(outputs["loss"].ndim, 0)
        self.assertEqual(tuple(outputs["pred_field"].shape), (2, 4, 16, 16))
        self.assertEqual(tuple(outputs["z_t"].shape), (2, 16, 16))
        self.assertEqual(outputs["z_t"].dtype, torch.long)
        self.assertTrue(torch.all(outputs["z_t"] <= trainer.mask_token_id))
        self.assertTrue(torch.equal(outputs["masked_counts"], outputs["target_masked_counts"]))

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

    def test_exact_count_corruption_matches_target(self):
        trainer = _make_trainer(
            objective_params={
                "corruption_mode": "exact_count",
                "time_eps": 0.5,
            }
        )
        outputs = trainer(_make_batch())
        self.assertTrue(torch.equal(outputs["masked_counts"], outputs["target_masked_counts"]))
        self.assertGreaterEqual(int(outputs["masked_counts"].min().item()), 1)

    def test_full_mask_and_high_mask_paths(self):
        batch = _make_batch()
        valid_counts = torch.full((2,), 16 * 16, dtype=torch.long)

        full_trainer = _make_trainer(
            objective_params={
                "corruption_mode": "exact_count",
                "full_mask_batch_fraction": 1.0,
                "high_mask_batch_fraction": 0.0,
            }
        )
        full_outputs = full_trainer(batch)
        self.assertTrue(torch.all(full_outputs["full_mask_rows"]))
        self.assertTrue(torch.equal(full_outputs["masked_counts"], valid_counts))
        self.assertTrue(torch.equal(full_outputs["target_masked_counts"], valid_counts))

        high_trainer = _make_trainer(
            objective_params={
                "corruption_mode": "exact_count",
                "full_mask_batch_fraction": 0.0,
                "high_mask_batch_fraction": 1.0,
                "high_mask_min_ratio": 0.85,
            }
        )
        high_outputs = high_trainer(batch)
        self.assertTrue(torch.all(high_outputs["high_mask_rows"]))
        self.assertTrue(torch.equal(high_outputs["masked_counts"], high_outputs["target_masked_counts"]))
        min_expected = torch.round(valid_counts.to(dtype=torch.float32) * 0.85).to(dtype=torch.long).clamp(min=1)
        self.assertTrue(torch.all(high_outputs["target_masked_counts"] >= min_expected))

    def test_class_balanced_objective_uses_normalized_weights(self):
        trainer = _make_trainer(
            objective_params={
                "corruption_mode": "exact_count",
                "class_balance_mode": "effective_num",
                "effective_num_beta": 0.999,
                "class_counts": [1024, 256, 64, 4],
            }
        )
        outputs = trainer(_make_batch())
        class_weights = trainer.objective.class_weights
        self.assertEqual(tuple(class_weights.shape), (trainer.num_classes,))
        self.assertLess(float(class_weights.min().item()), 1.0)
        self.assertGreater(float(class_weights.max().item()), 1.0)
        self.assertAlmostEqual(float(class_weights.mean().item()), 1.0, places=5)
        self.assertIn("class_weight_min", outputs["loss_dict"])
        self.assertIn("class_weight_max", outputs["loss_dict"])
        self.assertIn("class_weight_mean", outputs["loss_dict"])

    def test_class_balance_configuration_preserves_buffer_device(self):
        objective = DiscreteMaskDiffusionObjective(
            class_balance_mode="inverse_sqrt_frequency",
        )
        objective.configure_discrete_state(num_classes=4, mask_token_id=4)
        buffer_device = objective.class_weights.device
        objective.configure_class_balance([128, 64, 32, 16])
        self.assertEqual(objective.class_counts.device, buffer_device)
        self.assertEqual(objective.class_weights.device, buffer_device)
        self.assertEqual(objective.active_class_mask.device, buffer_device)

    def test_trainer_scans_class_counts_when_not_configured(self):
        trainer = _make_trainer(
            objective_params={
                "class_balance_mode": "inverse_sqrt_frequency",
            }
        )
        trainer._trainer = SimpleNamespace(
            datamodule=SimpleNamespace(datasets={"train": _ToyMaskDataset()}),
            is_global_zero=True,
            global_rank=0,
            world_size=1,
        )
        trainer._maybe_configure_objective_class_balance()
        self.assertEqual(tuple(trainer.objective.class_counts.shape), (trainer.num_classes,))
        self.assertGreater(float(trainer.objective.class_counts.sum().item()), 0.0)
        self.assertAlmostEqual(float(trainer.objective.class_weights.mean().item()), 1.0, places=5)

    def test_iterative_refinement_sampler_is_reproducible_and_locked_positions_stay_fixed(self):
        sampler = _make_sampler()
        model_fn = _make_contextual_model_fn(num_classes=4, mask_token_id=4)
        noise = torch.randn((1, 1, 16, 16))

        sample_a = sampler.sample(
            model_fn=model_fn,
            batch_size=1,
            latent_shape=(1, 16, 16),
            device=torch.device("cpu"),
            noise=noise,
            nfe=4,
        )
        sample_b = sampler.sample(
            model_fn=model_fn,
            batch_size=1,
            latent_shape=(1, 16, 16),
            device=torch.device("cpu"),
            noise=noise.clone(),
            nfe=4,
        )
        sample_c = sampler.sample(
            model_fn=model_fn,
            batch_size=1,
            latent_shape=(1, 16, 16),
            device=torch.device("cpu"),
            noise=torch.randn_like(noise),
            nfe=4,
        )
        self.assertTrue(torch.equal(sample_a, sample_b))
        self.assertFalse(torch.equal(sample_a, sample_c))

        history = sampler.sample_with_history(
            model_fn=model_fn,
            batch_size=1,
            latent_shape=(1, 16, 16),
            device=torch.device("cpu"),
            noise=noise,
            nfe=4,
        )
        state_history = history["state_history"]
        proposal_history = history["proposal_history"]
        locked_mask_history = history["locked_mask_history"]

        self.assertGreaterEqual(len(state_history), 2)
        self.assertEqual(len(state_history), len(locked_mask_history))
        self.assertGreaterEqual(len(proposal_history), 1)

        for step_idx in range(1, len(state_history)):
            prev_locked = locked_mask_history[step_idx - 1]
            self.assertTrue(torch.equal(state_history[step_idx][prev_locked], state_history[step_idx - 1][prev_locked]))

        unlocked_changed = False
        shared_steps = min(len(proposal_history) - 1, len(locked_mask_history) - 2)
        for step_idx in range(shared_steps):
            still_unlocked = (~locked_mask_history[step_idx + 1]) & (~locked_mask_history[step_idx + 2])
            if torch.any(still_unlocked & (proposal_history[step_idx] != proposal_history[step_idx + 1])):
                unlocked_changed = True
                break
        self.assertTrue(unlocked_changed)

    def test_memorization_diagnostic_configs_instantiate_and_keep_interfaces(self):
        config_paths = [
            REPO_ROOT / "configs" / "diagnostics" / "discrete_mask_prior_sit_highmask_refine_memorize_1.yaml",
            REPO_ROOT / "configs" / "diagnostics" / "discrete_mask_prior_sit_highmask_refine_memorize_4.yaml",
        ]
        for config_path in config_paths:
            config = OmegaConf.load(config_path)
            trainer = instantiate_from_config(config.model)
            batch = {
                "mask_index": torch.randint(
                    low=0,
                    high=trainer.num_classes,
                    size=(1, *trainer.mask_spatial_shape),
                    dtype=torch.long,
                )
            }
            with torch.no_grad():
                outputs = trainer(batch)
                self.assertEqual(outputs["loss"].ndim, 0)
                sampled = trainer.sample_latents(
                    batch_size=1,
                    nfe=1,
                    noise=torch.randn((1, trainer.latent_channels, *trainer.latent_spatial_shape)),
                )
                decoded = trainer.decode_latents(sampled)
            self.assertEqual(tuple(sampled.shape), (1, *trainer.mask_spatial_shape))
            self.assertEqual(decoded["mask_index"].dtype, torch.long)
            self.assertEqual(tuple(decoded["mask_onehot"].shape), (1, trainer.num_classes, *trainer.mask_spatial_shape))

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

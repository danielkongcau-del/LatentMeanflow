import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

from latent_meanflow.losses.semantic_structure import (
    adjacency_l1_loss,
    area_ratio_l1_loss,
    boundary_bce_loss,
    build_valid_mask,
    compute_class_adjacency_matrix,
    compute_class_area_ratios,
    mask_index_to_boundary_target,
    semantic_probs_to_soft_boundary,
)
from latent_meanflow.models.semantic_mask_vq_autoencoder import SemanticMaskVQAutoencoder
from latent_meanflow.trainers.token_mask_prior_trainer import TokenMaskPriorTrainer


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_tokenizer_model():
    return SemanticMaskVQAutoencoder(
        ddconfig={
            "double_z": False,
            "z_channels": 4,
            "resolution": 16,
            "in_channels": 4,
            "out_ch": 4,
            "ch": 32,
            "ch_mult": [1, 2],
            "num_res_blocks": 1,
            "attn_resolutions": [],
            "dropout": 0.0,
        },
        lossconfig={
            "target": "latent_meanflow.models.semantic_mask_vq_autoencoder.SemanticMaskVQLoss",
            "params": {
                "mask_ce_weight": 1.0,
                "mask_dice_weight": 0.0,
                "mask_focal_weight": 0.0,
                "vq_codebook_weight": 1.0,
                "vq_commit_weight": 0.25,
                "ignore_index": -1,
            },
        },
        embed_dim=8,
        codebook_size=32,
        num_classes=4,
        quantizer_config={
            "distance_metric": "cosine",
            "use_ema_update": True,
            "ema_decay": 0.99,
            "ema_eps": 1.0e-5,
            "dead_code_threshold": 1.0,
        },
    )


def _write_tokenizer_artifacts(tmpdir):
    tokenizer = _make_tokenizer_model()
    config = OmegaConf.create(
        {
            "model": {
                "target": "latent_meanflow.models.semantic_mask_vq_autoencoder.SemanticMaskVQAutoencoder",
                "params": {
                    "embed_dim": 8,
                    "codebook_size": 32,
                    "num_classes": 4,
                    "lossconfig": {
                        "target": "latent_meanflow.models.semantic_mask_vq_autoencoder.SemanticMaskVQLoss",
                        "params": {
                            "mask_ce_weight": 1.0,
                            "mask_dice_weight": 0.0,
                            "mask_focal_weight": 0.0,
                            "vq_codebook_weight": 1.0,
                            "vq_commit_weight": 0.25,
                            "ignore_index": -1,
                        },
                    },
                    "quantizer_config": {
                        "distance_metric": "cosine",
                        "use_ema_update": True,
                        "ema_decay": 0.99,
                        "ema_eps": 1.0e-5,
                        "dead_code_threshold": 1.0,
                    },
                    "ddconfig": {
                        "double_z": False,
                        "z_channels": 4,
                        "resolution": 16,
                        "in_channels": 4,
                        "out_ch": 4,
                        "ch": 32,
                        "ch_mult": [1, 2],
                        "num_res_blocks": 1,
                        "attn_resolutions": [],
                        "dropout": 0.0,
                    },
                },
            }
        }
    )
    config_path = Path(tmpdir) / "tokenizer.yaml"
    ckpt_path = Path(tmpdir) / "tokenizer.ckpt"
    OmegaConf.save(config, config_path)
    torch.save({"state_dict": tokenizer.state_dict()}, ckpt_path)
    return config_path, ckpt_path, tokenizer.latent_spatial_shape, tokenizer.codebook_size


def _make_prior_trainer(
    *,
    tokenizer_config_path,
    tokenizer_ckpt_path,
    token_spatial_shape,
    semantic_ce_weight=0.2,
    semantic_dice_weight=0.1,
    boundary_loss_weight=0.05,
    area_ratio_loss_weight=0.05,
    adjacency_loss_weight=0.02,
):
    return TokenMaskPriorTrainer(
        tokenizer_config_path=str(tokenizer_config_path),
        tokenizer_ckpt_path=str(tokenizer_ckpt_path),
        backbone_config={
            "target": "latent_meanflow.models.backbones.latent_interval_sit.LatentIntervalSiT",
            "params": {
                "input_size": list(token_spatial_shape),
                "patch_size": 2,
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
                "default_nfe": 2,
                "mask_schedule": "linear",
                "min_mask_ratio": 0.0,
                "max_mask_ratio": 1.0,
                "reveal_noise_scale": 0.2,
                "sample_temperature": 1.0,
            },
        },
        freeze_tokenizer=True,
        tokenizer_sample_posterior=False,
        semantic_ce_weight=semantic_ce_weight,
        semantic_dice_weight=semantic_dice_weight,
        boundary_loss_weight=boundary_loss_weight,
        area_ratio_loss_weight=area_ratio_loss_weight,
        adjacency_loss_weight=adjacency_loss_weight,
        log_sample_nfe=2,
    )


def _make_batch():
    mask_index = torch.zeros((2, 16, 16), dtype=torch.long)
    mask_index[:, :8, :8] = 1
    mask_index[:, :8, 8:] = 2
    mask_index[:, 8:, :] = 3
    mask_onehot = F.one_hot(mask_index, num_classes=4).float()
    return {
        "mask_index": mask_index,
        "mask_onehot": mask_onehot,
        "num_classes": torch.tensor([4, 4], dtype=torch.long),
    }


class TokenMaskPriorSmokeTest(unittest.TestCase):
    def test_token_mask_prior_forward_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, ckpt_path, token_spatial_shape, codebook_size = _write_tokenizer_artifacts(tmpdir)
            trainer = _make_prior_trainer(
                tokenizer_config_path=config_path,
                tokenizer_ckpt_path=ckpt_path,
                token_spatial_shape=token_spatial_shape,
            )

            outputs = trainer(_make_batch())
            self.assertEqual(outputs["loss"].ndim, 0)
            self.assertEqual(tuple(outputs["pred_field"].shape), (2, codebook_size, *token_spatial_shape))
            self.assertEqual(tuple(outputs["code_grid"].shape), (2, *token_spatial_shape))
            self.assertEqual(trainer.mask_token_id, codebook_size)
            self.assertEqual(tuple(outputs["semantic_mask_logits"].shape), (2, 4, 16, 16))
            self.assertIn("semantic_ce", outputs["loss_dict"])
            self.assertIn("semantic_dice", outputs["loss_dict"])
            self.assertIn("semantic_aux_total", outputs["loss_dict"])
            self.assertIn("boundary_loss", outputs["loss_dict"])
            self.assertIn("area_ratio_loss", outputs["loss_dict"])
            self.assertIn("adjacency_loss", outputs["loss_dict"])
            self.assertIn("structure_aux_total", outputs["loss_dict"])
            self.assertIn("semantic_pixel_accuracy", outputs["semantic_bridge_metrics"])
            self.assertIn("semantic_miou", outputs["semantic_bridge_metrics"])
            self.assertEqual(tuple(outputs["boundary_target"].shape), (2, 1, 16, 16))
            self.assertEqual(tuple(outputs["boundary_pred"].shape), (2, 1, 16, 16))
            self.assertEqual(tuple(outputs["pred_area_ratio"].shape), (2, 4))
            self.assertEqual(tuple(outputs["target_area_ratio"].shape), (2, 4))
            self.assertEqual(tuple(outputs["pred_adjacency"].shape), (2, 4, 4))
            self.assertEqual(tuple(outputs["target_adjacency"].shape), (2, 4, 4))

    def test_tokenizer_is_frozen_and_not_in_optimizer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, ckpt_path, token_spatial_shape, _ = _write_tokenizer_artifacts(tmpdir)
            trainer = _make_prior_trainer(
                tokenizer_config_path=config_path,
                tokenizer_ckpt_path=ckpt_path,
                token_spatial_shape=token_spatial_shape,
            )

            self.assertTrue(all(not param.requires_grad for param in trainer.tokenizer.parameters()))
            optimizer = trainer.configure_optimizers()
            tokenizer_param_ids = {id(param) for param in trainer.tokenizer.parameters()}
            optimizer_param_ids = {
                id(param)
                for group in optimizer.param_groups
                for param in group["params"]
            }
            self.assertTrue(tokenizer_param_ids.isdisjoint(optimizer_param_ids))

    def test_soft_decode_bridge_backpropagates_to_code_logits_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, ckpt_path, token_spatial_shape, codebook_size = _write_tokenizer_artifacts(tmpdir)
            trainer = _make_prior_trainer(
                tokenizer_config_path=config_path,
                tokenizer_ckpt_path=ckpt_path,
                token_spatial_shape=token_spatial_shape,
            )

            code_logits = torch.randn(
                (2, codebook_size, *token_spatial_shape),
                dtype=torch.float32,
                requires_grad=True,
            )
            decoded = trainer.tokenizer.decode_code_distribution(code_logits=code_logits)
            self.assertEqual(tuple(decoded["mask_logits"].shape), (2, 4, 16, 16))
            self.assertEqual(tuple(decoded["mask_probs"].shape), (2, 4, 16, 16))
            self.assertEqual(tuple(decoded["z_q"].shape), (2, 8, *token_spatial_shape))

            bridge_loss = decoded["mask_logits"].square().mean()
            bridge_loss.backward()

            self.assertIsNotNone(code_logits.grad)
            self.assertGreater(float(code_logits.grad.abs().sum().item()), 0.0)
            self.assertTrue(all(param.grad is None for param in trainer.tokenizer.parameters()))

    def test_semantic_structure_loss_utils_smoke(self):
        batch = _make_batch()
        mask_index = batch["mask_index"]
        mask_onehot = batch["mask_onehot"].permute(0, 3, 1, 2).float()
        mask_probs = (0.9 * mask_onehot) + (0.1 / 4.0)
        valid_mask = build_valid_mask(mask_index, ignore_index=None)

        boundary_target = mask_index_to_boundary_target(mask_index, ignore_index=None)
        boundary_pred = semantic_probs_to_soft_boundary(mask_probs, valid_mask=valid_mask)
        boundary_loss = boundary_bce_loss(boundary_pred, boundary_target, valid_mask=valid_mask)
        self.assertEqual(tuple(boundary_target.shape), (2, 1, 16, 16))
        self.assertEqual(tuple(boundary_pred.shape), (2, 1, 16, 16))
        self.assertTrue(torch.isfinite(boundary_loss))

        area_loss, pred_area_ratio, target_area_ratio = area_ratio_l1_loss(
            mask_probs,
            mask_onehot,
            valid_mask=valid_mask,
        )
        self.assertEqual(tuple(pred_area_ratio.shape), (2, 4))
        self.assertEqual(tuple(target_area_ratio.shape), (2, 4))
        self.assertTrue(torch.isfinite(area_loss))
        self.assertEqual(tuple(compute_class_area_ratios(mask_probs, valid_mask=valid_mask).shape), (2, 4))

        adjacency_loss, pred_adjacency, target_adjacency = adjacency_l1_loss(
            mask_probs,
            mask_onehot,
            valid_mask=valid_mask,
        )
        self.assertEqual(tuple(pred_adjacency.shape), (2, 4, 4))
        self.assertEqual(tuple(target_adjacency.shape), (2, 4, 4))
        self.assertTrue(torch.isfinite(adjacency_loss))
        self.assertEqual(tuple(compute_class_adjacency_matrix(mask_probs, valid_mask=valid_mask).shape), (2, 4, 4))

    def test_sampled_token_grids_decode_into_valid_semantic_masks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, ckpt_path, token_spatial_shape, codebook_size = _write_tokenizer_artifacts(tmpdir)
            trainer = _make_prior_trainer(
                tokenizer_config_path=config_path,
                tokenizer_ckpt_path=ckpt_path,
                token_spatial_shape=token_spatial_shape,
            )

            sampled = trainer.sample_latents(
                batch_size=2,
                nfe=2,
                device=torch.device("cpu"),
                noise=torch.randn((2, trainer.latent_channels, *trainer.latent_spatial_shape)),
            )
            self.assertEqual(tuple(sampled.shape), (2, *token_spatial_shape))
            self.assertEqual(sampled.dtype, torch.long)
            self.assertGreaterEqual(int(sampled.min().item()), 0)
            self.assertLess(int(sampled.max().item()), codebook_size)

            decoded = trainer.decode_latents(sampled)
            self.assertEqual(tuple(decoded["codes"].shape), (2, *token_spatial_shape))
            self.assertEqual(tuple(decoded["mask_index"].shape), (2, 16, 16))
            self.assertEqual(tuple(decoded["mask_onehot"].shape), (2, 4, 16, 16))

    def test_main_and_memorize_configs_pin_balanced_tokenizer(self):
        config_paths = [
            REPO_ROOT / "configs" / "token_mask_prior_vq_sit_tiny.yaml",
            REPO_ROOT / "configs" / "token_mask_prior_vq_sit.yaml",
            REPO_ROOT / "configs" / "diagnostics" / "token_mask_prior_vq_sit_memorize_1.yaml",
            REPO_ROOT / "configs" / "diagnostics" / "token_mask_prior_vq_sit_memorize_4.yaml",
        ]
        for config_path in config_paths:
            config = OmegaConf.load(config_path)
            self.assertEqual(
                config.model.target,
                "latent_meanflow.trainers.token_mask_prior_trainer.TokenMaskPriorTrainer",
            )
            self.assertEqual(
                str(config.model.params.tokenizer_config_path),
                "configs/semantic_mask_vq_tokenizer_main_balanced_256.yaml",
            )
            self.assertIsNone(config.model.params.tokenizer_ckpt_path)
            self.assertTrue(bool(config.model.params.freeze_tokenizer))
            self.assertFalse(bool(config.model.params.tokenizer_sample_posterior))
            objective_params = config.model.params.objective_config.params
            sampler_params = config.model.params.sampler_config.params
            self.assertEqual(str(objective_params.corruption_mode), "exact_count")
            self.assertAlmostEqual(float(objective_params.full_mask_batch_fraction), 0.25)
            self.assertAlmostEqual(float(objective_params.high_mask_batch_fraction), 0.50)
            self.assertAlmostEqual(float(objective_params.high_mask_min_ratio), 0.85)
            self.assertEqual(str(sampler_params.refinement_mode), "proposal_visible_refine")
            self.assertTrue(bool(sampler_params.final_full_reveal))
            self.assertAlmostEqual(float(sampler_params.min_keep_fraction), 0.15)
            self.assertAlmostEqual(float(sampler_params.lock_noise_scale), 0.10)
            self.assertAlmostEqual(float(sampler_params.reveal_noise_scale), 0.15)
            self.assertAlmostEqual(float(sampler_params.sample_temperature), 1.0)
        for config_path in config_paths:
            config = OmegaConf.load(config_path)
            self.assertAlmostEqual(float(config.model.params.semantic_ce_weight), 0.2)
            self.assertAlmostEqual(float(config.model.params.semantic_dice_weight), 0.1)
            self.assertAlmostEqual(float(config.model.params.boundary_loss_weight), 0.05)
            self.assertAlmostEqual(float(config.model.params.area_ratio_loss_weight), 0.05)
            self.assertAlmostEqual(float(config.model.params.adjacency_loss_weight), 0.02)

    def test_hifi_configs_pin_patch_size_one(self):
        expected_backbones = {
            REPO_ROOT / "configs" / "token_mask_prior_vq_sit_hifi.yaml": {
                "patch_size": 1,
                "hidden_size": 256,
                "depth": 8,
                "num_heads": 4,
            },
            REPO_ROOT / "configs" / "token_mask_prior_vq_sit_hifi_tiny.yaml": {
                "patch_size": 1,
                "hidden_size": 192,
                "depth": 6,
                "num_heads": 3,
            },
        }
        for config_path, expected in expected_backbones.items():
            config = OmegaConf.load(config_path)
            self.assertEqual(
                config.model.target,
                "latent_meanflow.trainers.token_mask_prior_trainer.TokenMaskPriorTrainer",
            )
            self.assertEqual(
                str(config.model.params.tokenizer_config_path),
                "configs/semantic_mask_vq_tokenizer_main_balanced_256.yaml",
            )
            self.assertTrue(bool(config.model.params.freeze_tokenizer))
            self.assertFalse(bool(config.model.params.tokenizer_sample_posterior))
            backbone_params = config.model.params.backbone_config.params
            self.assertEqual(int(backbone_params.patch_size), expected["patch_size"])
            self.assertEqual(int(backbone_params.hidden_size), expected["hidden_size"])
            self.assertEqual(int(backbone_params.depth), expected["depth"])
            self.assertEqual(int(backbone_params.num_heads), expected["num_heads"])
            self.assertAlmostEqual(float(config.model.params.semantic_ce_weight), 0.2)
            self.assertAlmostEqual(float(config.model.params.semantic_dice_weight), 0.1)
            self.assertAlmostEqual(float(config.model.params.boundary_loss_weight), 0.05)
            self.assertAlmostEqual(float(config.model.params.area_ratio_loss_weight), 0.05)
            self.assertAlmostEqual(float(config.model.params.adjacency_loss_weight), 0.02)

    def test_control_configs_keep_old_progressive_reveal_semantics(self):
        config_paths = [
            REPO_ROOT / "configs" / "token_mask_prior_vq_sit_tiny_control.yaml",
            REPO_ROOT / "configs" / "token_mask_prior_vq_sit_control.yaml",
        ]
        for config_path in config_paths:
            config = OmegaConf.load(config_path)
            self.assertEqual(
                config.model.target,
                "latent_meanflow.trainers.token_mask_prior_trainer.TokenMaskPriorTrainer",
            )
            self.assertEqual(
                str(config.model.params.tokenizer_config_path),
                "configs/semantic_mask_vq_tokenizer_main_balanced_256.yaml",
            )
            objective_params = config.model.params.objective_config.params
            sampler_params = config.model.params.sampler_config.params
            self.assertFalse(hasattr(objective_params, "corruption_mode"))
            self.assertEqual(str(sampler_params.refinement_mode), "progressive_reveal")
            self.assertAlmostEqual(float(sampler_params.min_keep_fraction), 0.0)
            self.assertAlmostEqual(float(sampler_params.lock_noise_scale), 0.15)

    def test_memorize_config_uses_fixed_subset(self):
        config = OmegaConf.load(
            REPO_ROOT / "configs" / "diagnostics" / "token_mask_prior_vq_sit_memorize_4.yaml"
        )
        self.assertEqual(config.data.params.train.target, "latent_meanflow.data.subset.FixedSubsetDataset")
        self.assertEqual(config.data.params.validation.target, "latent_meanflow.data.subset.FixedSubsetDataset")
        self.assertEqual(int(config.data.params.train.params.first_n), 4)
        self.assertEqual(int(config.data.params.validation.params.first_n), 4)

    def test_config_instantiation_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, ckpt_path, _, _ = _write_tokenizer_artifacts(tmpdir)
            config = OmegaConf.load(REPO_ROOT / "configs" / "token_mask_prior_vq_sit_tiny.yaml")
            OmegaConf.update(config, "model.params.tokenizer_config_path", str(config_path), merge=False)
            OmegaConf.update(config, "model.params.tokenizer_ckpt_path", str(ckpt_path), merge=False)

            trainer = instantiate_from_config(config.model)
            outputs = trainer(_make_batch())
            self.assertEqual(outputs["loss"].ndim, 0)
            self.assertIn("semantic_ce", outputs["loss_dict"])
            self.assertIn("semantic_dice", outputs["loss_dict"])
            self.assertIn("semantic_aux_total", outputs["loss_dict"])
            self.assertIn("boundary_loss", outputs["loss_dict"])
            self.assertIn("area_ratio_loss", outputs["loss_dict"])
            self.assertIn("adjacency_loss", outputs["loss_dict"])

    def test_refine_mainline_config_instantiates_and_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, ckpt_path, token_spatial_shape, _ = _write_tokenizer_artifacts(tmpdir)
            config = OmegaConf.load(REPO_ROOT / "configs" / "token_mask_prior_vq_sit_tiny.yaml")
            OmegaConf.update(config, "model.params.tokenizer_config_path", str(config_path), merge=False)
            OmegaConf.update(config, "model.params.tokenizer_ckpt_path", str(ckpt_path), merge=False)

            trainer = instantiate_from_config(config.model)
            self.assertEqual(str(trainer.objective.corruption_mode), "exact_count")
            self.assertEqual(str(trainer.sampler.refinement_mode), "proposal_visible_refine")
            sampled = trainer.sample_latents(
                batch_size=2,
                nfe=2,
                device=torch.device("cpu"),
                noise=torch.randn((2, trainer.latent_channels, *trainer.latent_spatial_shape)),
            )
            self.assertEqual(tuple(sampled.shape), (2, *token_spatial_shape))
            self.assertEqual(sampled.dtype, torch.long)

    def test_hifi_tiny_config_instantiates_and_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, ckpt_path, token_spatial_shape, _ = _write_tokenizer_artifacts(tmpdir)
            config = OmegaConf.load(REPO_ROOT / "configs" / "token_mask_prior_vq_sit_hifi_tiny.yaml")
            OmegaConf.update(config, "model.params.tokenizer_config_path", str(config_path), merge=False)
            OmegaConf.update(config, "model.params.tokenizer_ckpt_path", str(ckpt_path), merge=False)

            trainer = instantiate_from_config(config.model)
            self.assertEqual(int(trainer.backbone.patch_size), 1)
            sampled = trainer.sample_latents(
                batch_size=1,
                nfe=2,
                device=torch.device("cpu"),
                noise=torch.randn((1, trainer.latent_channels, *trainer.latent_spatial_shape)),
            )
            self.assertEqual(tuple(sampled.shape), (1, *token_spatial_shape))
            self.assertEqual(sampled.dtype, torch.long)

    def test_hifi_main_config_instantiates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, ckpt_path, _, _ = _write_tokenizer_artifacts(tmpdir)
            config = OmegaConf.load(REPO_ROOT / "configs" / "token_mask_prior_vq_sit_hifi.yaml")
            OmegaConf.update(config, "model.params.tokenizer_config_path", str(config_path), merge=False)
            OmegaConf.update(config, "model.params.tokenizer_ckpt_path", str(ckpt_path), merge=False)

            trainer = instantiate_from_config(config.model)
            self.assertEqual(int(trainer.backbone.patch_size), 1)
            self.assertEqual(int(trainer.backbone.hidden_size), 256)
            self.assertEqual(int(trainer.backbone.depth), 8)
            self.assertEqual(int(trainer.backbone.num_heads), 4)


if __name__ == "__main__":
    unittest.main()

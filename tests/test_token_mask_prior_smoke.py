import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

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


def _make_prior_trainer(*, tokenizer_config_path, tokenizer_ckpt_path, token_spatial_shape):
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


if __name__ == "__main__":
    unittest.main()

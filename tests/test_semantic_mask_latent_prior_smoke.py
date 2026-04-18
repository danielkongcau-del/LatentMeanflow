import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from latent_meanflow.models.semantic_mask_autoencoder import SemanticMaskAutoencoder
from latent_meanflow.trainers.semantic_mask_latent_prior_trainer import SemanticMaskLatentPriorTrainer


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_tokenizer_model():
    return SemanticMaskAutoencoder(
        ddconfig={
            "double_z": True,
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
            "target": "latent_meanflow.models.semantic_mask_autoencoder.SemanticMaskLoss",
            "params": {
                "mask_ce_weight": 1.0,
                "mask_dice_weight": 0.0,
                "mask_focal_weight": 0.0,
                "kl_weight": 1.0e-6,
                "ignore_index": -1,
            },
        },
        embed_dim=4,
        num_classes=4,
        sample_posterior=False,
    )


def _write_tokenizer_artifacts(tmpdir):
    tokenizer = _make_tokenizer_model()
    config = OmegaConf.create(
        {
            "model": {
                "target": "latent_meanflow.models.semantic_mask_autoencoder.SemanticMaskAutoencoder",
                "params": {
                    "embed_dim": 4,
                    "num_classes": 4,
                    "sample_posterior": False,
                    "lossconfig": {
                        "target": "latent_meanflow.models.semantic_mask_autoencoder.SemanticMaskLoss",
                        "params": {
                            "mask_ce_weight": 1.0,
                            "mask_dice_weight": 0.0,
                            "mask_focal_weight": 0.0,
                            "kl_weight": 1.0e-6,
                            "ignore_index": -1,
                        },
                    },
                    "ddconfig": {
                        "double_z": True,
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
    return config_path, ckpt_path, tokenizer.latent_spatial_shape, tokenizer.latent_channels


def _make_prior_trainer(*, tokenizer_config_path, tokenizer_ckpt_path, latent_spatial_shape):
    return SemanticMaskLatentPriorTrainer(
        tokenizer_config_path=str(tokenizer_config_path),
        tokenizer_ckpt_path=None if tokenizer_ckpt_path is None else str(tokenizer_ckpt_path),
        backbone_config={
            "target": "latent_meanflow.models.backbones.latent_interval_sit.LatentIntervalSiT",
            "params": {
                "input_size": list(latent_spatial_shape),
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
            "target": "latent_meanflow.objectives.diffusion.GaussianDiffusionObjective",
            "params": {
                "num_train_steps": 16,
                "beta_schedule": "cosine",
                "loss_type": "mse",
            },
        },
        sampler_config={
            "target": "latent_meanflow.samplers.diffusion.DDIMDiffusionSampler",
            "params": {
                "num_train_steps": 16,
                "beta_schedule": "cosine",
                "default_nfe": 2,
                "eta": 0.0,
                "clip_denoised": False,
            },
        },
        tokenizer_sample_posterior=False,
        freeze_tokenizer=True,
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


class SemanticMaskLatentPriorSmokeTest(unittest.TestCase):
    def test_frozen_tokenizer_latent_prior_forward_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, ckpt_path, latent_spatial_shape, latent_channels = _write_tokenizer_artifacts(tmpdir)
            trainer = _make_prior_trainer(
                tokenizer_config_path=config_path,
                tokenizer_ckpt_path=ckpt_path,
                latent_spatial_shape=latent_spatial_shape,
            )

            outputs = trainer(_make_batch())
            self.assertEqual(outputs["loss"].ndim, 0)
            self.assertEqual(tuple(outputs["pred_field"].shape), (2, latent_channels, *latent_spatial_shape))
            self.assertEqual(tuple(outputs["target_field"].shape), (2, latent_channels, *latent_spatial_shape))
            self.assertEqual(tuple(outputs["x_lat"].shape), (2, latent_channels, *latent_spatial_shape))
            self.assertEqual(trainer.num_classes, 4)
            self.assertEqual(trainer.latent_channels, latent_channels)
            self.assertEqual(tuple(trainer.latent_spatial_shape), tuple(latent_spatial_shape))

    def test_tokenizer_is_frozen_and_not_in_optimizer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, ckpt_path, latent_spatial_shape, _ = _write_tokenizer_artifacts(tmpdir)
            trainer = _make_prior_trainer(
                tokenizer_config_path=config_path,
                tokenizer_ckpt_path=ckpt_path,
                latent_spatial_shape=latent_spatial_shape,
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

    def test_encode_decode_contract_returns_continuous_latents_and_mask_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, ckpt_path, latent_spatial_shape, latent_channels = _write_tokenizer_artifacts(tmpdir)
            trainer = _make_prior_trainer(
                tokenizer_config_path=config_path,
                tokenizer_ckpt_path=ckpt_path,
                latent_spatial_shape=latent_spatial_shape,
            )

            sampled = trainer.sample_latents(batch_size=2, nfe=2, device=torch.device("cpu"))
            self.assertTrue(torch.is_floating_point(sampled))
            self.assertEqual(tuple(sampled.shape), (2, latent_channels, *latent_spatial_shape))

            decoded = trainer.decode_latents(sampled)
            self.assertEqual(set(decoded.keys()), {"mask_logits", "mask_probs", "mask_index", "mask_onehot"})
            self.assertEqual(tuple(decoded["mask_index"].shape), (2, 16, 16))
            self.assertEqual(tuple(decoded["mask_logits"].shape), (2, 4, 16, 16))

    def test_latent_prior_memorize_config_contract(self):
        config = OmegaConf.load(
            REPO_ROOT / "configs" / "diagnostics" / "latent_semantic_mask_prior_diffusion_memorize_4.yaml"
        )
        self.assertEqual(
            config.model.target,
            "latent_meanflow.trainers.semantic_mask_latent_prior_trainer.SemanticMaskLatentPriorTrainer",
        )
        self.assertEqual(
            config.data.params.train.target,
            "latent_meanflow.data.subset.FixedSubsetDataset",
        )
        self.assertEqual(
            config.data.params.train.params.dataset_config.target,
            "latent_meanflow.data.semantic_mask.MultiSemanticMaskDataset",
        )
        tokenizer_config = OmegaConf.load(REPO_ROOT / config.model.params.tokenizer_config_path)
        self.assertEqual(
            tokenizer_config.model.target,
            "latent_meanflow.models.semantic_mask_autoencoder.SemanticMaskAutoencoder",
        )

    def test_deterministic_latent_target_under_posterior_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, ckpt_path, latent_spatial_shape, _ = _write_tokenizer_artifacts(tmpdir)
            trainer = _make_prior_trainer(
                tokenizer_config_path=config_path,
                tokenizer_ckpt_path=ckpt_path,
                latent_spatial_shape=latent_spatial_shape,
            )
            batch = _make_batch()

            encoded_a = trainer.encode_batch(batch)
            encoded_b = trainer.encode_batch(batch)
            self.assertTrue(torch.allclose(encoded_a["z"], encoded_b["z"]))
            self.assertTrue(torch.equal(encoded_a["mask_index"], encoded_b["mask_index"]))


if __name__ == "__main__":
    unittest.main()

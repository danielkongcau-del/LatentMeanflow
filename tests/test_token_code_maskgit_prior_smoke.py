import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

from latent_meanflow.models.semantic_mask_vq_autoencoder import SemanticMaskVQAutoencoder
from latent_meanflow.trainers.token_code_maskgit_prior_trainer import TokenCodeMaskGitPriorTrainer
from scripts.sample_token_mask_prior import (
    extract_token_mask_prior_route_metadata,
    resolve_configured_tokenizer_artifacts,
    validate_token_mask_prior_checkpoint_contract,
)


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


def _write_runtime_ckpt(tmpdir, *, tokenizer_config_path, tokenizer_ckpt_path):
    ckpt_path = Path(tmpdir) / "token_code_maskgit_tiny" / "checkpoints" / "last.ckpt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {},
            "hyper_parameters": {
                "monitor": "val/sampled_monitor_error",
                "objective_name": "token_code_maskgit",
                "tokenizer_config_path": str(Path(tokenizer_config_path).resolve()),
                "tokenizer_ckpt_path": str(Path(tokenizer_ckpt_path).resolve()),
                "freeze_tokenizer": True,
                "tokenizer_sample_posterior": False,
            },
        },
        ckpt_path,
    )
    return ckpt_path


def _make_prior_trainer(
    *,
    tokenizer_config_path,
    tokenizer_ckpt_path,
    monitor="val/sampled_monitor_error",
    enable_validation_sample_metrics=True,
    validation_sample_batch_size=4,
    validation_sample_metric_batches=4,
):
    return TokenCodeMaskGitPriorTrainer(
        tokenizer_config_path=str(tokenizer_config_path),
        tokenizer_ckpt_path=str(tokenizer_ckpt_path),
        backbone_config={
            "target": "latent_meanflow.models.backbones.token_code_maskgit_transformer.TokenCodeMaskGitTransformerBackbone",
            "params": {
                "n_layer": 2,
                "n_head": 4,
                "n_embd": 64,
                "embd_pdrop": 0.0,
                "resid_pdrop": 0.0,
                "attn_pdrop": 0.0,
                "mask_schedule_type": "cosine",
            },
        },
        permuter_config={"target": "taming.modules.transformer.permuter.Identity"},
        freeze_tokenizer=True,
        tokenizer_sample_posterior=False,
        monitor=monitor,
        log_sample_nfe=4,
        label_smoothing=0.1,
        sample_temperature=1.0,
        sample_top_k=None,
        sample_base_gumbel_temp=4.5,
        enable_validation_sample_metrics=enable_validation_sample_metrics,
        validation_sample_batch_size=validation_sample_batch_size,
        validation_sample_metric_batches=validation_sample_metric_batches,
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


class TokenCodeMaskGitPriorSmokeTest(unittest.TestCase):
    def test_forward_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, ckpt_path, token_spatial_shape, codebook_size = _write_tokenizer_artifacts(tmpdir)
            trainer = _make_prior_trainer(
                tokenizer_config_path=config_path,
                tokenizer_ckpt_path=ckpt_path,
            )

            outputs = trainer(_make_batch())
            self.assertEqual(outputs["loss"].ndim, 0)
            self.assertEqual(tuple(outputs["code_grid"].shape), (2, *token_spatial_shape))
            self.assertEqual(tuple(outputs["code_sequence"].shape), (2, token_spatial_shape[0] * token_spatial_shape[1]))
            self.assertEqual(tuple(outputs["masked_input_tokens"].shape), (2, token_spatial_shape[0] * token_spatial_shape[1]))
            self.assertEqual(tuple(outputs["target_tokens"].shape), (2, token_spatial_shape[0] * token_spatial_shape[1]))
            self.assertEqual(
                tuple(outputs["token_logits"].shape),
                (2, token_spatial_shape[0] * token_spatial_shape[1], codebook_size),
            )
            self.assertIn("maskgit_ce", outputs["loss_dict"])
            self.assertIn("masked_token_accuracy", outputs["maskgit_metrics"])
            self.assertIn("masked_token_fraction", outputs["maskgit_metrics"])
            self.assertEqual(trainer.mask_token_id, codebook_size)

    def test_tokenizer_is_frozen_and_not_in_optimizer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, ckpt_path, _, _ = _write_tokenizer_artifacts(tmpdir)
            trainer = _make_prior_trainer(
                tokenizer_config_path=config_path,
                tokenizer_ckpt_path=ckpt_path,
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

    def test_sample_latents_is_reproducible_for_same_noise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, ckpt_path, token_spatial_shape, _ = _write_tokenizer_artifacts(tmpdir)
            trainer = _make_prior_trainer(
                tokenizer_config_path=config_path,
                tokenizer_ckpt_path=ckpt_path,
            )
            trainer.eval()
            noise = torch.randn((2, trainer.latent_channels, *trainer.latent_spatial_shape))

            sampled_a = trainer.sample_latents(
                batch_size=2,
                nfe=4,
                device=torch.device("cpu"),
                noise=noise,
            )
            sampled_b = trainer.sample_latents(
                batch_size=2,
                nfe=4,
                device=torch.device("cpu"),
                noise=noise,
            )

            self.assertEqual(tuple(sampled_a.shape), (2, *token_spatial_shape))
            self.assertTrue(torch.equal(sampled_a, sampled_b))

    def test_sampled_token_grids_decode_into_valid_semantic_masks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, ckpt_path, token_spatial_shape, codebook_size = _write_tokenizer_artifacts(tmpdir)
            trainer = _make_prior_trainer(
                tokenizer_config_path=config_path,
                tokenizer_ckpt_path=ckpt_path,
            )

            sampled = trainer.sample_latents(
                batch_size=2,
                nfe=4,
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

    def test_validation_sample_metrics_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, ckpt_path, _, _ = _write_tokenizer_artifacts(tmpdir)
            trainer = _make_prior_trainer(
                tokenizer_config_path=config_path,
                tokenizer_ckpt_path=ckpt_path,
            )

            metrics = trainer._validation_sample_metrics(_make_batch()["mask_index"])

            self.assertIn("sampled_class_hist_l1", metrics)
            self.assertIn("sampled_class_ratio_gap_0", metrics)
            self.assertIn("sampled_monitor_error", metrics)
            self.assertIn("sampled_pred_majority_class_ratio", metrics)
            self.assertIn("sampled_unique_class_count_gap", metrics)
            self.assertIn("sampled_boundary_ratio_gap", metrics)

    def test_validation_sample_metrics_are_averaged_across_batches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, ckpt_path, _, _ = _write_tokenizer_artifacts(tmpdir)
            trainer = _make_prior_trainer(
                tokenizer_config_path=config_path,
                tokenizer_ckpt_path=ckpt_path,
                validation_sample_metric_batches=2,
            )
            trainer._trainer = SimpleNamespace(
                is_global_zero=True,
                global_rank=0,
                world_size=1,
                strategy=None,
            )

            captured = []

            def fake_shared_step(batch, split):
                del split
                return torch.tensor(0.0), torch.tensor(0.0), {"mask_index": batch["mask_index"]}

            def fake_validation_sample_metrics(target_mask_index):
                mean_value = target_mask_index.float().mean()
                return {
                    "sampled_monitor_error": mean_value,
                    "sampled_majority_class_ratio_gap": mean_value + 1.0,
                }

            def capture_log_dict(metrics, **kwargs):
                captured.append((metrics, kwargs))

            trainer.shared_step = fake_shared_step
            trainer._validation_sample_metrics = fake_validation_sample_metrics
            trainer.log_dict = capture_log_dict

            batch_a = {"mask_index": torch.zeros((2, 16, 16), dtype=torch.long)}
            batch_b = {"mask_index": torch.full((2, 16, 16), 2, dtype=torch.long)}
            batch_c = {"mask_index": torch.full((2, 16, 16), 6, dtype=torch.long)}

            trainer.on_validation_epoch_start()
            trainer.validation_step(batch_a, batch_idx=0)
            trainer.validation_step(batch_b, batch_idx=1)
            trainer.validation_step(batch_c, batch_idx=2)
            trainer.on_validation_epoch_end()

            self.assertEqual(len(captured), 1)
            logged_metrics, logged_kwargs = captured[0]
            self.assertAlmostEqual(float(logged_metrics["val/sampled_monitor_error"].item()), 1.0)
            self.assertAlmostEqual(float(logged_metrics["val/sampled_majority_class_ratio_gap"].item()), 2.0)
            self.assertEqual(int(logged_kwargs["batch_size"]), 8)
            self.assertEqual(trainer._validation_sample_metric_batches_seen, 0)

    def test_sampled_monitor_requires_validation_sample_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, ckpt_path, _, _ = _write_tokenizer_artifacts(tmpdir)
            with self.assertRaisesRegex(ValueError, "monitor=val/sampled_monitor_error requires"):
                _make_prior_trainer(
                    tokenizer_config_path=config_path,
                    tokenizer_ckpt_path=ckpt_path,
                    enable_validation_sample_metrics=False,
                )

    def test_main_configs_pin_balanced_tokenizer_and_identity_permuter(self):
        expected_backbones = {
            REPO_ROOT / "configs" / "token_code_maskgit_tiny.yaml": {
                "n_layer": 4,
                "n_head": 4,
                "n_embd": 128,
            },
            REPO_ROOT / "configs" / "token_code_maskgit.yaml": {
                "n_layer": 8,
                "n_head": 8,
                "n_embd": 256,
            },
        }
        for config_path, expected in expected_backbones.items():
            config = OmegaConf.load(config_path)
            self.assertEqual(
                config.model.target,
                "latent_meanflow.trainers.token_code_maskgit_prior_trainer.TokenCodeMaskGitPriorTrainer",
            )
            self.assertEqual(
                str(config.model.params.tokenizer_config_path),
                "configs/semantic_mask_vq_tokenizer_main_balanced_256.yaml",
            )
            self.assertEqual(str(config.model.params.monitor), "val/sampled_monitor_error")
            self.assertIsNone(config.model.params.tokenizer_ckpt_path)
            self.assertTrue(bool(config.model.params.freeze_tokenizer))
            self.assertFalse(bool(config.model.params.tokenizer_sample_posterior))
            self.assertEqual(str(config.model.params.objective_name), "token_code_maskgit")
            self.assertEqual(
                str(config.model.params.permuter_config.target),
                "taming.modules.transformer.permuter.Identity",
            )
            backbone_params = config.model.params.backbone_config.params
            self.assertEqual(int(backbone_params.n_layer), expected["n_layer"])
            self.assertEqual(int(backbone_params.n_head), expected["n_head"])
            self.assertEqual(int(backbone_params.n_embd), expected["n_embd"])
            self.assertEqual(str(backbone_params.mask_schedule_type), "cosine")

    def test_config_instantiation_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, ckpt_path, _, _ = _write_tokenizer_artifacts(tmpdir)
            config = OmegaConf.load(REPO_ROOT / "configs" / "token_code_maskgit_tiny.yaml")
            OmegaConf.update(config, "model.params.tokenizer_config_path", str(config_path), merge=False)
            OmegaConf.update(config, "model.params.tokenizer_ckpt_path", str(ckpt_path), merge=False)

            trainer = instantiate_from_config(config.model)
            outputs = trainer(_make_batch())
            self.assertEqual(outputs["loss"].ndim, 0)
            self.assertIn("maskgit_ce", outputs["loss_dict"])
            self.assertIn("masked_token_accuracy", outputs["maskgit_metrics"])

    def test_route_metadata_reports_maskgit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path, ckpt_path, _, _ = _write_tokenizer_artifacts(tmpdir)
            config = OmegaConf.load(REPO_ROOT / "configs" / "token_code_maskgit_tiny.yaml")
            OmegaConf.update(config, "model.params.tokenizer_config_path", str(config_path), merge=False)
            OmegaConf.update(config, "model.params.tokenizer_ckpt_path", str(ckpt_path), merge=False)
            trainer = instantiate_from_config(config.model)

            metadata = extract_token_mask_prior_route_metadata(config=config, model=trainer)

            self.assertEqual(metadata["refinement_mode"], "canonical_maskgit")
            self.assertEqual(metadata["corruption_mode"], "masked_token_ce")
            self.assertFalse(bool(metadata["nfe_ignored"]))
            self.assertEqual(metadata["permuter"], "Identity")

    def test_resolved_tokenizer_artifacts_follow_runtime_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer_dir = Path(tmpdir) / "tokenizer_a"
            tokenizer_dir.mkdir(parents=True, exist_ok=True)
            tokenizer_config_path, tokenizer_ckpt_path, _, _ = _write_tokenizer_artifacts(tokenizer_dir)
            config = OmegaConf.load(REPO_ROOT / "configs" / "token_code_maskgit_tiny.yaml")
            OmegaConf.update(config, "model.params.tokenizer_config_path", str(tokenizer_config_path), merge=False)
            OmegaConf.update(config, "model.params.tokenizer_ckpt_path", str(tokenizer_ckpt_path), merge=False)

            resolved_config_path, resolved_ckpt_path = resolve_configured_tokenizer_artifacts(
                config,
                route_name="Token-code MaskGIT smoke test",
            )
            self.assertEqual(resolved_config_path, Path(tokenizer_config_path).resolve())
            self.assertEqual(resolved_ckpt_path, Path(tokenizer_ckpt_path).resolve())

    def test_runtime_tokenizer_override_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer_dir_a = Path(tmpdir) / "tokenizer_a"
            tokenizer_dir_b = Path(tmpdir) / "tokenizer_b"
            tokenizer_dir_a.mkdir(parents=True, exist_ok=True)
            tokenizer_dir_b.mkdir(parents=True, exist_ok=True)
            tokenizer_config_a, tokenizer_ckpt_a, _, _ = _write_tokenizer_artifacts(tokenizer_dir_a)
            tokenizer_config_b, tokenizer_ckpt_b, _, _ = _write_tokenizer_artifacts(tokenizer_dir_b)
            prior_ckpt_path = _write_runtime_ckpt(
                tmpdir,
                tokenizer_config_path=tokenizer_config_a,
                tokenizer_ckpt_path=tokenizer_ckpt_a,
            )

            config = OmegaConf.load(REPO_ROOT / "configs" / "token_code_maskgit_tiny.yaml")
            OmegaConf.update(config, "model.params.tokenizer_config_path", str(tokenizer_config_b), merge=False)
            OmegaConf.update(config, "model.params.tokenizer_ckpt_path", str(tokenizer_ckpt_b), merge=False)

            with self.assertRaisesRegex(ValueError, "resolved-config mismatch"):
                validate_token_mask_prior_checkpoint_contract(
                    config,
                    prior_ckpt_path,
                    config_path=REPO_ROOT / "configs" / "token_code_maskgit_tiny.yaml",
                )


if __name__ == "__main__":
    unittest.main()

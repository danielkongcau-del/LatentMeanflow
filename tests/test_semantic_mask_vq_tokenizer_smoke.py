import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

from latent_meanflow.data.subset import FixedSubsetDataset
from latent_meanflow.models.semantic_mask_vq_autoencoder import SemanticMaskVQAutoencoder


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_model(loss_params=None, quantizer_params=None):
    loss_params = dict(loss_params or {})
    quantizer_params = dict(quantizer_params or {})
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
                **loss_params,
            },
        },
        embed_dim=8,
        codebook_size=32,
        num_classes=4,
        quantizer_config=quantizer_params,
    )


def _make_batch():
    mask_index = torch.zeros((2, 16, 16), dtype=torch.long)
    mask_index[:, :8, :8] = 1
    mask_index[:, :8, 8:] = 2
    mask_index[:, 8:, :] = 3
    mask_onehot = F.one_hot(mask_index.clamp_min(0), num_classes=4).float()
    return {
        "mask_index": mask_index,
        "mask_onehot": mask_onehot,
        "num_classes": torch.tensor([4, 4], dtype=torch.long),
    }


class _ToyTokenizerMaskDataset:
    def __init__(self, length=6, num_classes=4):
        self.length = int(length)
        self.num_classes = int(num_classes)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        index = int(idx) % self.num_classes
        mask_index = torch.full((8, 8), fill_value=index, dtype=torch.long)
        mask_onehot = F.one_hot(mask_index, num_classes=self.num_classes).float()
        return {
            "mask_index": mask_index,
            "mask_onehot": mask_onehot,
            "num_classes": int(self.num_classes),
            "metadata": {"source_index": int(idx)},
        }


class SemanticMaskVQTokenizerSmokeTest(unittest.TestCase):
    def test_discrete_tokenizer_forward_smoke(self):
        model = _make_model()
        outputs = model(_make_batch())

        self.assertEqual(outputs["total_loss"].ndim, 0)
        self.assertEqual(tuple(outputs["mask_logits"].shape), (2, 4, 16, 16))
        self.assertEqual(tuple(outputs["recon_mask_index"].shape), (2, 16, 16))
        self.assertEqual(tuple(outputs["codes"].shape), (2, *model.latent_spatial_shape))
        self.assertEqual(tuple(outputs["z_q"].shape), (2, model.latent_channels, *model.latent_spatial_shape))

    def test_encode_decode_contract(self):
        model = _make_model()
        batch = _make_batch()

        encoded = model.encode_batch(batch, sample_posterior=False)
        self.assertEqual(set(encoded.keys()), {"z_e", "z_q", "codes", "mask_index", "mask_onehot", "quantizer_stats"})

        decoded_from_codes = model.decode_codes(encoded["codes"])
        decoded_from_latents = model.decode_latents(encoded["z_q"])
        self.assertEqual(set(decoded_from_codes.keys()), {"mask_logits", "mask_probs", "mask_index", "mask_onehot", "z_q", "codes"})
        self.assertEqual(set(decoded_from_latents.keys()), {"mask_logits", "mask_probs", "mask_index", "mask_onehot"})
        self.assertTrue(torch.equal(decoded_from_codes["mask_index"], decoded_from_latents["mask_index"]))

    def test_ignore_index_handling_is_safe_for_reconstruction_and_vq_losses(self):
        model = _make_model(
            loss_params={
                "mask_dice_weight": 1.0,
                "mask_focal_weight": 1.0,
            }
        )
        batch = _make_batch()
        batch["mask_index"] = batch["mask_index"].clone()
        batch["mask_index"][:, 0, 0] = -1
        batch["mask_onehot"] = batch["mask_onehot"].clone()
        batch["mask_onehot"][:, 0, 0, :] = 0.0

        outputs = model(batch)
        self.assertTrue(torch.isfinite(outputs["total_loss"]))
        self.assertTrue(torch.isfinite(outputs["loss_dict"]["mask_ce"]))
        self.assertTrue(torch.isfinite(outputs["loss_dict"]["mask_dice"]))
        self.assertTrue(torch.isfinite(outputs["loss_dict"]["mask_focal"]))
        self.assertTrue(torch.isfinite(outputs["loss_dict"]["vq_codebook"]))
        self.assertTrue(torch.isfinite(outputs["loss_dict"]["vq_commit"]))

    def test_stable_quantizer_forward_smoke(self):
        model = _make_model(
            quantizer_params={
                "distance_metric": "cosine",
                "use_ema_update": True,
                "ema_decay": 0.99,
                "ema_eps": 1.0e-5,
                "dead_code_threshold": 1.0,
            }
        )
        outputs = model(_make_batch())
        self.assertTrue(torch.isfinite(outputs["total_loss"]))
        self.assertEqual(float(outputs["loss_dict"]["vq_codebook"].item()), 0.0)
        self.assertGreaterEqual(int(outputs["quantizer_stats"]["used_code_count"].item()), 1)

    def test_strict_mask_index_mask_onehot_consistency(self):
        model = _make_model()
        batch = _make_batch()
        model(batch)

        bad_batch = _make_batch()
        bad_batch["mask_onehot"] = bad_batch["mask_onehot"].clone()
        bad_batch["mask_onehot"][0, 0, 0, :] = 0.0
        bad_batch["mask_onehot"][0, 0, 0, 2] = 1.0
        with self.assertRaises(ValueError):
            model(bad_batch)

    def test_memorize_config_contract(self):
        dataset = FixedSubsetDataset(
            dataset_config={
                "target": f"{__name__}._ToyTokenizerMaskDataset",
                "params": {"length": 6, "num_classes": 4},
            },
            first_n=4,
        )
        self.assertEqual(len(dataset), 4)

        config = OmegaConf.load(REPO_ROOT / "configs" / "diagnostics" / "semantic_mask_vq_tokenizer_memorize_4_256.yaml")
        self.assertEqual(
            config.model.target,
            "latent_meanflow.models.semantic_mask_vq_autoencoder.SemanticMaskVQAutoencoder",
        )
        self.assertEqual(config.data.params.train.target, "latent_meanflow.data.subset.FixedSubsetDataset")
        self.assertEqual(
            config.data.params.train.params.dataset_config.target,
            "latent_meanflow.data.semantic_mask.MultiSemanticMaskDataset",
        )
        model = instantiate_from_config(config.model)
        self.assertEqual(model.codebook_size, int(config.model.params.codebook_size))

    def test_hifi_memorize_config_contract(self):
        config = OmegaConf.load(REPO_ROOT / "configs" / "diagnostics" / "semantic_mask_vq_tokenizer_memorize_4_hifi_256.yaml")
        self.assertEqual(
            config.model.target,
            "latent_meanflow.models.semantic_mask_vq_autoencoder.SemanticMaskVQAutoencoder",
        )
        self.assertEqual(config.data.params.train.target, "latent_meanflow.data.subset.FixedSubsetDataset")
        self.assertEqual(str(config.model.params.monitor), "val/mask_ce")
        self.assertEqual(int(config.model.params.codebook_size), 512)
        self.assertEqual(list(config.model.params.ddconfig.ch_mult), [1, 2])
        self.assertEqual(str(config.model.params.quantizer_config.distance_metric), "cosine")
        self.assertTrue(bool(config.model.params.quantizer_config.use_ema_update))
        self.assertEqual(float(config.lightning.trainer.gradient_clip_val), 1.0)

        model = instantiate_from_config(config.model)
        self.assertEqual(model.codebook_size, 512)
        self.assertEqual(model.latent_spatial_shape, (128, 128))
        self.assertTrue(model.quantizer.use_ema_update)

    def test_codes_are_discrete_and_in_range(self):
        model = _make_model()
        encoded = model.encode_batch(_make_batch())
        codes = encoded["codes"]
        self.assertEqual(codes.dtype, torch.long)
        self.assertGreaterEqual(int(codes.min().item()), 0)
        self.assertLess(int(codes.max().item()), int(model.codebook_size))


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from latent_meanflow.data.subset import FixedSubsetDataset
from latent_meanflow.models.semantic_mask_autoencoder import SemanticMaskAutoencoder


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_model(loss_params=None):
    loss_params = dict(loss_params or {})
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
                **loss_params,
            },
        },
        embed_dim=4,
        num_classes=4,
        sample_posterior=True,
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


class SemanticMaskTokenizerSmokeTest(unittest.TestCase):
    def test_mask_only_tokenizer_forward_smoke(self):
        model = _make_model()
        batch = _make_batch()

        outputs = model(batch)
        self.assertEqual(outputs["total_loss"].ndim, 0)
        self.assertEqual(tuple(outputs["mask_logits"].shape), (2, 4, 16, 16))
        self.assertEqual(tuple(outputs["recon_mask_index"].shape), (2, 16, 16))
        self.assertEqual(outputs["z"].shape[0], 2)
        self.assertEqual(outputs["z"].shape[1], model.latent_channels)

    def test_encode_decode_contract(self):
        model = _make_model()
        batch = _make_batch()

        encoded = model.encode_batch(batch, sample_posterior=False)
        self.assertIn("z", encoded)
        self.assertIn("posterior", encoded)
        self.assertIn("mask_index", encoded)
        self.assertIn("mask_onehot", encoded)

        decoded = model.decode_latents(encoded["z"])
        self.assertEqual(set(decoded.keys()), {"mask_logits", "mask_probs", "mask_index", "mask_onehot"})
        self.assertEqual(tuple(decoded["mask_logits"].shape), (2, 4, 16, 16))
        self.assertEqual(tuple(decoded["mask_index"].shape), (2, 16, 16))
        self.assertEqual(tuple(decoded["mask_onehot"].shape), (2, 4, 16, 16))

    def test_ignore_index_handling_is_safe_for_ce_dice_and_focal(self):
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

    def test_fixed_subset_route_and_memorize_config_contract(self):
        dataset = FixedSubsetDataset(
            dataset_config={
                "target": f"{__name__}._ToyTokenizerMaskDataset",
                "params": {"length": 6, "num_classes": 4},
            },
            first_n=4,
        )
        self.assertEqual(len(dataset), 4)
        sample = dataset[0]
        self.assertEqual(set(sample.keys()), {"mask_index", "mask_onehot", "num_classes", "metadata"})

        config = OmegaConf.load(REPO_ROOT / "configs" / "diagnostics" / "semantic_mask_tokenizer_memorize_4_256.yaml")
        self.assertEqual(
            config.model.target,
            "latent_meanflow.models.semantic_mask_autoencoder.SemanticMaskAutoencoder",
        )
        self.assertEqual(
            config.data.params.train.target,
            "latent_meanflow.data.subset.FixedSubsetDataset",
        )
        self.assertEqual(
            config.data.params.train.params.dataset_config.target,
            "latent_meanflow.data.semantic_mask.MultiSemanticMaskDataset",
        )

    def test_mask_tokenizer_configs_follow_mask_only_route(self):
        config_paths = [
            REPO_ROOT / "configs" / "semantic_mask_tokenizer_tiny_256.yaml",
            REPO_ROOT / "configs" / "semantic_mask_tokenizer_mid_256.yaml",
            REPO_ROOT / "configs" / "semantic_mask_tokenizer_mid_plus_256.yaml",
            REPO_ROOT / "configs" / "diagnostics" / "semantic_mask_tokenizer_memorize_1_256.yaml",
            REPO_ROOT / "configs" / "diagnostics" / "semantic_mask_tokenizer_memorize_4_256.yaml",
        ]
        for config_path in config_paths:
            config = OmegaConf.load(config_path)
            self.assertEqual(
                config.model.target,
                "latent_meanflow.models.semantic_mask_autoencoder.SemanticMaskAutoencoder",
            )
            self.assertEqual(
                config.model.params.lossconfig.target,
                "latent_meanflow.models.semantic_mask_autoencoder.SemanticMaskLoss",
            )


if __name__ == "__main__":
    unittest.main()

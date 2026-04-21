import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image

from latent_meanflow.data.semantic_palette import MultiSemanticPaletteMaskDataset
from latent_meanflow.models.maskgit_palette_vq_tokenizer import MaskGitPaletteVQTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_mask_dataset(tmpdir):
    root = Path(tmpdir) / "dataset"
    mask_dir = root / "train" / "masks"
    val_dir = root / "val" / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[:16, :16] = 64
    mask[:16, 16:] = 128
    mask[16:, :] = 255
    Image.fromarray(mask).save(mask_dir / "sample.png")
    Image.fromarray(mask).save(val_dir / "sample.png")
    return root


def _write_pretrained_weights(tmpdir):
    weight_path = Path(tmpdir) / "maskgit_pretrained.bin"
    model = MaskGitPaletteVQTokenizer(
        pretrained_weight_path=None,
        resolution=32,
        num_classes=4,
        ignore_index=-1,
    )
    torch.save(model.state_dict(), weight_path)
    return weight_path


class MaskGitPaletteVQTokenizerSmokeTest(unittest.TestCase):
    def test_palette_dataset_returns_palette_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = _write_mask_dataset(tmpdir)
            dataset = MultiSemanticPaletteMaskDataset(
                roots=[str(dataset_root)],
                gray_to_class_id={0: 0, 64: 1, 128: 2, 255: 3},
                split="train",
                size=32,
                ignore_index=-1,
            )

            sample = dataset[0]
            self.assertIn("palette_image", sample)
            self.assertEqual(tuple(sample["palette_image"].shape), (3, 32, 32))
            self.assertGreaterEqual(float(sample["palette_image"].min()), 0.0)
            self.assertLessEqual(float(sample["palette_image"].max()), 1.0)
            self.assertEqual(tuple(sample["mask_index"].shape), (32, 32))

    def test_forward_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pretrained_path = _write_pretrained_weights(tmpdir)
            model = MaskGitPaletteVQTokenizer(
                pretrained_weight_path=str(pretrained_path),
                resolution=32,
                num_classes=4,
                ignore_index=-1,
            )

            mask_index = torch.zeros((1, 32, 32), dtype=torch.long)
            mask_index[:, :16, :16] = 1
            mask_index[:, :16, 16:] = 2
            mask_index[:, 16:, :] = 3
            palette_image = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
            palette_image[:, 0] = 0.25
            outputs = model({"palette_image": palette_image, "mask_index": mask_index})

            self.assertEqual(outputs["loss"].ndim, 0)
            self.assertIn("semantic_ce", outputs["loss_dict"])
            self.assertIn("semantic_dice", outputs["loss_dict"])
            self.assertIn("boundary_loss", outputs["loss_dict"])
            self.assertIn("vq_loss", outputs["loss_dict"])
            self.assertIn("pixel_accuracy", outputs["metrics"])
            self.assertIn("miou", outputs["metrics"])
            self.assertEqual(tuple(outputs["reconstructed_image"].shape), (1, 3, 32, 32))
            self.assertEqual(tuple(outputs["pred_mask_index"].shape), (1, 32, 32))
            self.assertEqual(tuple(outputs["code_indices"].shape), (1, 4))

    def test_config_instantiation_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pretrained_path = _write_pretrained_weights(tmpdir)
            dataset_root = _write_mask_dataset(tmpdir)
            config = OmegaConf.load(REPO_ROOT / "configs" / "maskgit_palette_vq_tokenizer_imagenet_f16_256.yaml")
            OmegaConf.update(config, "model.params.pretrained_weight_path", str(pretrained_path), merge=False)
            OmegaConf.update(config, "model.params.resolution", 32, merge=False)
            OmegaConf.update(config, "model.params.num_classes", 4, merge=False)
            OmegaConf.update(config, "data.params.train.params.roots", [str(dataset_root)], merge=False)
            OmegaConf.update(config, "data.params.validation.params.roots", [str(dataset_root)], merge=False)
            OmegaConf.update(config, "data.params.train.params.gray_to_class_id", {0: 0, 64: 1, 128: 2, 255: 3}, merge=False)
            OmegaConf.update(config, "data.params.validation.params.gray_to_class_id", {0: 0, 64: 1, 128: 2, 255: 3}, merge=False)
            model = instantiate_from_config(config.model)
            outputs = model(
                {
                    "palette_image": torch.zeros((1, 3, 32, 32), dtype=torch.float32),
                    "mask_index": torch.zeros((1, 32, 32), dtype=torch.long),
                }
            )
            self.assertEqual(outputs["loss"].ndim, 0)
            self.assertEqual(str(config.model.target), "latent_meanflow.models.maskgit_palette_vq_tokenizer.MaskGitPaletteVQTokenizer")
            self.assertEqual(str(config.data.params.train.target), "latent_meanflow.data.semantic_palette.MultiSemanticPaletteMaskDataset")


if __name__ == "__main__":
    unittest.main()

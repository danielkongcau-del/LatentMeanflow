import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LDM_ROOT = REPO_ROOT / "third_party" / "latent-diffusion"
TAMING_ROOT = LDM_ROOT / "taming-transformers"

for path in (REPO_ROOT, LDM_ROOT, TAMING_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from latent_meanflow.data.semantic_pair import MultiSemanticImageMaskPairDataset
from latent_meanflow.callbacks.semantic_logger import SemanticPairImageLogger
from latent_meanflow.models.semantic_autoencoder import SemanticPairAutoencoder


def write_sample(root, stem, rgb_value, mask_values):
    image_dir = root / "train" / "images"
    mask_dir = root / "train" / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image = np.full((32, 32, 3), rgb_value, dtype=np.uint8)
    mask = np.array(mask_values, dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(image_dir / f"{stem}.png")
    Image.fromarray(mask, mode="L").save(mask_dir / f"{stem}.png")


def build_synthetic_dataset(root):
    mask_a = np.tile(np.array([[0, 64], [128, 255]], dtype=np.uint8), (16, 16))
    mask_b = np.tile(np.array([[255, 128], [64, 0]], dtype=np.uint8), (16, 16))
    write_sample(root, "sample_a", 64, mask_a)
    write_sample(root, "sample_b", 160, mask_b)


def main():
    torch.manual_seed(0)

    with tempfile.TemporaryDirectory(prefix="semantic_autoencoder_smoke_") as tmpdir:
        root = Path(tmpdir)
        build_synthetic_dataset(root)

        dataset = MultiSemanticImageMaskPairDataset(
            roots=[root],
            split="train",
            size=32,
            image_dir="images",
            mask_dir="masks",
            gray_to_class_id={0: 0, 64: 1, 128: 2, 255: 3},
            ignore_index=-1,
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(loader))

        model = SemanticPairAutoencoder(
            ddconfig={
                "double_z": True,
                "z_channels": 4,
                "resolution": 32,
                "in_channels": 7,
                "out_ch": 7,
                "ch": 32,
                "ch_mult": [1, 2],
                "num_res_blocks": 1,
                "attn_resolutions": [],
                "dropout": 0.0,
            },
            lossconfig={
                "target": "latent_meanflow.models.semantic_autoencoder.SemanticPairLoss",
                "params": {
                    "rgb_l1_weight": 1.0,
                    "rgb_lpips_weight": 0.0,
                    "mask_ce_weight": 1.0,
                    "mask_dice_weight": 0.0,
                    "mask_focal_weight": 0.0,
                    "kl_weight": 1.0e-6,
                    "ignore_index": -1,
                },
            },
            embed_dim=4,
            num_classes=4,
            sample_posterior=True,
        )

        outputs = model(batch)
        total_loss = outputs["total_loss"]
        total_loss.backward()

        assert outputs["z"].shape[0] == 2
        assert outputs["rgb_recon"].shape == (2, 3, 32, 32)
        assert outputs["mask_logits"].shape == (2, 4, 32, 32)
        assert outputs["posterior"].mean.shape[1] == 4
        assert total_loss.ndim == 0
        assert model.rgb_head[-1].weight.grad is not None
        assert model.mask_head[-1].weight.grad is not None

        images = model.log_images(batch, sample_posterior=False)
        assert images["inputs_image"].shape == (2, 3, 32, 32)
        assert images["reconstructions_image"].shape == (2, 3, 32, 32)
        assert images["inputs_mask_index"].shape == (2, 1, 32, 32)
        assert images["reconstructions_mask_index"].shape == (2, 1, 32, 32)

        logger = SemanticPairImageLogger(
            batch_frequency=1,
            max_images=2,
            disabled=False,
            latest_only=True,
            ignore_index=-1,
        )
        logger.save_local(
            save_dir=str(root),
            split="train",
            images=images,
            global_step=0,
            current_epoch=0,
            batch_idx=0,
            num_classes=4,
        )
        assert (root / "semantic_images" / "train" / "inputs_image.png").exists()
        assert (root / "semantic_images" / "train" / "reconstructions_image.png").exists()
        assert (root / "semantic_images" / "train" / "inputs_mask_index.png").exists()
        assert (root / "semantic_images" / "train" / "reconstructions_mask_index.png").exists()

        print("Semantic autoencoder smoke test passed")
        print(f"z shape: {tuple(outputs['z'].shape)}")
        print(f"rgb_recon shape: {tuple(outputs['rgb_recon'].shape)}")
        print(f"mask_logits shape: {tuple(outputs['mask_logits'].shape)}")
        print(f"total_loss: {float(total_loss.detach().cpu()):.6f}")


if __name__ == "__main__":
    main()

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
LDM_ROOT = REPO_ROOT / "third_party" / "latent-diffusion"
TAMING_ROOT = LDM_ROOT / "taming-transformers"

for path in (REPO_ROOT, LDM_ROOT, TAMING_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from latent_meanflow.data.semantic_pair import MultiSemanticImageMaskPairDataset
from latent_meanflow.models.semantic_autoencoder import SemanticPairAutoencoder
from latent_meanflow.trainers.latent_fm_trainer import LatentFMTrainer


def write_sample(root, split, stem, rgb_value, mask_values):
    image_dir = root / split / "images"
    mask_dir = root / split / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image = np.full((32, 32, 3), rgb_value, dtype=np.uint8)
    mask = np.array(mask_values, dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(image_dir / f"{stem}.png")
    Image.fromarray(mask, mode="L").save(mask_dir / f"{stem}.png")


def build_dataset(root):
    mask_a = np.tile(np.array([[0, 64], [128, 255]], dtype=np.uint8), (16, 16))
    mask_b = np.tile(np.array([[255, 128], [64, 0]], dtype=np.uint8), (16, 16))
    for split in ["train", "val"]:
        write_sample(root, split, "sample_a", 64, mask_a)
        write_sample(root, split, "sample_b", 160, mask_b)


def build_tokenizer_ckpt(root):
    tokenizer = SemanticPairAutoencoder(
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
        sample_posterior=False,
    )
    ckpt_path = root / "tokenizer.ckpt"
    torch.save({"state_dict": tokenizer.state_dict()}, ckpt_path)
    return ckpt_path


def write_tokenizer_config(root):
    config_path = root / "tokenizer.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  target: latent_meanflow.models.semantic_autoencoder.SemanticPairAutoencoder",
                "  params:",
                "    embed_dim: 4",
                "    num_classes: 4",
                "    sample_posterior: false",
                "    lossconfig:",
                "      target: latent_meanflow.models.semantic_autoencoder.SemanticPairLoss",
                "      params:",
                "        rgb_l1_weight: 1.0",
                "        rgb_lpips_weight: 0.0",
                "        mask_ce_weight: 1.0",
                "        mask_dice_weight: 0.0",
                "        mask_focal_weight: 0.0",
                "        kl_weight: 1.0e-6",
                "        ignore_index: -1",
                "    ddconfig:",
                "      double_z: true",
                "      z_channels: 4",
                "      resolution: 32",
                "      in_channels: 7",
                "      out_ch: 7",
                "      ch: 32",
                "      ch_mult: [1, 2]",
                "      num_res_blocks: 1",
                "      attn_resolutions: []",
                "      dropout: 0.0",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def main():
    torch.manual_seed(0)

    with tempfile.TemporaryDirectory(prefix="latent_fm_smoke_") as tmpdir:
        root = Path(tmpdir)
        data_root = root / "data_root"
        build_dataset(data_root)
        tokenizer_ckpt = build_tokenizer_ckpt(root)
        tokenizer_config = write_tokenizer_config(root)

        dataset = MultiSemanticImageMaskPairDataset(
            roots=[data_root],
            split="train",
            size=32,
            image_dir="images",
            mask_dir="masks",
            gray_to_class_id={0: 0, 64: 1, 128: 2, 255: 3},
            ignore_index=-1,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(loader))

        model = LatentFMTrainer(
            tokenizer_config_path=tokenizer_config,
            tokenizer_ckpt_path=tokenizer_ckpt,
            backbone_config={
                "target": "latent_meanflow.models.backbones.latent_velocity_convnet.LatentVelocityConvNet",
                "params": {
                    "model_channels": 32,
                    "time_embed_dim": 64,
                    "num_res_blocks": 2,
                    "dropout": 0.0,
                },
            },
            objective_config={
                "target": "latent_meanflow.objectives.flow_matching.RectifiedFlowMatchingObjective",
                "params": {"time_eps": 1.0e-4, "loss_type": "mse"},
            },
            sample_posterior=False,
            freeze_tokenizer=True,
            use_class_condition=False,
            log_sample_steps=4,
        )

        outputs = model(batch)
        outputs["loss"].backward()
        assert outputs["x_lat"].shape == (2, 4, 16, 16)
        assert outputs["pred_velocity"].shape == outputs["target_velocity"].shape
        assert model.backbone.conv_out.weight.grad is not None

        sampled_latents = model.sample_latents(batch_size=2, num_steps=4, device=torch.device("cpu"))
        decoded = model.decode_latents(sampled_latents)
        assert decoded["rgb_recon"].shape == (2, 3, 32, 32)
        assert decoded["mask_logits"].shape == (2, 4, 32, 32)
        assert decoded["mask_index"].shape == (2, 32, 32)

        fm_ckpt = root / "latent_fm.ckpt"
        torch.save({"state_dict": model.state_dict()}, fm_ckpt)

        fm_config = root / "latent_fm.yaml"
        fm_config.write_text(
            "\n".join(
                [
                    "model:",
                    "  target: latent_meanflow.trainers.latent_fm_trainer.LatentFMTrainer",
                    "  params:",
                    f"    tokenizer_config_path: {tokenizer_config.as_posix()}",
                    f"    tokenizer_ckpt_path: {tokenizer_ckpt.as_posix()}",
                    "    sample_posterior: false",
                    "    freeze_tokenizer: true",
                    "    use_class_condition: false",
                    "    class_label_key: class_label",
                    "    log_sample_steps: 4",
                    "    backbone_config:",
                    "      target: latent_meanflow.models.backbones.latent_velocity_convnet.LatentVelocityConvNet",
                    "      params:",
                    "        model_channels: 32",
                    "        time_embed_dim: 64",
                    "        num_res_blocks: 2",
                    "        dropout: 0.0",
                    "    objective_config:",
                    "      target: latent_meanflow.objectives.flow_matching.RectifiedFlowMatchingObjective",
                    "      params:",
                    "        time_eps: 1.0e-4",
                    "        loss_type: mse",
                ]
            ),
            encoding="utf-8",
        )

        outdir = root / "samples"
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "sample_latent_fm.py"),
            "--config",
            str(fm_config),
            "--ckpt",
            str(fm_ckpt),
            "--outdir",
            str(outdir),
            "--n-samples",
            "2",
            "--batch-size",
            "2",
            "--steps",
            "4",
        ]
        subprocess.run(command, cwd=str(REPO_ROOT), check=True)

        assert (outdir / "image" / "000000.png").exists()
        assert (outdir / "mask_raw" / "000000.png").exists()
        assert (outdir / "mask_color" / "000000.png").exists()
        assert (outdir / "overlay" / "000000.png").exists()

        print("Latent FM smoke test passed")
        print(f"x_lat shape: {tuple(outputs['x_lat'].shape)}")
        print(f"pred_velocity shape: {tuple(outputs['pred_velocity'].shape)}")
        print(f"sampled rgb shape: {tuple(decoded['rgb_recon'].shape)}")
        print(f"sampled mask shape: {tuple(decoded['mask_index'].shape)}")


if __name__ == "__main__":
    main()

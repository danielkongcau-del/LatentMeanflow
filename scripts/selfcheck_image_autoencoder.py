import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
LDM_ROOT = REPO_ROOT / "third_party" / "latent-diffusion"
TAMING_ROOT = LDM_ROOT / "taming-transformers"

for path in (REPO_ROOT, LDM_ROOT, TAMING_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ldm.util import instantiate_from_config
from latent_meanflow.data.semantic_pair import MultiSemanticImageMaskPairDataset
from latent_meanflow.models.image_autoencoder import ImageAutoencoder


def write_sample(root, split, stem, image_value, mask_values):
    image_dir = root / split / "images"
    mask_dir = root / split / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[..., 0] = image_value
    image[..., 1] = np.linspace(0, 255, 32, dtype=np.uint8)[None, :]
    image[..., 2] = np.linspace(255, 0, 32, dtype=np.uint8)[:, None]
    mask = np.array(mask_values, dtype=np.uint8)

    Image.fromarray(image, mode="RGB").save(image_dir / f"{stem}.png")
    Image.fromarray(mask, mode="L").save(mask_dir / f"{stem}.png")


def build_synthetic_dataset(root):
    mask_a = np.tile(np.array([[36, 73], [109, 146]], dtype=np.uint8), (16, 16))
    mask_b = np.tile(np.array([[182, 219], [255, 36]], dtype=np.uint8), (16, 16))
    write_sample(root, "train", "sample_a", 64, mask_a)
    write_sample(root, "train", "sample_b", 160, mask_b)
    write_sample(root, "val", "sample_c", 96, mask_a)
    write_sample(root, "val", "sample_d", 192, mask_b)


def _instantiate_checked_in_configs():
    expectations = {
        "configs/autoencoder_image_256.yaml": (64, 64),
        "configs/autoencoder_image_f8_256.yaml": (32, 32),
        "configs/autoencoder_image_24gb_lpips_256.yaml": (64, 64),
        "configs/autoencoder_image_f8_24gb_lpips_256.yaml": (32, 32),
        "configs/autoencoder_image_adv_256.yaml": (64, 64),
        "configs/autoencoder_image_lpips_adv_256.yaml": (64, 64),
        "configs/autoencoder_image_f8_adv_256.yaml": (32, 32),
        "configs/autoencoder_image_f8_lpips_adv_256.yaml": (32, 32),
    }
    for rel_path, expected_spatial_shape in expectations.items():
        config = OmegaConf.load(REPO_ROOT / rel_path)
        model = instantiate_from_config(config.model)
        spatial_shape = tuple(int(v) for v in model.latent_spatial_shape)
        if spatial_shape != expected_spatial_shape:
            raise AssertionError(
                f"{rel_path} expected latent_spatial_shape {expected_spatial_shape}, got {spatial_shape}"
            )


def _run_forward_backward():
    with tempfile.TemporaryDirectory(prefix="image_autoencoder_smoke_") as tmpdir:
        root = Path(tmpdir)
        build_synthetic_dataset(root)

        dataset = MultiSemanticImageMaskPairDataset(
            roots=[root],
            split="train",
            size=32,
            image_dir="images",
            mask_dir="masks",
            gray_to_class_id={36: 0, 73: 1, 109: 2, 146: 3, 182: 4, 219: 5, 255: 6},
            ignore_index=-1,
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(loader))

        model = ImageAutoencoder(
            ddconfig={
                "double_z": True,
                "z_channels": 4,
                "resolution": 32,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 32,
                "ch_mult": [1, 2],
                "num_res_blocks": 1,
                "attn_resolutions": [],
                "dropout": 0.0,
            },
            lossconfig={
                "target": "latent_meanflow.models.image_autoencoder.ImageAutoencoderLoss",
                "params": {
                    "rgb_l1_weight": 1.0,
                    "rgb_lpips_weight": 0.0,
                    "kl_weight": 1.0e-6,
                },
            },
            embed_dim=4,
            sample_posterior=True,
        )

        outputs = model(batch)
        total_loss = outputs["total_loss"]
        total_loss.backward()

        assert outputs["z"].shape[0] == 2
        assert outputs["rgb_recon"].shape == (2, 3, 32, 32)
        assert outputs["posterior"].mean.shape[1] == 4
        assert total_loss.ndim == 0
        assert model.rgb_head[-1].weight.grad is not None

        images = model.log_images(batch, sample_posterior=False)
        assert images["inputs_image"].shape == (2, 3, 32, 32)
        assert images["reconstructions_image"].shape == (2, 3, 32, 32)


def _run_adversarial_config_smoke():
    with tempfile.TemporaryDirectory(prefix="image_autoencoder_adv_smoke_") as tmpdir:
        root = Path(tmpdir)
        build_synthetic_dataset(root)

        dataset = MultiSemanticImageMaskPairDataset(
            roots=[root],
            split="train",
            size=32,
            image_dir="images",
            mask_dir="masks",
            gray_to_class_id={36: 0, 73: 1, 109: 2, 146: 3, 182: 4, 219: 5, 255: 6},
            ignore_index=-1,
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(loader))

        model = ImageAutoencoder(
            ddconfig={
                "double_z": True,
                "z_channels": 4,
                "resolution": 32,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 32,
                "ch_mult": [1, 2],
                "num_res_blocks": 1,
                "attn_resolutions": [],
                "dropout": 0.0,
            },
            lossconfig={
                "target": "latent_meanflow.models.image_autoencoder.ImageAutoencoderLoss",
                "params": {
                    "rgb_l1_weight": 1.0,
                    "rgb_lpips_weight": 0.0,
                    "kl_weight": 1.0e-6,
                    "latent_channel_std_floor_weight": 0.1,
                    "latent_channel_std_floor": 0.2,
                    "latent_utilization_threshold": 0.05,
                },
            },
            embed_dim=4,
            sample_posterior=True,
            generator_adversarial_weight=0.05,
            discriminator_start_step=0,
            discriminator_config={
                "target": "latent_meanflow.models.image_autoencoder.PatchDiscriminator",
                "params": {
                    "in_channels": 3,
                    "base_channels": 16,
                    "max_channels": 64,
                    "num_layers": 3,
                    "use_spectral_norm": False,
                },
            },
        )

        outputs = model(batch)
        if "latent_std_floor_penalty" not in outputs["loss_dict"]:
            raise AssertionError("ImageAutoencoder loss_dict is missing latent anti-collapse metrics")
        optimizers = model.configure_optimizers()
        if not isinstance(optimizers, list) or len(optimizers) != 2:
            raise AssertionError("Adversarial ImageAutoencoder should expose two optimizers")


def _run_eval_script_smoke():
    with tempfile.TemporaryDirectory(prefix="image_tokenizer_eval_") as tmpdir:
        root = Path(tmpdir)
        build_synthetic_dataset(root)

        config = OmegaConf.create(
            {
                "model": {
                    "target": "latent_meanflow.models.image_autoencoder.ImageAutoencoder",
                    "params": {
                        "monitor": "val/total_loss",
                        "embed_dim": 4,
                        "sample_posterior": True,
                        "lossconfig": {
                            "target": "latent_meanflow.models.image_autoencoder.ImageAutoencoderLoss",
                            "params": {
                                "rgb_l1_weight": 1.0,
                                "rgb_lpips_weight": 0.0,
                                "kl_weight": 1.0e-6,
                            },
                        },
                        "ddconfig": {
                            "double_z": True,
                            "z_channels": 4,
                            "resolution": 32,
                            "in_channels": 3,
                            "out_ch": 3,
                            "ch": 32,
                            "ch_mult": [1, 2],
                            "num_res_blocks": 1,
                            "attn_resolutions": [],
                            "dropout": 0.0,
                        },
                    },
                },
                "data": {
                    "params": {
                        "batch_size": 1,
                        "num_workers": 0,
                        "validation": {
                            "target": "latent_meanflow.data.semantic_pair.MultiSemanticImageMaskPairDataset",
                            "params": {
                                "roots": [str(root)],
                                "split": "val",
                                "size": 32,
                                "image_dir": "images",
                                "mask_dir": "masks",
                                "gray_to_class_id": "configs/label_specs/remote_semantic.yaml",
                                "ignore_index": -1,
                            },
                        },
                        "train": {
                            "target": "latent_meanflow.data.semantic_pair.MultiSemanticImageMaskPairDataset",
                            "params": {
                                "roots": [str(root)],
                                "split": "train",
                                "size": 32,
                                "image_dir": "images",
                                "mask_dir": "masks",
                                "gray_to_class_id": "configs/label_specs/remote_semantic.yaml",
                                "ignore_index": -1,
                            },
                        },
                    }
                },
            }
        )
        config_path = root / "image_autoencoder_smoke.yaml"
        OmegaConf.save(config, config_path)

        model = instantiate_from_config(config.model)
        ckpt_path = root / "image_autoencoder.ckpt"
        torch.save({"state_dict": model.state_dict()}, ckpt_path)

        outdir = root / "eval_out"
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "eval_image_tokenizer.py"),
                "--config",
                str(config_path),
                "--ckpt",
                str(ckpt_path),
                "--outdir",
                str(outdir),
                "--max-batches",
                "1",
            ],
            check=True,
            cwd=REPO_ROOT,
        )

        summary = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
        candidate = summary["summaries"][0]
        if "rgb_l1" not in candidate or "latent_mean" not in candidate or "latent_shape" not in candidate:
            raise AssertionError("eval_image_tokenizer summary is missing required fields")
        if "channel_collapse" not in candidate or "downstream_readiness" not in candidate:
            raise AssertionError("eval_image_tokenizer summary is missing audit fields")
        if "config_metadata" not in candidate:
            raise AssertionError("eval_image_tokenizer summary is missing config metadata")
        if not (outdir / "summary.csv").exists():
            raise AssertionError("eval_image_tokenizer summary.csv was not written")

        audit_outdir = root / "audit_out"
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "audit_image_tokenizers.py"),
                "--candidate",
                f"smoke|{config_path}|{ckpt_path}",
                "--outdir",
                str(audit_outdir),
                "--max-batches",
                "1",
                "--export-visuals",
                "--visual-samples",
                "1",
                "--crop-size",
                "16",
            ],
            check=True,
            cwd=REPO_ROOT,
        )
        audit_summary = json.loads((audit_outdir / "summary.json").read_text(encoding="utf-8"))
        if not audit_summary["ranking"]:
            raise AssertionError("audit_image_tokenizers did not produce any ranking rows")
        if not (audit_outdir / "summary.csv").exists():
            raise AssertionError("audit_image_tokenizers summary.csv was not written")

        run_dir = root / "tokenizer_run"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = ckpt_dir / "epoch=000001.ckpt"
        last_ckpt = ckpt_dir / "last.ckpt"
        callback_state = {
            "callbacks": {
                "ModelCheckpoint{monitor=val/total_loss}": {
                    "monitor": "val/total_loss",
                    "best_model_path": str(best_ckpt),
                }
            }
        }
        torch.save(callback_state, last_ckpt)
        torch.save({}, best_ckpt)
        resolved_path = subprocess.check_output(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "find_image_tokenizer_checkpoint.py"),
                "--config",
                str(config_path),
                "--run-dir",
                str(run_dir),
            ],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
        if Path(resolved_path).resolve() != best_ckpt.resolve():
            raise AssertionError("find_image_tokenizer_checkpoint did not resolve the monitored best checkpoint")


def main():
    torch.manual_seed(0)
    _instantiate_checked_in_configs()
    _run_forward_backward()
    _run_adversarial_config_smoke()
    _run_eval_script_smoke()
    print("Image autoencoder selfcheck passed")
    print("checked configs: base/f8 and LPIPS variants instantiate with expected latent shapes")
    print("forward/backward: ok")
    print("adversarial config smoke: ok")
    print("eval_image_tokenizer smoke: ok")
    print("audit_image_tokenizers smoke: ok")
    print("find_image_tokenizer_checkpoint smoke: ok")


if __name__ == "__main__":
    main()

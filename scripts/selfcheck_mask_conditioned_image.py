import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
LDM_ROOT = REPO_ROOT / "third_party" / "latent-diffusion"
TAMING_ROOT = LDM_ROOT / "taming-transformers"
for path in (REPO_ROOT, LDM_ROOT, TAMING_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ldm.util import instantiate_from_config
from latent_meanflow.conditioning import LatentConditioning
from scripts.train_mask_conditioned_image import DEFAULT_CONFIGS


CONFIGS = [
    REPO_ROOT / "configs" / "latent_fm_mask2image_unet.yaml",
    REPO_ROOT / "configs" / "latent_meanflow_mask2image_unet.yaml",
    REPO_ROOT / "configs" / "latent_alphaflow_mask2image_unet.yaml",
    REPO_ROOT / "configs" / "latent_alphaflow_mask2image_unet_tiny.yaml",
    REPO_ROOT / "configs" / "latent_alphaflow_mask2image_f8_unet.yaml",
]


def _latent_shape_from_tokenizer_config(tokenizer_config_path):
    tokenizer_config = OmegaConf.load(tokenizer_config_path)
    tokenizer = instantiate_from_config(tokenizer_config.model)
    return tokenizer.latent_spatial_shape


def main():
    for config_path in CONFIGS:
        config = OmegaConf.load(config_path)
        target = str(OmegaConf.select(config, "model.target"))
        if "mask_conditioned_image_trainer" not in target:
            raise AssertionError(f"{config_path.name} does not target the mask-conditioned trainers")

        tokenizer_config_path = REPO_ROOT / str(
            OmegaConf.select(config, "model.params.tokenizer_config_path")
        )
        if "autoencoder_image" not in tokenizer_config_path.name:
            raise AssertionError(f"{config_path.name} does not point to an image-only tokenizer")

        backbone_cfg = OmegaConf.to_container(config.model.params.backbone_config, resolve=True)
        backbone_cfg.setdefault("params", {})
        backbone_cfg["params"]["in_channels"] = 4
        spatial_condition_channels = int(backbone_cfg["params"].get("spatial_condition_channels", 0))
        if spatial_condition_channels <= 0:
            raise AssertionError(f"{config_path.name} is missing spatial_condition_channels")

        backbone = instantiate_from_config(backbone_cfg)
        latent_h, latent_w = _latent_shape_from_tokenizer_config(tokenizer_config_path)
        z_t = torch.randn(2, 4, latent_h, latent_w, requires_grad=True)
        mask = torch.randn(2, spatial_condition_channels, latent_h, latent_w)
        t = torch.rand(2)
        condition = LatentConditioning(spatial=mask)
        if str(OmegaConf.select(config, "model.params.objective_name")) == "fm":
            out = backbone(z_t, t=t, condition=condition)
        else:
            delta_t = torch.rand(2) * 0.25
            out = backbone(z_t, t=t, delta_t=delta_t, condition=condition)
        if out.shape != z_t.shape:
            raise AssertionError(
                f"{config_path.name} produced wrong shape {tuple(out.shape)} for latent {tuple(z_t.shape)}"
            )
        out.square().mean().backward()

    expected_defaults = {
        "fm": "latent_fm_mask2image_unet.yaml",
        "meanflow": "latent_meanflow_mask2image_unet.yaml",
        "alphaflow": "latent_alphaflow_mask2image_unet.yaml",
    }
    for objective, expected_name in expected_defaults.items():
        if Path(DEFAULT_CONFIGS[objective]).name != expected_name:
            raise AssertionError(f"Unexpected wrapper default for {objective}: {DEFAULT_CONFIGS[objective]}")

    print("Mask-conditioned image route self-check passed.")


if __name__ == "__main__":
    main()

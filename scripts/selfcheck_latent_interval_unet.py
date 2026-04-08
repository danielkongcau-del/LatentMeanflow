import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
LDM_ROOT = REPO_ROOT / "third_party" / "latent-diffusion"
for path in (REPO_ROOT, LDM_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from latent_meanflow.models.backbones.latent_interval_unet import LatentIntervalUNet


def main():
    cfg = OmegaConf.load(REPO_ROOT / "configs" / "latent_meanflow_semantic_256_unet.yaml")
    model = LatentIntervalUNet(in_channels=4, **cfg.model.params.backbone_config.params)

    batch_size = 2
    z_t = torch.randn(batch_size, 4, 64, 64)
    t = torch.rand(batch_size)
    r = torch.rand(batch_size) * 0.5
    out = model(z_t, t=t, r=r)

    assert out.shape == z_t.shape, f"Expected output shape {tuple(z_t.shape)}, got {tuple(out.shape)}"
    params = sum(parameter.numel() for parameter in model.parameters())
    print(f"LatentIntervalUNet output shape ok: {tuple(out.shape)}")
    print(f"LatentIntervalUNet parameters: {params}")


if __name__ == "__main__":
    main()

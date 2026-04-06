import os
import shutil
from pathlib import Path

import torch


# Source file you downloaded.
SRC_PATH = "/root/autodl-tmp/latent-diffusion/vgg16-397923af.pth"


def main():
    src = Path(SRC_PATH)
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    hub_dir = Path(torch.hub.get_dir())
    ckpt_dir = hub_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dst = ckpt_dir / src.name
    shutil.copy2(src, dst)
    print(f"Copied to: {dst}")


if __name__ == "__main__":
    main()

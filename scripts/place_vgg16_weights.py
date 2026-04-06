import argparse
import shutil
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Copy VGG16 LPIPS weights into the Torch hub checkpoint cache.")
    parser.add_argument("--src", type=Path, required=True, help="Path to vgg16-397923af.pth")
    args = parser.parse_args()

    if not args.src.exists():
        raise FileNotFoundError(f"Source file not found: {args.src}")

    hub_dir = Path(torch.hub.get_dir())
    ckpt_dir = hub_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dst = ckpt_dir / args.src.name
    shutil.copy2(args.src, dst)
    print(f"Copied to: {dst}")


if __name__ == "__main__":
    main()

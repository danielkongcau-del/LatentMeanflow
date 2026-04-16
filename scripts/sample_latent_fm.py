import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
LDM_ROOT = REPO_ROOT / "third_party" / "latent-diffusion"
TAMING_ROOT = LDM_ROOT / "taming-transformers"

for path in (REPO_ROOT, LDM_ROOT, TAMING_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ldm.util import instantiate_from_config
from latent_meanflow.utils import colorize_mask_index, overlay_color_mask_on_image


DEFAULT_CONFIG = REPO_ROOT / "configs" / "latent_fm_semantic_256.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Sample paired RGB image and semantic mask outputs from the latent FM prior.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=REPO_ROOT / "outputs" / "latent_fm_samples")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--class-label", type=int, default=None, help="Optional image-level condition.")
    parser.add_argument("--overlay-alpha", type=float, default=0.4)
    return parser.parse_args()


def find_latest_fm_ckpt():
    candidates = sorted(
        REPO_ROOT.glob("logs/**/checkpoints/last.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if "latent_fm" in str(candidate).lower():
            return candidate
    return None


def load_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = state["state_dict"] if "state_dict" in state else state
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def save_pair(rgb_tensor, mask_index_tensor, outdir, index, num_classes, overlay_alpha):
    image_dir = outdir / "image"
    raw_mask_dir = outdir / "mask_raw"
    overlay_dir = outdir / "overlay"
    color_mask_dir = outdir / "mask_color"
    image_dir.mkdir(parents=True, exist_ok=True)
    raw_mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    color_mask_dir.mkdir(parents=True, exist_ok=True)

    rgb = torch.clamp((rgb_tensor + 1.0) / 2.0, 0.0, 1.0)
    rgb_np = (rgb.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    mask_index_np = mask_index_tensor.cpu().numpy().astype(np.uint16)
    color_mask_np = colorize_mask_index(mask_index_np.astype(np.int64), num_classes=num_classes)
    overlay_np = overlay_color_mask_on_image(rgb_np, color_mask_np, alpha=overlay_alpha)

    stem = f"{index:06}"
    Image.fromarray(rgb_np).save(image_dir / f"{stem}.png")
    Image.fromarray(mask_index_np).save(raw_mask_dir / f"{stem}.png")
    Image.fromarray(color_mask_np).save(color_mask_dir / f"{stem}.png")
    Image.fromarray(overlay_np).save(overlay_dir / f"{stem}.png")


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    ckpt_path = args.ckpt or find_latest_fm_ckpt()
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError("Latent FM checkpoint not found. Pass --ckpt explicitly.")
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    model = load_model(args.config, ckpt_path, device)
    if args.class_label is not None and not getattr(model, "use_class_condition", False):
        raise ValueError(
            "--class-label was provided, but this latent FM checkpoint is configured as unconditional."
        )
    remaining = args.n_samples
    index = 0

    while remaining > 0:
        batch_size = min(args.batch_size, remaining)
        condition = None
        if args.class_label is not None:
            condition = torch.full((batch_size,), args.class_label, device=device, dtype=torch.long)
        latents = model.sample_latents(batch_size=batch_size, num_steps=args.steps, device=device, condition=condition)
        decoded = model.decode_latents(latents)

        for sample_idx in range(batch_size):
            save_pair(
                decoded["rgb_recon"][sample_idx],
                decoded["mask_index"][sample_idx],
                outdir=args.outdir,
                index=index,
                num_classes=model.num_classes,
                overlay_alpha=args.overlay_alpha,
            )
            index += 1
        remaining -= batch_size


if __name__ == "__main__":
    main()

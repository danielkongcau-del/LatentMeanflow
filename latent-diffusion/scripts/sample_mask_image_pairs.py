import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TAMING_DIR = ROOT / "taming-transformers"
if TAMING_DIR.exists():
    sys.path.insert(0, str(TAMING_DIR))

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


def load_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    model.eval()
    return model


def save_pair(sample, out_image_dir, out_mask_dir, index, mask_threshold):
    rgb = sample[:3]
    mask = sample[3:4]

    rgb = torch.clamp((rgb + 1.0) / 2.0, 0.0, 1.0)
    mask = torch.clamp((mask + 1.0) / 2.0, 0.0, 1.0)
    mask = (mask >= mask_threshold).float()

    rgb_np = (rgb * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    mask_np = (mask * 255.0).byte().squeeze(0).cpu().numpy()

    Image.fromarray(rgb_np).save(os.path.join(out_image_dir, f"{index:06}.png"))
    Image.fromarray(mask_np, mode="L").save(os.path.join(out_mask_dir, f"{index:06}.png"))


def sample_latent(model, batch_size, steps, eta, conditioning=None):
    sampler = DDIMSampler(model)
    shape = (model.channels, model.image_size, model.image_size)
    samples, _ = sampler.sample(
        S=steps,
        batch_size=batch_size,
        shape=shape,
        conditioning=conditioning,
        eta=eta,
        verbose=False,
    )
    return model.decode_first_stage(samples)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mask_image/ldm_4ch_256.yaml", help="Path to training config yaml")
    parser.add_argument("--ckpt", type=str, default="logs/ldm/last.ckpt", help="Path to model checkpoint")
    parser.add_argument("--outdir", type=str, default="outputs/worm")
    parser.add_argument("--n_samples", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--class_label", type=int, default=3, help="fixed class id; -1 for random")
    parser.add_argument("--n_classes", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(args.seed)

    out_image_dir = os.path.join(args.outdir, "image")
    out_mask_dir = os.path.join(args.outdir, "mask")
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    model = load_model(args.config, args.ckpt, device)

    remaining = args.n_samples
    index = 0
    is_latent = hasattr(model, "decode_first_stage") and hasattr(model, "first_stage_model")
    is_adm = getattr(model.model, "conditioning_key", None) == "adm"

    while remaining > 0:
        bs = min(args.batch_size, remaining)
        if is_latent:
            conditioning = None
            if is_adm:
                if args.class_label >= 0:
                    labels = torch.full((bs,), args.class_label, device=device, dtype=torch.long)
                else:
                    labels = torch.randint(0, args.n_classes, (bs,), device=device)
                conditioning = labels
            samples = sample_latent(model, bs, args.ddim_steps, args.ddim_eta, conditioning=conditioning)
        else:
            samples = model.sample(batch_size=bs)
        for i in range(bs):
            save_pair(samples[i], out_image_dir, out_mask_dir, index, args.mask_threshold)
            index += 1
        remaining -= bs


if __name__ == "__main__":
    main()

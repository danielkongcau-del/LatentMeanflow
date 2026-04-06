import argparse
import sys
from pathlib import Path

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

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


def load_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    state = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
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

    Image.fromarray(rgb_np).save(out_image_dir / f"{index:06}.png")
    Image.fromarray(mask_np, mode="L").save(out_mask_dir / f"{index:06}.png")


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
    parser = argparse.ArgumentParser(description="Sample paired RGB image and mask outputs.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "ldm_4ch_256.yaml")
    parser.add_argument("--ckpt", type=Path, default=REPO_ROOT / "logs" / "ldm" / "checkpoints" / "last.ckpt")
    parser.add_argument("--outdir", type=Path, default=REPO_ROOT / "outputs" / "samples")
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--ddim-steps", type=int, default=200)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--class-label", type=int, default=3, help="Use -1 for random labels.")
    parser.add_argument("--n-classes", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    out_image_dir = args.outdir / "image"
    out_mask_dir = args.outdir / "mask"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.config, args.ckpt, device)

    remaining = args.n_samples
    index = 0
    is_latent = hasattr(model, "decode_first_stage") and hasattr(model, "first_stage_model")
    is_adm = getattr(model.model, "conditioning_key", None) == "adm"

    while remaining > 0:
        batch_size = min(args.batch_size, remaining)
        if is_latent:
            conditioning = None
            if is_adm:
                if args.class_label >= 0:
                    labels = torch.full((batch_size,), args.class_label, device=device, dtype=torch.long)
                else:
                    labels = torch.randint(0, args.n_classes, (batch_size,), device=device)
                conditioning = labels
            samples = sample_latent(
                model,
                batch_size,
                args.ddim_steps,
                args.ddim_eta,
                conditioning=conditioning,
            )
        else:
            samples = model.sample(batch_size=batch_size)

        for i in range(batch_size):
            save_pair(samples[i], out_image_dir, out_mask_dir, index, args.mask_threshold)
            index += 1
        remaining -= batch_size


if __name__ == "__main__":
    main()

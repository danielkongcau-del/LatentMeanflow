import argparse
import csv
import json
import sys
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

from latent_meanflow.utils import colorize_mask_index
from scripts.sample_latent_flow import (
    find_latest_flow_ckpt,
    load_config,
    load_model,
    validate_ckpt_matches_config,
)


DEFAULT_CONFIG = REPO_ROOT / "configs" / "latent_diffusion_mask_prior_sit.yaml"
DEFAULT_NFE_VALUES = [8, 4, 2, 1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sample unconditional semantic masks from the project-layer SiT-style diffusion baseline. "
            "Outputs raw class-id masks, colorized masks, and simple diagnostic panels."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nfe-values", type=int, nargs="+", default=DEFAULT_NFE_VALUES)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _prepare_outdir(path, overwrite):
    if path.exists():
        existing = list(path.rglob("*"))
        if existing and not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {path}. "
                "Use a fresh outdir or pass --overwrite."
            )
    path.mkdir(parents=True, exist_ok=True)


def _compute_boundary_map(mask_index, ignore_index=None):
    mask_index = np.asarray(mask_index, dtype=np.int64)
    boundary = np.zeros(mask_index.shape, dtype=np.uint8)
    valid_mask = np.ones(mask_index.shape, dtype=bool)
    if ignore_index is not None:
        valid_mask &= mask_index != int(ignore_index)

    vertical = (mask_index[1:, :] != mask_index[:-1, :]) & valid_mask[1:, :] & valid_mask[:-1, :]
    horizontal = (mask_index[:, 1:] != mask_index[:, :-1]) & valid_mask[:, 1:] & valid_mask[:, :-1]

    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    return (boundary * 255).astype(np.uint8)


def _make_raw_render(mask_index, num_classes, ignore_index=None):
    mask_index = np.asarray(mask_index, dtype=np.int64)
    raw = np.zeros(mask_index.shape, dtype=np.uint8)
    valid_mask = np.ones(mask_index.shape, dtype=bool)
    if ignore_index is not None:
        valid_mask &= mask_index != int(ignore_index)
    if np.any(valid_mask):
        denom = max(int(num_classes) - 1, 1)
        scaled = np.round(mask_index[valid_mask].astype(np.float32) * (255.0 / float(denom)))
        raw[valid_mask] = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
    return np.repeat(raw[:, :, None], 3, axis=2)


def _make_panel(mask_index, num_classes, ignore_index=None):
    raw_render = _make_raw_render(mask_index, num_classes=num_classes, ignore_index=ignore_index)
    color_render = colorize_mask_index(
        mask_index,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    boundary = _compute_boundary_map(mask_index, ignore_index=ignore_index)
    boundary_rgb = np.repeat(boundary[:, :, None], 3, axis=2)
    return raw_render, color_render, boundary_rgb, np.concatenate([raw_render, color_render, boundary_rgb], axis=1)


def _save_mask(mask_index, *, outdir, index, num_classes, ignore_index=None):
    raw_dir = outdir / "mask_raw"
    color_dir = outdir / "mask_color"
    boundary_dir = outdir / "boundary"
    panel_dir = outdir / "panel"
    for directory in (raw_dir, color_dir, boundary_dir, panel_dir):
        directory.mkdir(parents=True, exist_ok=True)

    mask_index = np.asarray(mask_index, dtype=np.int64)
    raw_mask = mask_index.copy()
    raw_mask[raw_mask < 0] = 65535
    raw_mask = raw_mask.astype(np.uint16)

    _, color_render, boundary_rgb, panel = _make_panel(
        mask_index,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    stem = f"{int(index):06}"
    Image.fromarray(raw_mask).save(raw_dir / f"{stem}.png")
    Image.fromarray(color_render).save(color_dir / f"{stem}.png")
    Image.fromarray(boundary_rgb).save(boundary_dir / f"{stem}.png")
    Image.fromarray(panel).save(panel_dir / f"{stem}.png")


@torch.no_grad()
def generate_mask_prior_sweep(*, model, outdir, nfe_values, seed, n_samples, batch_size):
    device = model.device
    latent_shape = (model.latent_channels, *model.latent_spatial_shape)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise_bank = torch.randn((int(n_samples), *latent_shape), generator=generator)

    summary_rows = []
    for nfe in nfe_values:
        nfe = int(nfe)
        nfe_dir = outdir / f"nfe{nfe}"
        _prepare_outdir(nfe_dir, overwrite=True)

        for start in range(0, int(n_samples), int(batch_size)):
            current_batch = min(int(batch_size), int(n_samples) - start)
            noise = noise_bank[start : start + current_batch].to(device)
            sampled_latents = model.sample_latents(
                batch_size=current_batch,
                nfe=nfe,
                device=device,
                noise=noise,
            )
            decoded = model.decode_latents(sampled_latents)
            for local_idx in range(current_batch):
                _save_mask(
                    decoded["mask_index"][local_idx].detach().cpu().numpy(),
                    outdir=nfe_dir,
                    index=start + local_idx,
                    num_classes=model.num_classes,
                )

        summary_rows.append(
            {
                "nfe": nfe,
                "outdir": str(nfe_dir),
                "mask_raw_count": len(list((nfe_dir / "mask_raw").glob("*.png"))),
                "mask_color_count": len(list((nfe_dir / "mask_color").glob("*.png"))),
                "boundary_count": len(list((nfe_dir / "boundary").glob("*.png"))),
                "panel_count": len(list((nfe_dir / "panel").glob("*.png"))),
            }
        )
    return summary_rows


@torch.no_grad()
def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))

    ckpt_path = args.ckpt or find_latest_flow_ckpt(args.config, config)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError("Mask-prior checkpoint not found. Pass --ckpt explicitly.")
    validate_ckpt_matches_config(args.config, ckpt_path)

    outdir = args.outdir.resolve()
    _prepare_outdir(outdir, overwrite=args.overwrite)

    model = load_model(config, ckpt_path, device=device)
    summary_rows = generate_mask_prior_sweep(
        model=model,
        outdir=outdir,
        nfe_values=args.nfe_values,
        seed=args.seed,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
    )

    summary = {
        "task": "p(semantic_mask)",
        "config": str(args.config.resolve()),
        "checkpoint": str(ckpt_path.resolve()),
        "seed": int(args.seed),
        "n_samples": int(args.n_samples),
        "batch_size": int(args.batch_size),
        "nfe_values": [int(value) for value in args.nfe_values],
        "monitor": getattr(model, "monitor", None),
        "num_classes": int(model.num_classes),
        "mask_spatial_shape": [int(model.latent_spatial_shape[0]), int(model.latent_spatial_shape[1])],
        "results": summary_rows,
    }

    summary_json_path = outdir / "summary.json"
    summary_csv_path = outdir / "summary.csv"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "nfe",
                "outdir",
                "mask_raw_count",
                "mask_color_count",
                "boundary_count",
                "panel_count",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved mask-prior sweep to {outdir}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()

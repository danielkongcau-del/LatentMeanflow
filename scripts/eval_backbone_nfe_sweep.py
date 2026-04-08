import argparse
import csv
import json
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

from scripts.sample_latent_flow import load_config, load_model, save_pair, validate_ckpt_matches_config


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one latent-flow checkpoint under a fixed NFE sweep. "
            "This is intended for apples-to-apples backbone benchmarking, not for ad-hoc one-step previews."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--nfe-values", type=int, nargs="+", default=[8, 4, 2, 1])
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--class-label", type=int, default=None)
    parser.add_argument("--overlay-alpha", type=float, default=0.4)
    parser.add_argument("--two-step-time", type=float, default=None)
    parser.add_argument(
        "--expected-monitor",
        type=str,
        default="val/base_error_mean",
        help="Benchmark safety check. The config monitor should match this value.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing outdir. By default the script refuses to mix results.",
    )
    return parser.parse_args()


def _prepare_outdir(path, overwrite):
    if path.exists():
        existing_files = list(path.rglob("*"))
        if existing_files and not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {path}. "
                "Use a fresh outdir or pass --overwrite explicitly."
            )
    path.mkdir(parents=True, exist_ok=True)


def _check_monitor(config, expected_monitor):
    configured_monitor = OmegaConf.select(config, "model.params.monitor")
    if expected_monitor is None:
        return configured_monitor
    if configured_monitor != expected_monitor:
        raise ValueError(
            f"Benchmark monitor mismatch: expected '{expected_monitor}', got '{configured_monitor}'. "
            "This benchmark route is intended to compare checkpoints selected by the same validation signal."
        )
    return configured_monitor


def _make_fixed_noise_bank(n_samples, latent_shape, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn((n_samples, *latent_shape), generator=generator)


def _make_condition(batch_size, class_label, device):
    if class_label is None:
        return None
    return torch.full((batch_size,), int(class_label), device=device, dtype=torch.long)


def _count_pngs(path):
    return len(list(path.glob("*.png")))


@torch.no_grad()
def main():
    args = parse_args()
    config = load_config(args.config)
    monitor = _check_monitor(config, expected_monitor=args.expected_monitor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))

    ckpt_path = args.ckpt.resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    validate_ckpt_matches_config(args.config, ckpt_path)

    outdir = args.outdir.resolve()
    _prepare_outdir(outdir, overwrite=args.overwrite)

    model = load_model(config, ckpt_path, device=device)
    if args.class_label is not None and not getattr(model, "use_class_condition", False):
        raise ValueError("--class-label was provided, but this checkpoint is configured as unconditional.")
    if args.two_step_time is not None and hasattr(model, "sampler") and hasattr(model.sampler, "two_step_time"):
        model.sampler.two_step_time = float(args.two_step_time)

    latent_shape = (model.latent_channels, *model.latent_spatial_shape)
    noise_bank = _make_fixed_noise_bank(args.n_samples, latent_shape=latent_shape, seed=args.seed)

    summary = {
        "config": str(args.config.resolve()),
        "checkpoint": str(ckpt_path),
        "run_dir": str(ckpt_path.parent.parent.resolve()),
        "objective_name": OmegaConf.select(config, "model.params.objective_name"),
        "monitor": monitor,
        "seed": int(args.seed),
        "n_samples": int(args.n_samples),
        "batch_size": int(args.batch_size),
        "nfe_values": [int(value) for value in args.nfe_values],
        "noise_protocol": "shared fixed latent noise bank reused across all NFE values",
        "results": [],
    }

    for nfe in args.nfe_values:
        nfe = int(nfe)
        nfe_dir = outdir / f"nfe{nfe}"
        _prepare_outdir(nfe_dir, overwrite=args.overwrite)

        remaining = args.n_samples
        index = 0
        while remaining > 0:
            batch_size = min(args.batch_size, remaining)
            batch_noise = noise_bank[index : index + batch_size].to(device)
            condition = _make_condition(batch_size, args.class_label, device=device)
            latents = model.sample_latents(
                batch_size=batch_size,
                nfe=nfe,
                device=device,
                condition=condition,
                noise=batch_noise,
            )
            decoded = model.decode_latents(latents)

            for sample_idx in range(batch_size):
                save_pair(
                    decoded["rgb_recon"][sample_idx],
                    decoded["mask_index"][sample_idx],
                    outdir=nfe_dir,
                    index=index + sample_idx,
                    num_classes=model.num_classes,
                    overlay_alpha=args.overlay_alpha,
                )
            index += batch_size
            remaining -= batch_size

        result = {
            "nfe": nfe,
            "outdir": str(nfe_dir),
            "image_count": _count_pngs(nfe_dir / "image"),
            "mask_raw_count": _count_pngs(nfe_dir / "mask_raw"),
            "mask_color_count": _count_pngs(nfe_dir / "mask_color"),
            "overlay_count": _count_pngs(nfe_dir / "overlay"),
        }
        summary["results"].append(result)

    summary_json_path = outdir / "summary.json"
    summary_csv_path = outdir / "summary.csv"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["nfe", "outdir", "image_count", "mask_raw_count", "mask_color_count", "overlay_count"],
        )
        writer.writeheader()
        writer.writerows(summary["results"])

    print(f"Saved benchmark sweep to {outdir}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()

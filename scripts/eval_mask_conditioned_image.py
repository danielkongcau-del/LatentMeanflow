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

from scripts.sample_mask_conditioned_image import (
    DEFAULT_CONFIG,
    DEFAULT_NFE_VALUES,
    generate_mask_conditioned_sweep,
    load_examples,
    load_config,
    load_model,
    validate_ckpt_matches_config,
    find_latest_flow_ckpt,
    _prepare_outdir,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the project-layer mask-conditioned image route. "
            "This script fixes the checkpoint, runs or reuses an NFE sweep, and "
            "reports GT sanity metrics for p(image | semantic mask)."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, default=None)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument(
        "--label-spec",
        type=Path,
        default=REPO_ROOT / "configs" / "label_specs" / "remote_semantic.yaml",
    )
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--nfe-values", type=int, nargs="+", default=DEFAULT_NFE_VALUES)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--overlay-alpha", type=float, default=0.4)
    parser.add_argument("--two-step-time", type=float, default=None)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_rgb_png(path):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def _list_pngs(path):
    return sorted(path.glob("*.png"))


def _make_lpips():
    from taming.modules.losses.lpips import LPIPS

    model = LPIPS().eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _compute_rgb_metrics(generated_dir, ground_truth_dir, *, device, compute_lpips):
    generated_paths = _list_pngs(generated_dir)
    ground_truth_paths = _list_pngs(ground_truth_dir)
    if not generated_paths or not ground_truth_paths:
        return None
    paired = list(zip(generated_paths, ground_truth_paths))
    if not paired:
        return None

    lpips_model = _make_lpips().to(device) if compute_lpips else None
    l1_values = []
    lpips_values = []
    for generated_path, ground_truth_path in paired:
        generated = _load_rgb_png(generated_path)
        ground_truth = _load_rgb_png(ground_truth_path)
        l1_values.append(float(np.abs(generated - ground_truth).mean()))
        if lpips_model is not None:
            generated_tensor = torch.from_numpy(generated).permute(2, 0, 1).unsqueeze(0).to(device=device)
            ground_truth_tensor = torch.from_numpy(ground_truth).permute(2, 0, 1).unsqueeze(0).to(device=device)
            generated_tensor = generated_tensor * 2.0 - 1.0
            ground_truth_tensor = ground_truth_tensor * 2.0 - 1.0
            lpips_value = lpips_model(generated_tensor.contiguous(), ground_truth_tensor.contiguous()).mean()
            lpips_values.append(float(lpips_value.item()))

    metrics = {
        "l1_mean": float(np.mean(l1_values)),
        "l1_std": float(np.std(l1_values)),
        "lpips_mean": None if not lpips_values else float(np.mean(lpips_values)),
        "lpips_std": None if not lpips_values else float(np.std(lpips_values)),
        "sample_count": len(paired),
        "fid": None,
        "kid": None,
    }
    return metrics


def _resolve_nfe_dirs(generated_root, requested_nfe_values):
    resolved = []
    for nfe in requested_nfe_values:
        path = generated_root / f"nfe{int(nfe)}"
        if path.exists():
            resolved.append((int(nfe), path))
    if resolved:
        return resolved
    auto_dirs = []
    for path in sorted(generated_root.glob("nfe*")):
        try:
            nfe = int(path.name.replace("nfe", ""))
        except ValueError:
            continue
        auto_dirs.append((nfe, path))
    return auto_dirs


@torch.no_grad()
def main():
    args = parse_args()
    outdir = args.outdir.resolve()
    _prepare_outdir(outdir, overwrite=args.overwrite)

    config = load_config(args.config)
    generated_root = None if args.generated_root is None else args.generated_root.resolve()
    source_mode = None

    if generated_root is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(int(args.seed))
        ckpt_path = args.ckpt or find_latest_flow_ckpt(args.config, config)
        if ckpt_path is None or not ckpt_path.exists():
            raise FileNotFoundError("Mask-conditioned checkpoint not found. Pass --ckpt explicitly.")
        validate_ckpt_matches_config(args.config, ckpt_path)
        model = load_model(config, ckpt_path, device=device)
        if not hasattr(model, "build_condition_from_mask_onehot"):
            raise TypeError(f"Config '{args.config.name}' is not a mask-conditioned image route.")
        if args.two_step_time is not None and hasattr(model, "sampler") and hasattr(model.sampler, "two_step_time"):
            model.sampler.two_step_time = float(args.two_step_time)

        examples, source_mode = load_examples(
            config,
            split=args.split,
            mask_dir=args.mask_dir,
            image_dir=args.image_dir,
            label_spec=args.label_spec,
            n_samples=args.n_samples,
        )
        generated_root = outdir / "generated"
        _prepare_outdir(generated_root, overwrite=args.overwrite)
        generate_mask_conditioned_sweep(
            model=model,
            examples=examples,
            outdir=generated_root,
            nfe_values=args.nfe_values,
            seed=args.seed,
            batch_size=args.batch_size,
            overlay_alpha=args.overlay_alpha,
        )
        backbone = getattr(model, "backbone", None)
    else:
        backbone = None

    nfe_dirs = _resolve_nfe_dirs(generated_root, args.nfe_values)
    if not nfe_dirs:
        raise FileNotFoundError(f"No nfe* directories found under {generated_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for nfe, nfe_dir in nfe_dirs:
        generated_dir = nfe_dir / "generated_image"
        ground_truth_dir = nfe_dir / "ground_truth_image"
        metrics = _compute_rgb_metrics(
            generated_dir,
            ground_truth_dir,
            device=device,
            compute_lpips=not args.skip_lpips,
        )
        result = {
            "nfe": int(nfe),
            "outdir": str(nfe_dir),
            "generated_image_count": len(_list_pngs(generated_dir)),
            "ground_truth_image_count": len(_list_pngs(ground_truth_dir)),
            "input_mask_raw_count": len(_list_pngs(nfe_dir / "input_mask_raw")),
            "input_mask_color_count": len(_list_pngs(nfe_dir / "input_mask_color")),
            "overlay_count": len(_list_pngs(nfe_dir / "overlay")),
            "panel_count": len(_list_pngs(nfe_dir / "panel")),
            "l1_mean": None if metrics is None else metrics["l1_mean"],
            "l1_std": None if metrics is None else metrics["l1_std"],
            "lpips_mean": None if metrics is None else metrics["lpips_mean"],
            "lpips_std": None if metrics is None else metrics["lpips_std"],
            "fid": None if metrics is None else metrics["fid"],
            "kid": None if metrics is None else metrics["kid"],
            "metrics_note": (
                "GT metrics are sanity-only because p(image | semantic_mask) is one-to-many."
            ),
        }
        results.append(result)

    summary = {
        "config": str(args.config.resolve()),
        "checkpoint": None if args.ckpt is None else str(args.ckpt.resolve()),
        "generated_root": str(generated_root),
        "source_mode": source_mode,
        "task": "p(image | semantic_mask)",
        "condition_mode": None if backbone is None else getattr(backbone, "condition_mode", None),
        "condition_source": None if backbone is None else getattr(backbone, "condition_source", None),
        "use_boundary_condition": None if backbone is None else bool(getattr(backbone, "use_boundary_condition", False)),
        "boundary_mode": None if backbone is None else getattr(backbone, "boundary_mode", None),
        "use_semantic_condition_encoder": None
        if backbone is None
        else bool(getattr(backbone, "use_semantic_condition_encoder", False)),
        "nfe_values": [int(value) for value, _ in nfe_dirs],
        "primary_readout": [
            "layout faithfulness to input mask",
            "texture plausibility",
            "few-step stability at NFE=8/4/2/1",
        ],
        "non_primary_metrics": [
            "L1 to GT (sanity only)",
            "LPIPS to GT (sanity only)",
        ],
        "results": results,
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
                "generated_image_count",
                "ground_truth_image_count",
                "input_mask_raw_count",
                "input_mask_color_count",
                "overlay_count",
                "panel_count",
                "l1_mean",
                "l1_std",
                "lpips_mean",
                "lpips_std",
                "fid",
                "kid",
                "metrics_note",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved mask-conditioned evaluation summary to {outdir}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()

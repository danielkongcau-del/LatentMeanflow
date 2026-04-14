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

from ldm.util import instantiate_from_config
from latent_meanflow.utils.palette import (
    UNDEFINED_CLASS_ID,
    build_lookup_table,
    infer_num_classes,
    resolve_gray_to_class_id,
)
from scripts.sample_latent_flow import (
    find_latest_flow_ckpt,
    load_config,
    load_model,
    validate_ckpt_matches_config,
)
from scripts.sample_mask_prior import (
    DEFAULT_CONFIG,
    DEFAULT_NFE_VALUES,
    _prepare_outdir,
    _resolve_nfe_dirs,
    generate_mask_prior_sweep,
)


MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the project-layer unconditional semantic-mask route p(semantic_mask). "
            "This baseline compares generated masks against real-mask distribution statistics "
            "and nearest-real layout agreement; it does not assume paired targets."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--generated-root", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument(
        "--label-spec",
        type=Path,
        default=REPO_ROOT / "configs" / "label_specs" / "remote_semantic.yaml",
    )
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nfe-values", type=int, nargs="+", default=DEFAULT_NFE_VALUES)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--two-step-time", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-monitor", type=str, default="val/base_error_mean")
    return parser.parse_args()


def _check_monitor(config, expected_monitor):
    configured_monitor = config.model.params.get("monitor")
    if expected_monitor is not None and configured_monitor != expected_monitor:
        raise ValueError(
            f"Evaluation monitor mismatch: expected '{expected_monitor}', got '{configured_monitor}'. "
            "Use the best checkpoint selected by val/base_error_mean for the main baseline."
        )
    return configured_monitor


def _load_mask_png(path):
    mask = np.asarray(Image.open(path), dtype=np.int64)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask png, got shape {mask.shape} from {path}")
    return mask


def _resolve_split_key(config, split_name):
    normalized = str(split_name).lower()
    if normalized == "val":
        normalized = "validation"
    dataset_cfg = config.data.params.get(normalized)
    if dataset_cfg is None:
        raise KeyError(f"Config does not define data.params.{normalized}")
    return normalized


def _load_reference_masks_from_dataset(config, split, n_samples):
    dataset_cfg = config.data.params[split]
    dataset = instantiate_from_config(dataset_cfg)
    limit = min(len(dataset), int(n_samples))
    masks = []
    for index in range(limit):
        sample = dataset[index]
        masks.append(np.asarray(sample["mask_index"], dtype=np.int64))
    return masks, split


def _load_reference_masks_from_dir(mask_dir, *, label_spec, size, n_samples):
    gray_to_class_id, ignore_index = resolve_gray_to_class_id(label_spec, ignore_index=None)
    num_classes = infer_num_classes(gray_to_class_id, ignore_index=ignore_index)
    lookup = build_lookup_table(gray_to_class_id, undefined_value=UNDEFINED_CLASS_ID)

    mask_paths = []
    for ext in MASK_EXTS:
        mask_paths.extend(Path(mask_dir).glob(f"*{ext}"))
    mask_paths = sorted(mask_paths)[: int(n_samples)]
    if not mask_paths:
        raise FileNotFoundError(f"No mask files found under {mask_dir}")

    masks = []
    for mask_path in mask_paths:
        mask_image = Image.open(mask_path)
        if mask_image.mode != "L":
            mask_image = mask_image.convert("L")
        if size is not None:
            mask_image = mask_image.resize((int(size[1]), int(size[0])), resample=Image.NEAREST)
        mask_raw = np.asarray(mask_image, dtype=np.uint8)
        mask_index = lookup[mask_raw].astype(np.int64, copy=False)
        undefined_mask = mask_index == UNDEFINED_CLASS_ID
        if np.any(undefined_mask):
            unknown_values = sorted(int(value) for value in np.unique(mask_raw[undefined_mask]).tolist())
            raise ValueError(
                f"Mask {mask_path} contains gray values missing from {label_spec}: {unknown_values}"
            )
        masks.append(mask_index)
    return masks, "mask_dir", int(num_classes), ignore_index


def _resolve_label_spec_metadata(label_spec):
    gray_to_class_id, ignore_index = resolve_gray_to_class_id(label_spec, ignore_index=None)
    num_classes = infer_num_classes(gray_to_class_id, ignore_index=ignore_index)
    return int(num_classes), ignore_index


def _boundary_density(mask, ignore_index=None):
    valid_mask = np.ones(mask.shape, dtype=bool)
    if ignore_index is not None:
        valid_mask &= mask != int(ignore_index)
    vertical = (mask[1:, :] != mask[:-1, :]) & valid_mask[1:, :] & valid_mask[:-1, :]
    horizontal = (mask[:, 1:] != mask[:, :-1]) & valid_mask[:, 1:] & valid_mask[:, :-1]
    boundary_pixels = float(vertical.sum() + horizontal.sum())
    valid_pixels = float(max(1, valid_mask.sum()))
    return boundary_pixels / valid_pixels


def _class_ratio(mask, num_classes, ignore_index=None):
    valid_mask = np.ones(mask.shape, dtype=bool)
    if ignore_index is not None:
        valid_mask &= mask != int(ignore_index)
    valid_values = mask[valid_mask]
    if valid_values.size == 0:
        return np.zeros((num_classes,), dtype=np.float64)
    counts = np.bincount(valid_values.astype(np.int64), minlength=num_classes).astype(np.float64)
    counts /= float(valid_values.size)
    return counts


def _class_presence(mask, num_classes, ignore_index=None):
    ratio = _class_ratio(mask, num_classes=num_classes, ignore_index=ignore_index)
    return (ratio > 0.0).astype(np.float64)


def _mask_miou(mask_a, mask_b, num_classes, ignore_index=None):
    valid_mask = np.ones(mask_a.shape, dtype=bool)
    if ignore_index is not None:
        valid_mask &= mask_a != int(ignore_index)
        valid_mask &= mask_b != int(ignore_index)
    if not np.any(valid_mask):
        return 0.0

    values = []
    for class_id in range(int(num_classes)):
        pred = (mask_a == class_id) & valid_mask
        target = (mask_b == class_id) & valid_mask
        union = np.logical_or(pred, target).sum()
        if union <= 0:
            continue
        intersection = np.logical_and(pred, target).sum()
        values.append(float(intersection) / float(union))
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _load_generated_masks(nfe_dir, n_samples):
    mask_dir = nfe_dir / "mask_raw"
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask_raw directory under {nfe_dir}")
    mask_paths = sorted(mask_dir.glob("*.png"))[: int(n_samples)]
    if not mask_paths:
        raise RuntimeError(f"No generated masks found under {mask_dir}")
    masks = [_load_mask_png(path) for path in mask_paths]
    stems = [path.stem for path in mask_paths]
    return masks, stems


def _summarize_distribution(masks, *, num_classes, ignore_index=None):
    ratios = np.stack(
        [_class_ratio(mask, num_classes=num_classes, ignore_index=ignore_index) for mask in masks],
        axis=0,
    )
    presences = np.stack(
        [_class_presence(mask, num_classes=num_classes, ignore_index=ignore_index) for mask in masks],
        axis=0,
    )
    boundary = np.asarray([_boundary_density(mask, ignore_index=ignore_index) for mask in masks], dtype=np.float64)
    unique_class_count = np.asarray(
        [int(presence.sum()) for presence in presences],
        dtype=np.float64,
    )
    return {
        "class_ratio_mean": ratios.mean(axis=0),
        "class_presence_mean": presences.mean(axis=0),
        "boundary_density_mean": float(boundary.mean()),
        "unique_class_count_mean": float(unique_class_count.mean()),
    }


def _nearest_real_mious(fake_masks, real_masks, *, num_classes, ignore_index=None):
    best_scores = []
    for fake_mask in fake_masks:
        scores = [
            _mask_miou(fake_mask, real_mask, num_classes=num_classes, ignore_index=ignore_index)
            for real_mask in real_masks
        ]
        best_scores.append(max(scores) if scores else 0.0)
    return np.asarray(best_scores, dtype=np.float64)


def _pairwise_fake_mious(fake_masks, *, num_classes, ignore_index=None):
    scores = []
    for left in range(len(fake_masks)):
        for right in range(left + 1, len(fake_masks)):
            scores.append(
                _mask_miou(
                    fake_masks[left],
                    fake_masks[right],
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                )
            )
    return np.asarray(scores, dtype=np.float64)


def _write_markdown_report(path, summary):
    lines = [
        "# Mask Prior Report",
        "",
        f"- task: `{summary['task']}`",
        f"- config: `{summary['config']}`",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- monitor: `{summary['monitor']}`",
        f"- reference source: `{summary['reference_source']}`",
        "",
        "Primary readout: nearest-real mIoU plus collapse sanity. This is an unconditional `p(mask)` route, so there is no paired target mask.",
        "",
        "| NFE | nearest-real mIoU | pairwise fake mIoU | class-ratio L1 | boundary gap | unique-class gap |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in summary["results"]:
        pairwise = result["pairwise_fake_miou_mean"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(result["nfe"])),
                    f"{float(result['nearest_real_miou_mean']):.4f}",
                    "n/a" if pairwise is None else f"{float(pairwise):.4f}",
                    f"{float(result['class_ratio_l1']):.4f}",
                    f"{float(result['boundary_density_gap']):.4f}",
                    f"{float(result['unique_class_count_gap']):.4f}",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Higher nearest-real mIoU is better. Higher pairwise fake mIoU can indicate mode collapse.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main():
    args = parse_args()
    config = load_config(args.config)
    monitor = _check_monitor(config, expected_monitor=args.expected_monitor)
    num_classes, ignore_index = _resolve_label_spec_metadata(args.label_spec)
    outdir = args.outdir.resolve()
    _prepare_outdir(outdir, overwrite=args.overwrite)

    generated_root = None if args.generated_root is None else args.generated_root.resolve()
    ckpt_path = None if args.ckpt is None else args.ckpt.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))

    if generated_root is None:
        if ckpt_path is None:
            ckpt_path = find_latest_flow_ckpt(args.config, config)
        if ckpt_path is None or not ckpt_path.exists():
            raise FileNotFoundError("Mask-prior checkpoint not found. Pass --ckpt explicitly.")
        validate_ckpt_matches_config(args.config, ckpt_path)
        model = load_model(config, ckpt_path, device=device)
        if args.two_step_time is not None and hasattr(model, "sampler") and hasattr(model.sampler, "two_step_time"):
            model.sampler.two_step_time = float(args.two_step_time)
        generated_root = outdir / "generated"
        _prepare_outdir(generated_root, overwrite=args.overwrite)
        generate_mask_prior_sweep(
            model=model,
            outdir=generated_root,
            nfe_values=args.nfe_values,
            seed=args.seed,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
        )

    if args.mask_dir is None:
        split_key = _resolve_split_key(config, args.split)
        reference_masks, reference_source = _load_reference_masks_from_dataset(
            config,
            split=split_key,
            n_samples=args.n_samples,
        )
    else:
        size = tuple(int(value) for value in config.model.params.mask_spatial_shape)
        reference_masks, reference_source, resolved_num_classes, resolved_ignore_index = _load_reference_masks_from_dir(
            args.mask_dir.resolve(),
            label_spec=args.label_spec,
            size=size,
            n_samples=args.n_samples,
        )
        if int(resolved_num_classes) != int(num_classes):
            raise ValueError(
                f"Label-spec class count mismatch: expected {num_classes}, got {resolved_num_classes}"
            )
        ignore_index = resolved_ignore_index

    nfe_dirs = _resolve_nfe_dirs(generated_root, args.nfe_values)
    if not nfe_dirs:
        raise FileNotFoundError(f"No nfe* directories found under {generated_root}")

    reference_stats = _summarize_distribution(
        reference_masks,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    results = []
    for nfe, nfe_dir in nfe_dirs:
        generated_masks, stems = _load_generated_masks(nfe_dir, n_samples=args.n_samples)
        generated_stats = _summarize_distribution(
            generated_masks,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        nearest_real_mious = _nearest_real_mious(
            generated_masks,
            reference_masks,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        pairwise_fake_mious = _pairwise_fake_mious(
            generated_masks,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        results.append(
            {
                "nfe": int(nfe),
                "outdir": str(nfe_dir.resolve()),
                "generated_mask_count": len(generated_masks),
                "reference_mask_count": len(reference_masks),
                "mask_stem_example": None if not stems else stems[0],
                "nearest_real_miou_mean": float(nearest_real_mious.mean()),
                "nearest_real_miou_std": float(nearest_real_mious.std()),
                "pairwise_fake_miou_mean": None
                if pairwise_fake_mious.size == 0
                else float(pairwise_fake_mious.mean()),
                "pairwise_fake_miou_std": None
                if pairwise_fake_mious.size == 0
                else float(pairwise_fake_mious.std()),
                "class_ratio_l1": float(
                    np.abs(generated_stats["class_ratio_mean"] - reference_stats["class_ratio_mean"]).sum()
                ),
                "class_presence_l1": float(
                    np.abs(generated_stats["class_presence_mean"] - reference_stats["class_presence_mean"]).sum()
                ),
                "boundary_density_generated_mean": generated_stats["boundary_density_mean"],
                "boundary_density_reference_mean": reference_stats["boundary_density_mean"],
                "boundary_density_gap": abs(
                    generated_stats["boundary_density_mean"] - reference_stats["boundary_density_mean"]
                ),
                "unique_class_count_generated_mean": generated_stats["unique_class_count_mean"],
                "unique_class_count_reference_mean": reference_stats["unique_class_count_mean"],
                "unique_class_count_gap": abs(
                    generated_stats["unique_class_count_mean"] - reference_stats["unique_class_count_mean"]
                ),
                "generated_class_ratio_mean_json": json.dumps(
                    generated_stats["class_ratio_mean"].round(6).tolist(),
                    ensure_ascii=True,
                ),
                "reference_class_ratio_mean_json": json.dumps(
                    reference_stats["class_ratio_mean"].round(6).tolist(),
                    ensure_ascii=True,
                ),
                "generated_class_presence_mean_json": json.dumps(
                    generated_stats["class_presence_mean"].round(6).tolist(),
                    ensure_ascii=True,
                ),
                "reference_class_presence_mean_json": json.dumps(
                    reference_stats["class_presence_mean"].round(6).tolist(),
                    ensure_ascii=True,
                ),
                "metrics_note": (
                    "nearest_real_miou_mean measures layout plausibility against the real-mask bank. "
                    "pairwise_fake_miou_mean is a collapse sanity metric; class_ratio_l1 and boundary_density_gap "
                    "are marginal-distribution sanity metrics."
                ),
            }
        )

    summary = {
        "task": "p(semantic_mask)",
        "config": str(args.config.resolve()),
        "checkpoint": None if ckpt_path is None else str(ckpt_path),
        "generated_root": str(generated_root),
        "monitor": monitor,
        "seed": int(args.seed),
        "n_samples": int(args.n_samples),
        "nfe_values": [int(value) for value, _ in nfe_dirs],
        "primary_metric": "nearest_real_miou_mean",
        "reference_source": reference_source if args.mask_dir is None else str(args.mask_dir.resolve()),
        "label_spec": str(args.label_spec.resolve()),
        "num_classes": int(num_classes),
        "primary_readout": [
            "nearest_real_miou_mean",
            "pairwise_fake_miou_mean",
            "class_ratio_l1",
            "boundary_density_gap",
        ],
        "protocol_notes": {
            "task_type": "This route models p(semantic_mask) only.",
            "paired_target_note": "Unconditional mask generation has no paired ground truth; evaluation uses real-mask bank statistics and nearest-real agreement.",
            "nfe_rule": "Use the fixed NFE=8/4/2/1 sweep instead of reporting only NFE=1.",
            "collapse_note": "A very high pairwise_fake_miou_mean may indicate reduced diversity or mode collapse.",
        },
        "results": results,
    }

    summary_json_path = outdir / "summary.json"
    summary_csv_path = outdir / "summary.csv"
    report_md_path = outdir / "report.md"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "nfe",
                "outdir",
                "generated_mask_count",
                "reference_mask_count",
                "nearest_real_miou_mean",
                "nearest_real_miou_std",
                "pairwise_fake_miou_mean",
                "pairwise_fake_miou_std",
                "class_ratio_l1",
                "class_presence_l1",
                "boundary_density_generated_mean",
                "boundary_density_reference_mean",
                "boundary_density_gap",
                "unique_class_count_generated_mean",
                "unique_class_count_reference_mean",
                "unique_class_count_gap",
                "generated_class_ratio_mean_json",
                "reference_class_ratio_mean_json",
                "generated_class_presence_mean_json",
                "reference_class_presence_mean_json",
                "metrics_note",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    _write_markdown_report(report_md_path, summary)

    print(f"Saved mask-prior JSON to {summary_json_path}")
    print(f"Saved mask-prior CSV to {summary_csv_path}")
    print(f"Saved mask-prior report to {report_md_path}")


if __name__ == "__main__":
    main()

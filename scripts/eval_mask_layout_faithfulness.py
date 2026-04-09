import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LDM_ROOT = REPO_ROOT / "third_party" / "latent-diffusion"
TAMING_ROOT = LDM_ROOT / "taming-transformers"

for path in (REPO_ROOT, LDM_ROOT, TAMING_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from scripts.eval_mask_conditioned_image import _compute_rgb_metrics
from scripts.eval_semantic_pair_generation import (
    BoundaryF1Meter,
    HFSegmentationTeacher,
    _load_png_image_uint8,
    _load_png_mask,
    _resolve_teacher_remap,
    _sanitize_teacher_mask,
    _save_teacher_pair,
)
from scripts.sample_mask_conditioned_image import (
    DEFAULT_CONFIG,
    DEFAULT_NFE_VALUES,
    _prepare_outdir,
    apply_tokenizer_overrides,
    generate_mask_conditioned_sweep,
    load_config,
    load_examples,
    load_model,
    validate_ckpt_matches_config,
)
from latent_meanflow.trainers.mask_conditioned_image_trainer import resolve_mask_condition_channels


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate layout faithfulness for p(image | semantic_mask). "
            "This protocol reports teacher-aligned mask agreement under a fixed "
            "few-step NFE sweep. L1/LPIPS to GT remain sanity-only metrics."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument(
        "--tokenizer-config",
        type=Path,
        default=None,
        help="Optional override for model.params.tokenizer_config_path.",
    )
    parser.add_argument(
        "--tokenizer-ckpt",
        type=Path,
        default=None,
        help="Optional override for model.params.tokenizer_ckpt_path.",
    )
    parser.add_argument("--generated-root", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument(
        "--label-spec",
        type=Path,
        default=REPO_ROOT / "configs" / "label_specs" / "remote_semantic.yaml",
    )
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--nfe-values", type=int, nargs="+", default=DEFAULT_NFE_VALUES)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--overlay-alpha", type=float, default=0.4)
    parser.add_argument("--two-step-time", type=float, default=None)
    parser.add_argument("--boundary-tolerance-px", type=int, default=2)
    parser.add_argument(
        "--small-region-threshold-ratio",
        type=float,
        default=0.02,
        help="Heuristic threshold: target regions smaller than this fraction count toward small-region metrics.",
    )
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-monitor", type=str, default="val/base_error_mean")
    parser.add_argument(
        "--teacher-hf-model",
        type=str,
        default=None,
        help="Hugging Face segmentation teacher used to parse generated RGB images.",
    )
    parser.add_argument(
        "--teacher-mask-root",
        type=Path,
        default=None,
        help="Directory containing precomputed teacher masks in nfe*/teacher_mask_raw or nfe*/mask_raw.",
    )
    parser.add_argument(
        "--teacher-remap-json",
        type=Path,
        default=None,
        help="Optional JSON mapping from teacher label ids to this dataset's semantic ids.",
    )
    return parser.parse_args()


def _check_monitor(config, expected_monitor):
    configured_monitor = config.model.params.get("monitor")
    if expected_monitor is not None and configured_monitor != expected_monitor:
        raise ValueError(
            f"Evaluation monitor mismatch: expected '{expected_monitor}', got '{configured_monitor}'. "
            "Use best checkpoints selected by val/base_error_mean for the layout-faithfulness protocol."
        )
    return configured_monitor


def _count_pngs(path):
    if not path.exists():
        return 0
    return len(list(path.glob("*.png")))


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


def _collect_stems(nfe_dir, n_samples):
    image_dir = nfe_dir / "generated_image"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing generated_image directory under {nfe_dir}")
    stems = sorted(path.stem for path in image_dir.glob("*.png"))
    if not stems:
        raise RuntimeError(f"No generated images found under {image_dir}")
    return stems[: int(n_samples)]


def _load_target_masks(nfe_dir, stems, num_classes):
    masks = []
    raw_dir = nfe_dir / "input_mask_raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing input_mask_raw directory under {nfe_dir}")
    for stem in stems:
        mask = _load_png_mask(raw_dir / f"{stem}.png")
        masks.append(_sanitize_teacher_mask(mask, num_classes=num_classes, remap=None))
    return masks


def _load_teacher_predictions_from_root(root, nfe, stems, num_classes, remap):
    nfe_dir = root / f"nfe{int(nfe)}"
    candidate_dirs = [nfe_dir / "teacher_mask_raw", nfe_dir / "mask_raw"]
    mask_dir = next((candidate for candidate in candidate_dirs if candidate.exists()), None)
    if mask_dir is None:
        raise FileNotFoundError(
            f"Teacher mask directory not found under {nfe_dir}. Expected teacher_mask_raw/ or mask_raw/."
        )
    masks = []
    for stem in stems:
        mask_path = mask_dir / f"{stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing teacher mask for stem {stem}: {mask_path}")
        mask = _load_png_mask(mask_path)
        masks.append(_sanitize_teacher_mask(mask, num_classes=num_classes, remap=remap))
    return masks


def _compute_layout_metrics(
    *,
    target_masks,
    teacher_masks,
    num_classes,
    boundary_tolerance_px,
    small_region_threshold_ratio,
):
    if len(target_masks) != len(teacher_masks):
        raise ValueError(
            f"Target/teacher mask count mismatch: {len(target_masks)} target masks vs {len(teacher_masks)} teacher masks."
        )

    intersections = torch.zeros(num_classes, dtype=torch.float64)
    unions = torch.zeros(num_classes, dtype=torch.float64)
    small_intersections = torch.zeros(num_classes, dtype=torch.float64)
    small_unions = torch.zeros(num_classes, dtype=torch.float64)
    small_region_hits = torch.zeros(num_classes, dtype=torch.float64)
    matched_pixels = 0.0
    total_pixels = 0.0
    boundary_meter = BoundaryF1Meter(tolerance_px=boundary_tolerance_px)

    for target_mask, teacher_mask in zip(target_masks, teacher_masks):
        valid_mask = (target_mask >= 0) & (teacher_mask >= 0)
        matched_pixels += float((target_mask[valid_mask] == teacher_mask[valid_mask]).sum().item())
        total_pixels += float(valid_mask.sum().item())

        target_valid = target_mask[valid_mask]
        teacher_valid = teacher_mask[valid_mask]
        valid_pixels = max(1.0, float(valid_mask.sum().item()))
        for class_id in range(num_classes):
            target_class = target_valid == class_id
            teacher_class = teacher_valid == class_id
            intersection = float(torch.logical_and(target_class, teacher_class).sum().item())
            union = float(torch.logical_or(target_class, teacher_class).sum().item())
            intersections[class_id] += intersection
            unions[class_id] += union

            target_area = float(target_class.sum().item())
            if target_area > 0.0 and (target_area / valid_pixels) <= float(small_region_threshold_ratio):
                small_intersections[class_id] += intersection
                small_unions[class_id] += union
                small_region_hits[class_id] += 1.0

        teacher_for_boundary = teacher_mask.clone()
        target_for_boundary = target_mask.clone()
        teacher_for_boundary[~valid_mask] = -1
        target_for_boundary[~valid_mask] = -1
        boundary_meter.update(teacher_for_boundary, target_for_boundary)

    teacher_per_class_iou = {}
    teacher_iou_values = []
    small_region_per_class_iou = {}
    small_region_iou_values = []
    for class_id in range(num_classes):
        union = float(unions[class_id].item())
        iou = None if union <= 0.0 else float(intersections[class_id].item() / union)
        teacher_per_class_iou[int(class_id)] = iou
        if iou is not None:
            teacher_iou_values.append(iou)

        small_union = float(small_unions[class_id].item())
        small_iou = None if small_union <= 0.0 else float(small_intersections[class_id].item() / small_union)
        small_region_per_class_iou[int(class_id)] = small_iou
        if small_iou is not None:
            small_region_iou_values.append(small_iou)

    return {
        "teacher_miou": None if not teacher_iou_values else float(sum(teacher_iou_values) / len(teacher_iou_values)),
        "teacher_per_class_iou": teacher_per_class_iou,
        "boundary_f1": float(boundary_meter.compute()),
        "layout_pixel_accuracy": None if total_pixels <= 0.0 else float(matched_pixels / total_pixels),
        "small_region_miou": None
        if not small_region_iou_values
        else float(sum(small_region_iou_values) / len(small_region_iou_values)),
        "small_region_per_class_iou": small_region_per_class_iou,
        "small_region_threshold_ratio": float(small_region_threshold_ratio),
        "small_region_active_class_count": int((small_region_hits > 0).sum().item()),
    }


def _resolve_condition_metadata(config):
    params = config.model.params
    backbone_params = config.model.params.backbone_config.params
    return {
        "condition_mode": backbone_params.get("condition_mode"),
        "condition_source": params.get("condition_source", backbone_params.get("condition_source")),
        "use_boundary_condition": bool(backbone_params.get("use_boundary_condition", False)),
        "boundary_mode": backbone_params.get("boundary_mode"),
        "use_semantic_condition_encoder": bool(backbone_params.get("use_semantic_condition_encoder", False)),
    }


def _resolve_num_classes(config, label_spec_override):
    label_spec = config.model.params.get("semantic_mask_label_spec_path", None)
    if label_spec_override is not None:
        label_spec = label_spec_override
    num_classes = resolve_mask_condition_channels(semantic_mask_label_spec_path=label_spec)
    if num_classes is None:
        raise ValueError("Could not infer semantic mask class count for layout-faithfulness evaluation.")
    return int(num_classes)


def _format_value(value, precision=4):
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{precision}f}"


def _write_markdown_report(path, summary):
    lines = [
        "# Mask Layout Faithfulness Report",
        "",
        f"- task: `{summary['task']}`",
        f"- config: `{summary['config']}`",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- monitor: `{summary['monitor']}`",
        f"- primary metric: `{summary['primary_metric']}`",
        f"- teacher source: `{summary['teacher_source']}`",
        f"- condition mode: `{summary['condition_mode']}`",
        f"- condition source: `{summary['condition_source']}`",
        f"- boundary mode: `{summary['boundary_mode']}`",
        f"- semantic condition encoder: `{summary['use_semantic_condition_encoder']}`",
        "",
        "Primary readout: teacher-aligned layout faithfulness. `L1` / `LPIPS` remain sanity-only because `p(image | semantic_mask)` is one-to-many.",
        "",
        "| NFE | teacher mIoU | Boundary F1 | layout pixel acc | small-region mIoU | L1 sanity | LPIPS sanity |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in summary["results"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_value(result["nfe"], precision=0),
                    _format_value(result["teacher_miou"]),
                    _format_value(result["boundary_f1"]),
                    _format_value(result["layout_pixel_accuracy"]),
                    _format_value(result["small_region_miou"]),
                    _format_value(result["l1_mean"]),
                    _format_value(result["lpips_mean"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Per-class teacher IoU is stored in `summary.json` / `summary.csv`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main():
    args = parse_args()
    if (args.teacher_hf_model is None) == (args.teacher_mask_root is None):
        raise ValueError(
            "Pass exactly one teacher source: either --teacher-hf-model or --teacher-mask-root."
        )

    config = load_config(args.config)
    apply_tokenizer_overrides(
        config,
        tokenizer_config=args.tokenizer_config,
        tokenizer_ckpt=args.tokenizer_ckpt,
    )
    monitor = _check_monitor(config, expected_monitor=args.expected_monitor)
    condition_metadata = _resolve_condition_metadata(config)
    num_classes = _resolve_num_classes(config, args.label_spec)

    outdir = args.outdir.resolve()
    _prepare_outdir(outdir, overwrite=args.overwrite)

    generated_root = None if args.generated_root is None else args.generated_root.resolve()
    source_mode = None
    ckpt_path = None if args.ckpt is None else args.ckpt.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))

    if generated_root is None:
        if ckpt_path is None:
            raise ValueError("--ckpt is required unless --generated-root is provided.")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
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

    nfe_dirs = _resolve_nfe_dirs(generated_root, args.nfe_values)
    if not nfe_dirs:
        raise FileNotFoundError(f"No nfe* directories found under {generated_root}")

    teacher_remap = _resolve_teacher_remap(args.teacher_remap_json)
    teacher_model = None
    if args.teacher_hf_model is not None:
        teacher_model = HFSegmentationTeacher(args.teacher_hf_model, device=device)
    teacher_mask_root = None if args.teacher_mask_root is None else args.teacher_mask_root.resolve()

    results = []
    for nfe, nfe_dir in nfe_dirs:
        stems = _collect_stems(nfe_dir, n_samples=args.n_samples)
        target_masks = _load_target_masks(nfe_dir, stems=stems, num_classes=num_classes)

        teacher_masks = []
        rgb_batch = []
        rgb_stems = []

        def flush_teacher_batch():
            nonlocal rgb_batch, rgb_stems, teacher_masks
            if not rgb_batch or teacher_model is None:
                return
            rgb_tensor = torch.stack(rgb_batch, dim=0)
            predicted_masks = teacher_model.predict(rgb_tensor.to(device))
            for stem, rgb_uint8, teacher_mask in zip(rgb_stems, rgb_batch, predicted_masks):
                teacher_mask = _sanitize_teacher_mask(
                    teacher_mask,
                    num_classes=num_classes,
                    remap=teacher_remap,
                )
                teacher_masks.append(teacher_mask)
                _save_teacher_pair(
                    teacher_mask=teacher_mask,
                    rgb_uint8=rgb_uint8,
                    outdir=nfe_dir,
                    stem=stem,
                    num_classes=num_classes,
                    overlay_alpha=args.overlay_alpha,
                )
            rgb_batch = []
            rgb_stems = []

        if teacher_mask_root is not None:
            teacher_masks = _load_teacher_predictions_from_root(
                root=teacher_mask_root,
                nfe=nfe,
                stems=stems,
                num_classes=num_classes,
                remap=teacher_remap,
            )
        else:
            for stem in stems:
                rgb_uint8 = _load_png_image_uint8(nfe_dir / "generated_image" / f"{stem}.png")
                rgb_batch.append(rgb_uint8)
                rgb_stems.append(stem)
                if len(rgb_batch) >= 8:
                    flush_teacher_batch()
            flush_teacher_batch()

        layout_metrics = _compute_layout_metrics(
            target_masks=target_masks,
            teacher_masks=teacher_masks,
            num_classes=num_classes,
            boundary_tolerance_px=args.boundary_tolerance_px,
            small_region_threshold_ratio=args.small_region_threshold_ratio,
        )
        gt_metrics = _compute_rgb_metrics(
            nfe_dir / "generated_image",
            nfe_dir / "ground_truth_image",
            device=device,
            compute_lpips=not args.skip_lpips,
        )

        results.append(
            {
                "nfe": int(nfe),
                "outdir": str(nfe_dir.resolve()),
                "input_mask_raw_count": _count_pngs(nfe_dir / "input_mask_raw"),
                "input_mask_color_count": _count_pngs(nfe_dir / "input_mask_color"),
                "generated_image_count": _count_pngs(nfe_dir / "generated_image"),
                "ground_truth_image_count": _count_pngs(nfe_dir / "ground_truth_image"),
                "overlay_count": _count_pngs(nfe_dir / "overlay"),
                "panel_count": _count_pngs(nfe_dir / "panel"),
                "teacher_mask_count": _count_pngs(nfe_dir / "teacher_mask_raw"),
                "teacher_mask_color_count": _count_pngs(nfe_dir / "teacher_mask_color"),
                "teacher_overlay_count": _count_pngs(nfe_dir / "teacher_overlay"),
                "teacher_miou": layout_metrics["teacher_miou"],
                "teacher_per_class_iou_json": json.dumps(
                    layout_metrics["teacher_per_class_iou"],
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                "boundary_f1": layout_metrics["boundary_f1"],
                "layout_pixel_accuracy": layout_metrics["layout_pixel_accuracy"],
                "small_region_miou": layout_metrics["small_region_miou"],
                "small_region_per_class_iou_json": json.dumps(
                    layout_metrics["small_region_per_class_iou"],
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                "small_region_threshold_ratio": layout_metrics["small_region_threshold_ratio"],
                "small_region_active_class_count": layout_metrics["small_region_active_class_count"],
                "l1_mean": None if gt_metrics is None else gt_metrics["l1_mean"],
                "l1_std": None if gt_metrics is None else gt_metrics["l1_std"],
                "lpips_mean": None if gt_metrics is None else gt_metrics["lpips_mean"],
                "lpips_std": None if gt_metrics is None else gt_metrics["lpips_std"],
                "metrics_note": "Primary metrics are teacher_miou and boundary_f1. L1/LPIPS are sanity-only.",
            }
        )

    summary = {
        "task": "p(image | semantic_mask)",
        "config": str(args.config.resolve()),
        "checkpoint": None if ckpt_path is None else str(ckpt_path),
        "tokenizer_config": str(Path(config.model.params.tokenizer_config_path).resolve()),
        "tokenizer_checkpoint": str(Path(config.model.params.tokenizer_ckpt_path).resolve()),
        "generated_root": str(generated_root),
        "monitor": monitor,
        "source_mode": source_mode,
        "seed": int(args.seed),
        "n_samples": int(args.n_samples),
        "nfe_values": [int(value) for value, _ in nfe_dirs],
        "primary_metric": "teacher_miou",
        "teacher_source": args.teacher_hf_model if args.teacher_hf_model is not None else str(teacher_mask_root),
        "teacher_remap_json": None if args.teacher_remap_json is None else str(args.teacher_remap_json.resolve()),
        "boundary_tolerance_px": int(args.boundary_tolerance_px),
        "small_region_threshold_ratio": float(args.small_region_threshold_ratio),
        "condition_mode": condition_metadata["condition_mode"],
        "condition_source": condition_metadata["condition_source"],
        "use_boundary_condition": condition_metadata["use_boundary_condition"],
        "boundary_mode": condition_metadata["boundary_mode"],
        "use_semantic_condition_encoder": condition_metadata["use_semantic_condition_encoder"],
        "primary_readout": [
            "teacher_miou",
            "boundary_f1",
            "layout_pixel_accuracy",
            "small_region_miou",
        ],
        "sanity_only_metrics": [
            "l1_mean",
            "lpips_mean",
        ],
        "protocol_notes": {
            "checkpoint_rule": "Use best checkpoints selected by val/base_error_mean.",
            "task_type": "This is a one-to-many p(image | semantic_mask) task.",
            "teacher_definition": "Layout faithfulness compares the input semantic mask M to teacher segmentation S(I_hat).",
            "nfe_rule": "Do not report only NFE=1. Use the fixed NFE=8/4/2/1 sweep.",
            "small_region_metric": "small_region_miou is a heuristic subset metric over classes occupying <= small_region_threshold_ratio of valid pixels in a sample.",
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
                "input_mask_raw_count",
                "input_mask_color_count",
                "generated_image_count",
                "ground_truth_image_count",
                "overlay_count",
                "panel_count",
                "teacher_mask_count",
                "teacher_mask_color_count",
                "teacher_overlay_count",
                "teacher_miou",
                "boundary_f1",
                "layout_pixel_accuracy",
                "small_region_miou",
                "small_region_threshold_ratio",
                "small_region_active_class_count",
                "l1_mean",
                "l1_std",
                "lpips_mean",
                "lpips_std",
                "teacher_per_class_iou_json",
                "small_region_per_class_iou_json",
                "metrics_note",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    _write_markdown_report(report_md_path, summary)

    print(f"Saved layout-faithfulness JSON to {summary_json_path}")
    print(f"Saved layout-faithfulness CSV to {summary_csv_path}")
    print(f"Saved layout-faithfulness report to {report_md_path}")


if __name__ == "__main__":
    main()

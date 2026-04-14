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

from latent_meanflow.utils.segmentation_teacher import (
    BoundaryF1Meter,
    load_teacher_model,
    predict_masks_for_paths,
    resolve_label_spec_metadata,
    write_teacher_mask_triplet,
)
from scripts.sample_latent_flow import (
    load_config,
    load_model,
    validate_ckpt_matches_config,
)
from scripts.sample_mask_prior import (
    DEFAULT_NFE_VALUES,
    _prepare_outdir,
    _resolve_nfe_dirs,
    generate_mask_prior_sweep,
)
from scripts.sample_mask_conditioned_image import (
    apply_tokenizer_overrides,
    generate_mask_conditioned_sweep,
)


DEFAULT_MASK_CONFIG = REPO_ROOT / "configs" / "latent_alphaflow_mask_prior_unet.yaml"
DEFAULT_RENDERER_CONFIG = (
    REPO_ROOT
    / "configs"
    / "ablations"
    / "latent_alphaflow_mask2image_unet_fullres_pyramid_boundary_encoder.yaml"
)
DEFAULT_RENDERER_TOKENIZER_CONFIG = REPO_ROOT / "configs" / "autoencoder_image_lpips_adv_256.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compose unconditional p(semantic_mask) samples through the frozen p(image | semantic_mask) renderer, "
            "then evaluate rendered images with the frozen in-domain segmentation teacher."
        )
    )
    parser.add_argument("--mask-config", type=Path, default=DEFAULT_MASK_CONFIG)
    parser.add_argument("--mask-ckpt", type=Path, default=None)
    parser.add_argument("--mask-generated-root", type=Path, default=None)
    parser.add_argument("--renderer-config", type=Path, default=DEFAULT_RENDERER_CONFIG)
    parser.add_argument("--renderer-ckpt", type=Path, default=None)
    parser.add_argument("--renderer-tokenizer-config", type=Path, default=DEFAULT_RENDERER_TOKENIZER_CONFIG)
    parser.add_argument("--renderer-tokenizer-ckpt", type=Path, default=None)
    parser.add_argument("--composed-root", type=Path, default=None)
    parser.add_argument("--teacher-run-dir", type=Path, required=True)
    parser.add_argument("--teacher-ckpt", type=Path, default=None)
    parser.add_argument(
        "--label-spec",
        type=Path,
        default=REPO_ROOT / "configs" / "label_specs" / "remote_semantic.yaml",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--mask-batch-size", type=int, default=8)
    parser.add_argument("--renderer-batch-size", type=int, default=4)
    parser.add_argument("--teacher-batch-size", type=int, default=4)
    parser.add_argument("--mask-nfe-values", type=int, nargs="+", default=DEFAULT_NFE_VALUES)
    parser.add_argument("--renderer-nfe-values", type=int, nargs="+", default=DEFAULT_NFE_VALUES)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--overlay-alpha", type=float, default=0.4)
    parser.add_argument("--mask-two-step-time", type=float, default=None)
    parser.add_argument("--renderer-two-step-time", type=float, default=None)
    parser.add_argument("--boundary-tolerance-px", type=int, default=2)
    parser.add_argument("--small-region-threshold-ratio", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-mask-monitor", type=str, default="val/base_error_mean")
    parser.add_argument("--expected-renderer-monitor", type=str, default="val/base_error_mean")
    return parser.parse_args()


def _check_monitor(config, expected_monitor, route_name):
    configured_monitor = config.model.params.get("monitor")
    if expected_monitor is not None and configured_monitor != expected_monitor:
        raise ValueError(
            f"{route_name} monitor mismatch: expected '{expected_monitor}', got '{configured_monitor}'. "
            "Use best checkpoints selected by val/base_error_mean for the frozen-compose protocol."
        )
    return configured_monitor


def _load_mask_png(path):
    mask = np.asarray(Image.open(path), dtype=np.int64)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask png, got shape {mask.shape} from {path}")
    return mask


def _build_onehot(mask_index, num_classes, ignore_index=None):
    mask_index = np.asarray(mask_index, dtype=np.int64)
    onehot = np.zeros(mask_index.shape + (int(num_classes),), dtype=np.float32)
    valid_mask = np.ones(mask_index.shape, dtype=bool)
    if ignore_index is not None:
        valid_mask &= mask_index != int(ignore_index)
    if np.any(valid_mask):
        flat_onehot = onehot.reshape(-1, int(num_classes))
        flat_index = mask_index.reshape(-1)
        valid_positions = np.nonzero(valid_mask.reshape(-1))[0]
        flat_onehot[valid_positions, flat_index[valid_positions]] = 1.0
    return onehot


def _load_generated_mask_examples(mask_nfe_dir, *, num_classes, ignore_index, n_samples):
    mask_dir = Path(mask_nfe_dir) / "mask_raw"
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask_raw directory under {mask_nfe_dir}")
    mask_paths = sorted(mask_dir.glob("*.png"))[: int(n_samples)]
    if not mask_paths:
        raise FileNotFoundError(f"No generated masks found under {mask_dir}")

    examples = []
    for mask_path in mask_paths:
        mask_index = _load_mask_png(mask_path)
        if np.any(mask_index < 0):
            raise ValueError(f"Generated mask contains negative class ids: {mask_path}")
        if int(mask_index.max()) >= int(num_classes):
            raise ValueError(
                f"Generated mask {mask_path} contains class id {int(mask_index.max())}, "
                f"but label spec expects classes [0, {int(num_classes) - 1}]"
            )
        examples.append(
            {
                "stem": mask_path.stem,
                "mask_index": mask_index,
                "mask_onehot": _build_onehot(mask_index, num_classes=num_classes, ignore_index=ignore_index),
                "ground_truth_image": None,
                "image_path": None,
                "mask_path": str(mask_path),
            }
        )
    return examples


def _collect_stems(nfe_dir, n_samples):
    image_dir = Path(nfe_dir) / "generated_image"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing generated_image directory under {nfe_dir}")
    stems = sorted(path.stem for path in image_dir.glob("*.png"))
    if not stems:
        raise RuntimeError(f"No generated images found under {image_dir}")
    return stems[: int(n_samples)]


def _resolve_mask_nfe_dirs(root, requested_nfe_values):
    root = Path(root)
    resolved = []
    for nfe in requested_nfe_values:
        path = root / f"mask_nfe{int(nfe)}"
        if path.exists():
            resolved.append((int(nfe), path))
    if resolved:
        return resolved
    auto_dirs = []
    for path in sorted(root.glob("mask_nfe*")):
        try:
            nfe = int(path.name.replace("mask_nfe", ""))
        except ValueError:
            continue
        auto_dirs.append((nfe, path))
    return auto_dirs


def _load_target_masks(nfe_dir, stems):
    raw_dir = Path(nfe_dir) / "input_mask_raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing input_mask_raw directory under {nfe_dir}")
    return [torch.from_numpy(_load_mask_png(raw_dir / f"{stem}.png").astype(np.int64, copy=False)) for stem in stems]


def _load_rgb_uint8(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


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


def _write_markdown_report(path, summary):
    lines = [
        "# Mask Prior Composed Renderer Evaluation",
        "",
        f"- task decomposition: `{summary['task_decomposition']}`",
        f"- mask config: `{summary['mask_config']}`",
        f"- renderer config: `{summary['renderer_config']}`",
        f"- teacher run dir: `{summary['teacher_run_dir']}`",
        f"- mask checkpoint: `{summary['mask_checkpoint']}`",
        f"- renderer checkpoint: `{summary['renderer_checkpoint']}`",
        "",
        "This protocol freezes the downstream `p(image | semantic_mask)` renderer and the in-domain teacher. It measures whether sampled `p(mask)` layouts remain valid when composed through the renderer.",
        "",
        "| mask NFE | renderer NFE | teacher mIoU | Boundary F1 | layout pixel acc | small-region mIoU |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in summary["results"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(result["mask_nfe"])),
                    str(int(result["renderer_nfe"])),
                    f"{float(result['teacher_miou']):.4f}" if result["teacher_miou"] is not None else "n/a",
                    f"{float(result['boundary_f1']):.4f}" if result["boundary_f1"] is not None else "n/a",
                    f"{float(result['layout_pixel_accuracy']):.4f}"
                    if result["layout_pixel_accuracy"] is not None
                    else "n/a",
                    f"{float(result['small_region_miou']):.4f}" if result["small_region_miou"] is not None else "n/a",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `teacher_miou`, `boundary_f1`, `layout_pixel_accuracy`, and `small_region_miou` are the primary compose metrics.",
            "- If these collapse while the frozen renderer route is already validated, the first suspect is the sampled mask distribution rather than the renderer itself.",
            "- Formal reporting uses the fixed `mask_nfe=8/4/2/1` and `renderer_nfe=8/4/2/1` sweep with the same seed and sample count.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main():
    args = parse_args()
    outdir = args.outdir.resolve()
    _prepare_outdir(outdir, overwrite=args.overwrite)

    mask_config = load_config(args.mask_config)
    renderer_config = load_config(args.renderer_config)
    mask_monitor = _check_monitor(mask_config, args.expected_mask_monitor, route_name="mask prior")
    renderer_monitor = _check_monitor(renderer_config, args.expected_renderer_monitor, route_name="renderer")
    apply_tokenizer_overrides(
        renderer_config,
        tokenizer_config=args.renderer_tokenizer_config,
        tokenizer_ckpt=args.renderer_tokenizer_ckpt,
    )

    label_metadata = resolve_label_spec_metadata(args.label_spec)
    num_classes = int(label_metadata["num_classes"])
    ignore_index = label_metadata["ignore_index"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))

    mask_generated_root = None if args.mask_generated_root is None else args.mask_generated_root.resolve()
    composed_root = None if args.composed_root is None else args.composed_root.resolve()
    mask_ckpt_path = None if args.mask_ckpt is None else args.mask_ckpt.resolve()
    renderer_ckpt_path = None if args.renderer_ckpt is None else args.renderer_ckpt.resolve()

    if mask_generated_root is None:
        if mask_ckpt_path is None or not mask_ckpt_path.exists():
            raise FileNotFoundError("Mask-prior checkpoint not found. Pass --mask-ckpt explicitly.")
        validate_ckpt_matches_config(args.mask_config, mask_ckpt_path)
        mask_model = load_model(mask_config, mask_ckpt_path, device=device)
        if args.mask_two_step_time is not None and hasattr(mask_model, "sampler") and hasattr(mask_model.sampler, "two_step_time"):
            mask_model.sampler.two_step_time = float(args.mask_two_step_time)
        mask_generated_root = outdir / "mask_prior_generated"
        _prepare_outdir(mask_generated_root, overwrite=args.overwrite)
        generate_mask_prior_sweep(
            model=mask_model,
            outdir=mask_generated_root,
            nfe_values=args.mask_nfe_values,
            seed=args.seed,
            n_samples=args.n_samples,
            batch_size=args.mask_batch_size,
        )

    if composed_root is None:
        if renderer_ckpt_path is None or not renderer_ckpt_path.exists():
            raise FileNotFoundError("Renderer checkpoint not found. Pass --renderer-ckpt explicitly.")
        validate_ckpt_matches_config(args.renderer_config, renderer_ckpt_path)
        renderer_model = load_model(renderer_config, renderer_ckpt_path, device=device)
        if not hasattr(renderer_model, "build_condition_from_mask_onehot"):
            raise TypeError(f"Config '{args.renderer_config.name}' is not a mask-conditioned image route.")
        if args.renderer_two_step_time is not None and hasattr(renderer_model, "sampler") and hasattr(renderer_model.sampler, "two_step_time"):
            renderer_model.sampler.two_step_time = float(args.renderer_two_step_time)

        composed_root = outdir / "composed_renderer"
        _prepare_outdir(composed_root, overwrite=args.overwrite)
        mask_nfe_dirs = _resolve_nfe_dirs(mask_generated_root, args.mask_nfe_values)
        if not mask_nfe_dirs:
            raise FileNotFoundError(f"No nfe* directories found under {mask_generated_root}")
        for mask_nfe, mask_nfe_dir in mask_nfe_dirs:
            examples = _load_generated_mask_examples(
                mask_nfe_dir,
                num_classes=num_classes,
                ignore_index=ignore_index,
                n_samples=args.n_samples,
            )
            compose_outdir = composed_root / f"mask_nfe{int(mask_nfe)}"
            _prepare_outdir(compose_outdir, overwrite=True)
            generate_mask_conditioned_sweep(
                model=renderer_model,
                examples=examples,
                outdir=compose_outdir,
                nfe_values=args.renderer_nfe_values,
                seed=args.seed,
                batch_size=args.renderer_batch_size,
                overlay_alpha=args.overlay_alpha,
            )

    teacher_model, teacher_metadata = load_teacher_model(
        run_dir=args.teacher_run_dir.resolve(),
        device=device,
        checkpoint_path=None if args.teacher_ckpt is None else args.teacher_ckpt.resolve(),
    )
    teacher_input_size_hw = (int(teacher_metadata["height"]), int(teacher_metadata["width"]))

    results = []
    mask_nfe_dirs = _resolve_mask_nfe_dirs(composed_root, args.mask_nfe_values)
    if not mask_nfe_dirs:
        raise FileNotFoundError(f"No mask_nfe* directories found under {composed_root}")
    for mask_nfe, compose_root in mask_nfe_dirs:
        renderer_nfe_dirs = _resolve_nfe_dirs(compose_root, args.renderer_nfe_values)
        if not renderer_nfe_dirs:
            raise FileNotFoundError(f"No renderer nfe* directories found under {compose_root}")
        for renderer_nfe, renderer_nfe_dir in renderer_nfe_dirs:
            stems = _collect_stems(renderer_nfe_dir, n_samples=args.n_samples)
            target_masks = _load_target_masks(renderer_nfe_dir, stems=stems)
            image_paths = [renderer_nfe_dir / "generated_image" / f"{stem}.png" for stem in stems]
            teacher_masks = predict_masks_for_paths(
                image_paths=image_paths,
                model=teacher_model,
                input_size_hw=teacher_input_size_hw,
                device=device,
                batch_size=args.teacher_batch_size,
                output_size_mode="original",
            )
            for stem, image_path, teacher_mask in zip(stems, image_paths, teacher_masks):
                write_teacher_mask_triplet(
                    mask_index=teacher_mask.cpu().numpy(),
                    rgb_uint8=_load_rgb_uint8(image_path),
                    outdir=renderer_nfe_dir,
                    stem=stem,
                    num_classes=num_classes,
                    overlay_alpha=args.overlay_alpha,
                )

            layout_metrics = _compute_layout_metrics(
                target_masks=target_masks,
                teacher_masks=teacher_masks,
                num_classes=num_classes,
                boundary_tolerance_px=args.boundary_tolerance_px,
                small_region_threshold_ratio=args.small_region_threshold_ratio,
            )
            results.append(
                {
                    "mask_nfe": int(mask_nfe),
                    "renderer_nfe": int(renderer_nfe),
                    "outdir": str(renderer_nfe_dir.resolve()),
                    "input_mask_raw_count": len(list((renderer_nfe_dir / "input_mask_raw").glob("*.png"))),
                    "input_mask_color_count": len(list((renderer_nfe_dir / "input_mask_color").glob("*.png"))),
                    "generated_image_count": len(list((renderer_nfe_dir / "generated_image").glob("*.png"))),
                    "overlay_count": len(list((renderer_nfe_dir / "overlay").glob("*.png"))),
                    "panel_count": len(list((renderer_nfe_dir / "panel").glob("*.png"))),
                    "teacher_mask_count": len(list((renderer_nfe_dir / "teacher_mask_raw").glob("*.png"))),
                    "teacher_mask_color_count": len(list((renderer_nfe_dir / "teacher_mask_color").glob("*.png"))),
                    "teacher_overlay_count": len(list((renderer_nfe_dir / "teacher_overlay").glob("*.png"))),
                    "teacher_miou": layout_metrics["teacher_miou"],
                    "boundary_f1": layout_metrics["boundary_f1"],
                    "layout_pixel_accuracy": layout_metrics["layout_pixel_accuracy"],
                    "small_region_miou": layout_metrics["small_region_miou"],
                    "small_region_threshold_ratio": layout_metrics["small_region_threshold_ratio"],
                    "small_region_active_class_count": layout_metrics["small_region_active_class_count"],
                    "teacher_per_class_iou_json": json.dumps(
                        layout_metrics["teacher_per_class_iou"],
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                    "small_region_per_class_iou_json": json.dumps(
                        layout_metrics["small_region_per_class_iou"],
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                    "metrics_note": (
                        "Compose metrics are measured with a frozen renderer and a frozen in-domain teacher. "
                        "If this protocol collapses, suspect p(mask) first."
                    ),
                }
            )

    summary = {
        "task_decomposition": "p(mask) + p(image | semantic_mask)",
        "mask_config": str(args.mask_config.resolve()),
        "mask_checkpoint": None if mask_ckpt_path is None else str(mask_ckpt_path),
        "mask_generated_root": None if mask_generated_root is None else str(mask_generated_root),
        "mask_monitor": mask_monitor,
        "renderer_config": str(args.renderer_config.resolve()),
        "renderer_checkpoint": None if renderer_ckpt_path is None else str(renderer_ckpt_path),
        "renderer_tokenizer_config": str(Path(renderer_config.model.params.tokenizer_config_path).resolve()),
        "renderer_tokenizer_checkpoint": str(Path(renderer_config.model.params.tokenizer_ckpt_path).resolve()),
        "renderer_monitor": renderer_monitor,
        "composed_root": str(composed_root),
        "teacher_run_dir": str(args.teacher_run_dir.resolve()),
        "teacher_checkpoint": teacher_metadata["checkpoint_path"],
        "teacher_net_name": teacher_metadata["net_name"],
        "label_spec": str(args.label_spec.resolve()),
        "seed": int(args.seed),
        "n_samples": int(args.n_samples),
        "mask_nfe_values": [int(value) for value, _ in mask_nfe_dirs],
        "renderer_nfe_values": sorted({int(row["renderer_nfe"]) for row in results}),
        "primary_metric": "teacher_miou",
        "primary_readout": [
            "teacher_miou",
            "boundary_f1",
            "layout_pixel_accuracy",
            "small_region_miou",
        ],
        "protocol_notes": {
            "task_type": "This protocol evaluates whether sampled p(mask) outputs remain valid when composed through the frozen p(image | semantic_mask) renderer.",
            "renderer_status": "The downstream renderer is frozen and acts as an evaluator, not a trainable part of this protocol.",
            "teacher_status": "The in-domain segmentation teacher is frozen and evaluated live on composed renderer outputs.",
            "checkpoint_rule": "Use best checkpoints selected by val/base_error_mean for both p(mask) and p(image | semantic_mask).",
            "nfe_rule": "Formal reporting uses the fixed mask NFE=8/4/2/1 and renderer NFE=8/4/2/1 sweep.",
            "seed_rule": "Use the same seed and the same sample count for every compose comparison.",
        },
        "results": results,
    }

    summary_json_path = outdir / "summary.json"
    summary_csv_path = outdir / "summary.csv"
    summary_md_path = outdir / "summary.md"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mask_nfe",
                "renderer_nfe",
                "outdir",
                "input_mask_raw_count",
                "input_mask_color_count",
                "generated_image_count",
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
                "teacher_per_class_iou_json",
                "small_region_per_class_iou_json",
                "metrics_note",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    _write_markdown_report(summary_md_path, summary)

    print(f"Saved composed-renderer JSON to {summary_json_path}")
    print(f"Saved composed-renderer CSV to {summary_csv_path}")
    print(f"Saved composed-renderer markdown to {summary_md_path}")


if __name__ == "__main__":
    main()

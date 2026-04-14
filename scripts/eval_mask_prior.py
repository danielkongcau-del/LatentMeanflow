import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
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
AREA_RATIO_HISTOGRAM_BINS = np.asarray(
    [0.0, 1.0e-4, 1.0e-3, 5.0e-3, 1.0e-2, 2.0e-2, 5.0e-2, 1.0e-1, 2.0e-1, 5.0e-1, 1.0],
    dtype=np.float64,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the project-layer unconditional semantic-mask route p(semantic_mask). "
            "This protocol is distributional: it compares generated masks against a real-mask bank "
            "using class-area, topology, boundary, and small-region statistics instead of paired IoU."
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
    parser.add_argument(
        "--small-region-threshold-ratio",
        type=float,
        default=0.02,
        help="Connected components no larger than this ratio of valid pixels count as small regions.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-monitor", type=str, default="val/base_error_mean")
    return parser.parse_args()


def _check_monitor(config, expected_monitor):
    configured_monitor = config.model.params.get("monitor")
    if expected_monitor is not None and configured_monitor != expected_monitor:
        raise ValueError(
            f"Evaluation monitor mismatch: expected '{expected_monitor}', got '{configured_monitor}'. "
            "Use the best checkpoint selected by val/base_error_mean for the main mask-prior baseline."
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


def _valid_mask(mask, ignore_index=None):
    valid_mask = np.ones(mask.shape, dtype=bool)
    if ignore_index is not None:
        valid_mask &= mask != int(ignore_index)
    return valid_mask


def _class_ratio(mask, num_classes, ignore_index=None):
    valid_mask = _valid_mask(mask, ignore_index=ignore_index)
    valid_values = mask[valid_mask]
    if valid_values.size == 0:
        return np.zeros((num_classes,), dtype=np.float64), np.zeros((num_classes,), dtype=np.float64), 0.0
    counts = np.bincount(valid_values.astype(np.int64), minlength=num_classes).astype(np.float64)
    ratios = counts / float(valid_values.size)
    return ratios, counts, float(valid_values.size)


def _boundary_length(mask, ignore_index=None):
    valid_mask = _valid_mask(mask, ignore_index=ignore_index)
    vertical = (mask[1:, :] != mask[:-1, :]) & valid_mask[1:, :] & valid_mask[:-1, :]
    horizontal = (mask[:, 1:] != mask[:, :-1]) & valid_mask[:, 1:] & valid_mask[:, :-1]
    boundary_length = float(vertical.sum() + horizontal.sum())
    valid_pixels = float(max(1, valid_mask.sum()))
    return boundary_length, boundary_length / valid_pixels


def _component_profile(mask, *, num_classes, ignore_index=None, small_region_threshold_ratio=0.02):
    valid_mask = _valid_mask(mask, ignore_index=ignore_index)
    valid_pixels = float(max(1, valid_mask.sum()))

    component_count = np.zeros((num_classes,), dtype=np.float64)
    largest_component_area_ratio = np.zeros((num_classes,), dtype=np.float64)
    mean_component_area_ratio = np.zeros((num_classes,), dtype=np.float64)
    small_region_count = np.zeros((num_classes,), dtype=np.float64)
    small_region_area_ratio = np.zeros((num_classes,), dtype=np.float64)
    component_area_lists = {int(class_id): [] for class_id in range(int(num_classes))}

    for class_id in range(int(num_classes)):
        binary_mask = ((mask == class_id) & valid_mask).astype(np.uint8)
        if int(binary_mask.sum()) <= 0:
            continue
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        if int(num_labels) <= 1:
            continue
        component_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        component_ratios = component_areas / valid_pixels
        component_count[class_id] = float(component_ratios.size)
        mean_component_area_ratio[class_id] = float(component_ratios.mean())
        largest_component_area_ratio[class_id] = float(component_ratios.max())
        component_area_lists[int(class_id)].extend(component_ratios.tolist())

        small_mask = component_ratios <= float(small_region_threshold_ratio)
        if np.any(small_mask):
            small_region_count[class_id] = float(small_mask.sum())
            small_region_area_ratio[class_id] = float(component_ratios[small_mask].sum())

    return {
        "component_count": component_count,
        "largest_component_area_ratio": largest_component_area_ratio,
        "mean_component_area_ratio": mean_component_area_ratio,
        "small_region_count": small_region_count,
        "small_region_area_ratio": small_region_area_ratio,
        "component_area_lists": component_area_lists,
    }


def _quantile_or_none(values, q):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    return float(np.quantile(values, q))


def _mean_or_none(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    return float(values.mean())


def _std_or_none(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    return float(values.std())


def _per_class_stats_from_matrix(matrix):
    stats = {}
    for class_id in range(matrix.shape[1]):
        values = np.asarray(matrix[:, class_id], dtype=np.float64)
        stats[int(class_id)] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95)),
        }
    return stats


def _per_class_component_area_stats(component_area_lists):
    stats = {}
    for class_id, values in component_area_lists.items():
        values = np.asarray(values, dtype=np.float64)
        stats[int(class_id)] = {
            "mean": _mean_or_none(values),
            "std": _std_or_none(values),
            "median": _quantile_or_none(values, 0.5),
            "p95": _quantile_or_none(values, 0.95),
            "count": int(values.size),
        }
    return stats


def _area_ratio_histograms(ratios, bins):
    hist_counts = {}
    hist_freqs = {}
    for class_id in range(ratios.shape[1]):
        counts, _ = np.histogram(ratios[:, class_id], bins=bins)
        hist_counts[int(class_id)] = counts.astype(np.int64)
        hist_freqs[int(class_id)] = counts.astype(np.float64) / float(max(1, ratios.shape[0]))
    return hist_counts, hist_freqs


def _summarize_distribution(masks, *, num_classes, ignore_index=None, small_region_threshold_ratio=0.02):
    if not masks:
        raise ValueError("Mask collection must not be empty.")

    ratios_list = []
    counts_total = np.zeros((num_classes,), dtype=np.float64)
    total_valid_pixels = 0.0
    boundary_lengths = []
    boundary_ratios = []
    unique_class_counts = []

    component_count_rows = []
    largest_component_rows = []
    mean_component_rows = []
    small_region_count_rows = []
    small_region_area_rows = []
    component_area_lists = {int(class_id): [] for class_id in range(int(num_classes))}

    for mask in masks:
        mask = np.asarray(mask, dtype=np.int64)
        ratios, counts, valid_pixels = _class_ratio(mask, num_classes=num_classes, ignore_index=ignore_index)
        ratios_list.append(ratios)
        counts_total += counts
        total_valid_pixels += valid_pixels
        unique_class_counts.append(float((ratios > 0.0).sum()))

        boundary_length, boundary_ratio = _boundary_length(mask, ignore_index=ignore_index)
        boundary_lengths.append(boundary_length)
        boundary_ratios.append(boundary_ratio)

        component_profile = _component_profile(
            mask,
            num_classes=num_classes,
            ignore_index=ignore_index,
            small_region_threshold_ratio=small_region_threshold_ratio,
        )
        component_count_rows.append(component_profile["component_count"])
        largest_component_rows.append(component_profile["largest_component_area_ratio"])
        mean_component_rows.append(component_profile["mean_component_area_ratio"])
        small_region_count_rows.append(component_profile["small_region_count"])
        small_region_area_rows.append(component_profile["small_region_area_ratio"])
        for class_id, values in component_profile["component_area_lists"].items():
            component_area_lists[int(class_id)].extend(values)

    ratios = np.stack(ratios_list, axis=0)
    presences = (ratios > 0.0).astype(np.float64)
    component_count_matrix = np.stack(component_count_rows, axis=0)
    largest_component_matrix = np.stack(largest_component_rows, axis=0)
    mean_component_matrix = np.stack(mean_component_rows, axis=0)
    small_region_count_matrix = np.stack(small_region_count_rows, axis=0)
    small_region_area_matrix = np.stack(small_region_area_rows, axis=0)
    histogram_counts, histogram_freqs = _area_ratio_histograms(ratios, bins=AREA_RATIO_HISTOGRAM_BINS)

    global_class_pixel_ratio = (
        counts_total / float(max(1.0, total_valid_pixels))
        if total_valid_pixels > 0.0
        else np.zeros((num_classes,), dtype=np.float64)
    )

    return {
        "sample_count": int(len(masks)),
        "histogram_bins": AREA_RATIO_HISTOGRAM_BINS.tolist(),
        "global_class_pixel_ratio": global_class_pixel_ratio,
        "class_ratio_stats": _per_class_stats_from_matrix(ratios),
        "class_presence_mean": presences.mean(axis=0),
        "class_area_histogram_counts": {
            int(class_id): counts.tolist() for class_id, counts in histogram_counts.items()
        },
        "class_area_histogram_freq": {
            int(class_id): freqs.round(6).tolist() for class_id, freqs in histogram_freqs.items()
        },
        "boundary_length_mean": float(np.mean(boundary_lengths)),
        "boundary_length_std": float(np.std(boundary_lengths)),
        "boundary_length_ratio_mean": float(np.mean(boundary_ratios)),
        "boundary_length_ratio_std": float(np.std(boundary_ratios)),
        "unique_class_count_mean": float(np.mean(unique_class_counts)),
        "unique_class_count_std": float(np.std(unique_class_counts)),
        "component_count_stats": _per_class_stats_from_matrix(component_count_matrix),
        "largest_component_area_ratio_stats": _per_class_stats_from_matrix(largest_component_matrix),
        "mean_component_area_ratio_stats": _per_class_stats_from_matrix(mean_component_matrix),
        "component_area_ratio_stats": _per_class_component_area_stats(component_area_lists),
        "small_region_count_stats": _per_class_stats_from_matrix(small_region_count_matrix),
        "small_region_area_ratio_stats": _per_class_stats_from_matrix(small_region_area_matrix),
        "small_region_sample_frequency": (small_region_count_matrix > 0.0).mean(axis=0),
        "small_region_threshold_ratio": float(small_region_threshold_ratio),
    }


def _mask_miou(mask_a, mask_b, num_classes, ignore_index=None):
    valid_mask = _valid_mask(mask_a, ignore_index=ignore_index) & _valid_mask(mask_b, ignore_index=ignore_index)
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


def _get_per_class_vector(stats_map, field_name, num_classes):
    values = np.zeros((num_classes,), dtype=np.float64)
    for class_id in range(int(num_classes)):
        value = stats_map[int(class_id)][field_name]
        values[class_id] = 0.0 if value is None else float(value)
    return values


def _compare_distribution(generated_stats, reference_stats, *, num_classes):
    generated_hist = np.asarray(
        [generated_stats["class_area_histogram_freq"][int(class_id)] for class_id in range(int(num_classes))],
        dtype=np.float64,
    )
    reference_hist = np.asarray(
        [reference_stats["class_area_histogram_freq"][int(class_id)] for class_id in range(int(num_classes))],
        dtype=np.float64,
    )
    histogram_l1_per_class = np.abs(generated_hist - reference_hist).sum(axis=1)

    generated_component_count = _get_per_class_vector(
        generated_stats["component_count_stats"],
        "mean",
        num_classes=num_classes,
    )
    reference_component_count = _get_per_class_vector(
        reference_stats["component_count_stats"],
        "mean",
        num_classes=num_classes,
    )
    generated_component_area = _get_per_class_vector(
        generated_stats["component_area_ratio_stats"],
        "mean",
        num_classes=num_classes,
    )
    reference_component_area = _get_per_class_vector(
        reference_stats["component_area_ratio_stats"],
        "mean",
        num_classes=num_classes,
    )
    generated_small_freq = np.asarray(generated_stats["small_region_sample_frequency"], dtype=np.float64)
    reference_small_freq = np.asarray(reference_stats["small_region_sample_frequency"], dtype=np.float64)

    return {
        "global_class_pixel_ratio_l1": float(
            np.abs(
                np.asarray(generated_stats["global_class_pixel_ratio"], dtype=np.float64)
                - np.asarray(reference_stats["global_class_pixel_ratio"], dtype=np.float64)
            ).sum()
        ),
        "class_area_histogram_l1_mean": float(histogram_l1_per_class.mean()),
        "class_area_histogram_l1_max": float(histogram_l1_per_class.max()),
        "histogram_l1_per_class": histogram_l1_per_class,
        "component_count_l1_mean": float(np.abs(generated_component_count - reference_component_count).mean()),
        "component_area_ratio_l1_mean": float(np.abs(generated_component_area - reference_component_area).mean()),
        "boundary_length_ratio_gap": abs(
            float(generated_stats["boundary_length_ratio_mean"])
            - float(reference_stats["boundary_length_ratio_mean"])
        ),
        "small_region_frequency_l1_mean": float(np.abs(generated_small_freq - reference_small_freq).mean()),
        "unique_class_count_gap": abs(
            float(generated_stats["unique_class_count_mean"]) - float(reference_stats["unique_class_count_mean"])
        ),
    }


def _to_json_ready(value):
    if isinstance(value, dict):
        return {str(key): _to_json_ready(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_to_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _write_markdown_report(path, summary):
    lines = [
        "# Mask Prior Evaluation Protocol",
        "",
        f"- task: `{summary['task']}`",
        f"- config: `{summary['config']}`",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- monitor: `{summary['monitor']}`",
        f"- reference source: `{summary['reference_source']}`",
        f"- small-region threshold ratio: `{summary['small_region_threshold_ratio']}`",
        "",
        "Mask-only evaluation is distributional. It compares generated masks against the real split on area, topology, boundary, and small-region statistics. `nearest_real_miou_mean` remains a real-bank plausibility sanity metric, not a paired target metric.",
        "",
        "| NFE | nearest-real mIoU | pairwise fake mIoU | global class ratio L1 | area hist L1 | component count gap | boundary gap | small-region freq gap |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
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
                    f"{float(result['global_class_pixel_ratio_l1']):.4f}",
                    f"{float(result['class_area_histogram_l1_mean']):.4f}",
                    f"{float(result['component_count_l1_mean']):.4f}",
                    f"{float(result['boundary_length_ratio_gap']):.4f}",
                    f"{float(result['small_region_frequency_l1_mean']):.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Lower is better for all gap metrics.",
            "- Higher `nearest_real_miou_mean` is better.",
            "- Very high `pairwise_fake_miou_mean` can indicate reduced diversity or mode collapse.",
            "- `summary.json` contains the full per-class histograms and connected-component statistics for both the real bank and every generated NFE sweep.",
        ]
    )
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
        small_region_threshold_ratio=args.small_region_threshold_ratio,
    )
    results = []
    for nfe, nfe_dir in nfe_dirs:
        generated_masks, stems = _load_generated_masks(nfe_dir, n_samples=args.n_samples)
        generated_stats = _summarize_distribution(
            generated_masks,
            num_classes=num_classes,
            ignore_index=ignore_index,
            small_region_threshold_ratio=args.small_region_threshold_ratio,
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
        distribution_gap = _compare_distribution(
            generated_stats,
            reference_stats,
            num_classes=num_classes,
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
                "global_class_pixel_ratio_l1": distribution_gap["global_class_pixel_ratio_l1"],
                "class_area_histogram_l1_mean": distribution_gap["class_area_histogram_l1_mean"],
                "class_area_histogram_l1_max": distribution_gap["class_area_histogram_l1_max"],
                "component_count_l1_mean": distribution_gap["component_count_l1_mean"],
                "component_area_ratio_l1_mean": distribution_gap["component_area_ratio_l1_mean"],
                "boundary_length_generated_mean": float(generated_stats["boundary_length_mean"]),
                "boundary_length_reference_mean": float(reference_stats["boundary_length_mean"]),
                "boundary_length_ratio_generated_mean": float(generated_stats["boundary_length_ratio_mean"]),
                "boundary_length_ratio_reference_mean": float(reference_stats["boundary_length_ratio_mean"]),
                "boundary_length_ratio_gap": distribution_gap["boundary_length_ratio_gap"],
                "small_region_frequency_l1_mean": distribution_gap["small_region_frequency_l1_mean"],
                "unique_class_count_generated_mean": float(generated_stats["unique_class_count_mean"]),
                "unique_class_count_reference_mean": float(reference_stats["unique_class_count_mean"]),
                "unique_class_count_gap": distribution_gap["unique_class_count_gap"],
                "generated_global_class_pixel_ratio_json": json.dumps(
                    _to_json_ready(np.asarray(generated_stats["global_class_pixel_ratio"]).round(6)),
                    ensure_ascii=True,
                ),
                "reference_global_class_pixel_ratio_json": json.dumps(
                    _to_json_ready(np.asarray(reference_stats["global_class_pixel_ratio"]).round(6)),
                    ensure_ascii=True,
                ),
                "generated_class_ratio_stats_json": json.dumps(
                    _to_json_ready(generated_stats["class_ratio_stats"]),
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                "reference_class_ratio_stats_json": json.dumps(
                    _to_json_ready(reference_stats["class_ratio_stats"]),
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                "generated_component_count_stats_json": json.dumps(
                    _to_json_ready(generated_stats["component_count_stats"]),
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                "reference_component_count_stats_json": json.dumps(
                    _to_json_ready(reference_stats["component_count_stats"]),
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                "generated_component_area_ratio_stats_json": json.dumps(
                    _to_json_ready(generated_stats["component_area_ratio_stats"]),
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                "reference_component_area_ratio_stats_json": json.dumps(
                    _to_json_ready(reference_stats["component_area_ratio_stats"]),
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                "generated_small_region_frequency_json": json.dumps(
                    _to_json_ready(np.asarray(generated_stats["small_region_sample_frequency"]).round(6)),
                    ensure_ascii=True,
                ),
                "reference_small_region_frequency_json": json.dumps(
                    _to_json_ready(np.asarray(reference_stats["small_region_sample_frequency"]).round(6)),
                    ensure_ascii=True,
                ),
                "generated_class_area_histogram_json": json.dumps(
                    _to_json_ready(generated_stats["class_area_histogram_freq"]),
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                "reference_class_area_histogram_json": json.dumps(
                    _to_json_ready(reference_stats["class_area_histogram_freq"]),
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                "metrics_note": (
                    "Mask-only evaluation is distributional. Global class-area, connected-component, boundary, "
                    "and small-region statistics are the main protocol. nearest_real_miou_mean remains a real-bank "
                    "plausibility sanity metric; pairwise_fake_miou_mean is a diversity-collapse sanity metric."
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
        "primary_metric": "mask_distribution_gap_bundle",
        "reference_source": reference_source if args.mask_dir is None else str(args.mask_dir.resolve()),
        "label_spec": str(args.label_spec.resolve()),
        "num_classes": int(num_classes),
        "small_region_threshold_ratio": float(args.small_region_threshold_ratio),
        "primary_readout": [
            "global_class_pixel_ratio_l1",
            "class_area_histogram_l1_mean",
            "component_count_l1_mean",
            "boundary_length_ratio_gap",
            "small_region_frequency_l1_mean",
            "nearest_real_miou_mean",
        ],
        "protocol_notes": {
            "task_type": "This route models p(semantic_mask) only.",
            "paired_target_note": "Unconditional mask generation has no paired target mask; evaluation compares generated masks against the real-mask distribution and a real-mask bank.",
            "nfe_rule": "Use the fixed NFE=8/4/2/1 sweep instead of reporting only NFE=1.",
            "small_region_note": "Small regions are connected components whose area is no larger than small_region_threshold_ratio of valid pixels in a sample.",
            "boundary_note": "Boundary length is measured by vertical and horizontal class-transition counts, normalized by valid pixels.",
            "checkpoint_rule": "Use best checkpoints selected by val/base_error_mean.",
        },
        "reference_distribution": _to_json_ready(reference_stats),
        "results": results,
    }

    summary_json_path = outdir / "summary.json"
    summary_csv_path = outdir / "summary.csv"
    summary_md_path = outdir / "summary.md"
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
                "global_class_pixel_ratio_l1",
                "class_area_histogram_l1_mean",
                "class_area_histogram_l1_max",
                "component_count_l1_mean",
                "component_area_ratio_l1_mean",
                "boundary_length_generated_mean",
                "boundary_length_reference_mean",
                "boundary_length_ratio_generated_mean",
                "boundary_length_ratio_reference_mean",
                "boundary_length_ratio_gap",
                "small_region_frequency_l1_mean",
                "unique_class_count_generated_mean",
                "unique_class_count_reference_mean",
                "unique_class_count_gap",
                "generated_global_class_pixel_ratio_json",
                "reference_global_class_pixel_ratio_json",
                "generated_class_ratio_stats_json",
                "reference_class_ratio_stats_json",
                "generated_component_count_stats_json",
                "reference_component_count_stats_json",
                "generated_component_area_ratio_stats_json",
                "reference_component_area_ratio_stats_json",
                "generated_small_region_frequency_json",
                "reference_small_region_frequency_json",
                "generated_class_area_histogram_json",
                "reference_class_area_histogram_json",
                "metrics_note",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    _write_markdown_report(summary_md_path, summary)
    _write_markdown_report(report_md_path, summary)

    print(f"Saved mask-prior JSON to {summary_json_path}")
    print(f"Saved mask-prior CSV to {summary_csv_path}")
    print(f"Saved mask-prior markdown to {summary_md_path}")
    print(f"Saved compatibility report to {report_md_path}")


if __name__ == "__main__":
    main()

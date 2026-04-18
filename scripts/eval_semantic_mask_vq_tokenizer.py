import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
LDM_ROOT = REPO_ROOT / "third_party" / "latent-diffusion"
TAMING_ROOT = LDM_ROOT / "taming-transformers"

for path in (REPO_ROOT, LDM_ROOT, TAMING_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ldm.util import instantiate_from_config
from latent_meanflow.utils import colorize_mask_index
from scripts.sample_latent_flow import load_config, load_model, validate_ckpt_matches_config


DEFAULT_CONFIG = REPO_ROOT / "configs" / "semantic_mask_vq_tokenizer_main_stable_256.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the project-layer mask-only discrete tokenizer by reconstruction quality. "
            "This is tokenizer reconstruction evaluation, not unconditional generation evaluation."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--small-region-threshold-ratio",
        type=float,
        default=0.02,
        help="Connected components no larger than this ratio of valid pixels count as small regions.",
    )
    parser.add_argument(
        "--worst-k",
        type=int,
        default=32,
        help="Export up to this many lowest-mIoU sample panels into analysis/worst_miou_panel.",
    )
    parser.add_argument(
        "--worst-per-class-k",
        type=int,
        default=8,
        help="Export up to this many worst panels per semantic class into analysis/worst_per_class_panel/.",
    )
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


def _resolve_split_key(config, split_name):
    normalized = str(split_name).lower()
    if normalized == "val":
        normalized = "validation"
    dataset_cfg = config.data.params.get(normalized)
    if dataset_cfg is None:
        raise KeyError(f"Config does not define data.params.{normalized}")
    return normalized


def OmegaConf_select(config, key, default=None):
    from omegaconf import OmegaConf

    return OmegaConf.select(config, key, default=default)


def _resolve_ignore_index(config):
    return OmegaConf_select(config, "model.params.lossconfig.params.ignore_index", default=None)


def _resolve_label_names(dataset_config, num_classes):
    label_spec_path = OmegaConf_select(dataset_config, "params.gray_to_class_id", default=None)
    if label_spec_path is None:
        return {class_id: f"class_{class_id}" for class_id in range(int(num_classes))}
    label_spec_path = Path(str(label_spec_path))
    if not label_spec_path.is_absolute():
        label_spec_path = REPO_ROOT / label_spec_path
    if not label_spec_path.exists():
        return {class_id: f"class_{class_id}" for class_id in range(int(num_classes))}

    from omegaconf import OmegaConf

    spec = OmegaConf.load(label_spec_path)
    class_names = OmegaConf.select(spec, "class_names", default={})
    resolved = {}
    for class_id in range(int(num_classes)):
        value = None
        if isinstance(class_names, dict):
            value = class_names.get(class_id)
            if value is None:
                value = class_names.get(str(class_id))
        resolved[class_id] = str(value) if value is not None else f"class_{class_id}"
    return resolved


def _valid_mask(mask, ignore_index=None):
    valid_mask = np.ones(mask.shape, dtype=bool)
    if ignore_index is not None:
        valid_mask &= mask != int(ignore_index)
    return valid_mask


def _boundary_map(mask, ignore_index=None):
    mask = np.asarray(mask, dtype=np.int64)
    boundary = np.zeros(mask.shape, dtype=bool)
    valid_mask = _valid_mask(mask, ignore_index=ignore_index)

    vertical = (mask[1:, :] != mask[:-1, :]) & valid_mask[1:, :] & valid_mask[:-1, :]
    horizontal = (mask[:, 1:] != mask[:, :-1]) & valid_mask[:, 1:] & valid_mask[:, :-1]
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    return boundary


def _boundary_length_ratio(mask, ignore_index=None):
    boundary = _boundary_map(mask, ignore_index=ignore_index)
    valid_pixels = float(max(1, _valid_mask(mask, ignore_index=ignore_index).sum()))
    return float(boundary.sum()) / valid_pixels


def _boundary_compare_rgb(target_mask, recon_mask, ignore_index=None):
    target_boundary = _boundary_map(target_mask, ignore_index=ignore_index)
    recon_boundary = _boundary_map(recon_mask, ignore_index=ignore_index)
    rgb = np.zeros(target_mask.shape + (3,), dtype=np.uint8)

    overlap = target_boundary & recon_boundary
    target_only = target_boundary & (~recon_boundary)
    recon_only = recon_boundary & (~target_boundary)

    rgb[target_only] = np.asarray([255, 0, 0], dtype=np.uint8)
    rgb[recon_only] = np.asarray([0, 255, 0], dtype=np.uint8)
    rgb[overlap] = np.asarray([255, 255, 0], dtype=np.uint8)
    return rgb


def _mask_miou(pred_mask, target_mask, num_classes, ignore_index=None):
    valid_mask = _valid_mask(pred_mask, ignore_index=ignore_index) & _valid_mask(target_mask, ignore_index=ignore_index)
    if not np.any(valid_mask):
        return 0.0, [0.0 for _ in range(int(num_classes))]

    per_class = []
    for class_id in range(int(num_classes)):
        pred = (pred_mask == class_id) & valid_mask
        target = (target_mask == class_id) & valid_mask
        union = np.logical_or(pred, target).sum()
        if union <= 0:
            per_class.append(None)
            continue
        intersection = np.logical_and(pred, target).sum()
        per_class.append(float(intersection) / float(union))
    valid_ious = [value for value in per_class if value is not None]
    miou = 0.0 if not valid_ious else float(sum(valid_ious) / len(valid_ious))
    return miou, [0.0 if value is None else float(value) for value in per_class]


def _pixel_accuracy(pred_mask, target_mask, ignore_index=None):
    valid_mask = _valid_mask(target_mask, ignore_index=ignore_index)
    valid_count = int(valid_mask.sum())
    if valid_count <= 0:
        return 0.0, 0, 0
    correct = int((pred_mask[valid_mask] == target_mask[valid_mask]).sum())
    return float(correct) / float(valid_count), correct, valid_count


def _small_region_presence(mask, *, num_classes, ignore_index=None, threshold_ratio=0.02):
    valid_mask = _valid_mask(mask, ignore_index=ignore_index)
    valid_pixels = float(max(1, valid_mask.sum()))
    presence = np.zeros((int(num_classes),), dtype=np.float64)

    for class_id in range(int(num_classes)):
        binary_mask = ((mask == class_id) & valid_mask).astype(np.uint8)
        if int(binary_mask.sum()) <= 0:
            continue
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        if int(num_labels) <= 1:
            continue
        component_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        component_ratios = component_areas / valid_pixels
        if np.any(component_ratios <= float(threshold_ratio)):
            presence[class_id] = 1.0
    return presence


def _mask_index_to_raw(mask_index):
    raw_mask = np.asarray(mask_index, dtype=np.int64).copy()
    raw_mask[raw_mask < 0] = 65535
    return raw_mask.astype(np.uint16)


def _compute_confusion_matrix(pred_mask, target_mask, num_classes, ignore_index=None):
    valid_mask = _valid_mask(target_mask, ignore_index=ignore_index)
    if not np.any(valid_mask):
        return np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    pred_valid = np.asarray(pred_mask, dtype=np.int64)[valid_mask]
    target_valid = np.asarray(target_mask, dtype=np.int64)[valid_mask]
    flat_index = target_valid * int(num_classes) + pred_valid
    confusion = np.bincount(flat_index, minlength=int(num_classes) * int(num_classes))
    return confusion.reshape(int(num_classes), int(num_classes)).astype(np.int64, copy=False)


def _present_class_ids(mask, num_classes, ignore_index=None):
    valid_mask = _valid_mask(mask, ignore_index=ignore_index)
    if not np.any(valid_mask):
        return []
    present_ids = np.unique(np.asarray(mask, dtype=np.int64)[valid_mask])
    return [int(class_id) for class_id in present_ids.tolist() if 0 <= int(class_id) < int(num_classes)]


def _summarize_confusion(confusion_matrix, class_names):
    confusion_matrix = np.asarray(confusion_matrix, dtype=np.int64)
    target_counts = confusion_matrix.sum(axis=1).astype(np.float64)
    pred_counts = confusion_matrix.sum(axis=0).astype(np.float64)
    true_positive = np.diag(confusion_matrix).astype(np.float64)
    union = target_counts + pred_counts - true_positive

    per_class_metrics = []
    for class_id in range(confusion_matrix.shape[0]):
        target_count = float(target_counts[class_id])
        pred_count = float(pred_counts[class_id])
        tp = float(true_positive[class_id])
        per_class_metrics.append(
            {
                "class_id": int(class_id),
                "class_name": str(class_names[int(class_id)]),
                "target_pixels": int(target_count),
                "pred_pixels": int(pred_count),
                "precision": 0.0 if pred_count <= 0.0 else float(tp / pred_count),
                "recall": 0.0 if target_count <= 0.0 else float(tp / target_count),
                "iou": 0.0 if float(union[class_id]) <= 0.0 else float(tp / float(union[class_id])),
            }
        )

    row_denominator = target_counts[:, None]
    row_normalized = np.divide(
        confusion_matrix.astype(np.float64),
        np.maximum(row_denominator, 1.0),
        out=np.zeros_like(confusion_matrix, dtype=np.float64),
        where=row_denominator > 0.0,
    )
    return {
        "class_names": [str(class_names[class_id]) for class_id in range(confusion_matrix.shape[0])],
        "matrix": confusion_matrix.astype(np.int64).tolist(),
        "row_normalized": row_normalized.astype(np.float64).tolist(),
        "per_class_metrics": per_class_metrics,
    }


def _copy_ranked_panels(panel_dir, destination_dir, ranked_rows):
    destination_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for rank, row in enumerate(ranked_rows, start=1):
        source_path = panel_dir / f"{row['stem']}.png"
        if not source_path.exists():
            continue
        destination_path = destination_dir / f"{int(rank):03d}_{row['stem']}.png"
        shutil.copy2(source_path, destination_path)
        exported.append(str(destination_path.name))
    return exported


def _safe_filename_component(value):
    safe = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "class"


def _save_reconstruction_triplet(
    *,
    target_mask,
    recon_mask,
    outdir,
    stem,
    num_classes,
    ignore_index=None,
):
    input_raw_dir = outdir / "input_mask_raw"
    input_color_dir = outdir / "input_mask_color"
    recon_raw_dir = outdir / "recon_mask_raw"
    recon_color_dir = outdir / "recon_mask_color"
    panel_dir = outdir / "panel"
    for directory in (input_raw_dir, input_color_dir, recon_raw_dir, recon_color_dir, panel_dir):
        directory.mkdir(parents=True, exist_ok=True)

    target_raw = _mask_index_to_raw(target_mask)
    recon_raw = _mask_index_to_raw(recon_mask)
    target_color = colorize_mask_index(target_mask, num_classes=num_classes, ignore_index=ignore_index)
    recon_color = colorize_mask_index(recon_mask, num_classes=num_classes, ignore_index=ignore_index)
    boundary_rgb = _boundary_compare_rgb(target_mask, recon_mask, ignore_index=ignore_index)
    panel = np.concatenate([target_color, recon_color, boundary_rgb], axis=1)

    Image.fromarray(target_raw).save(input_raw_dir / f"{stem}.png")
    Image.fromarray(target_color).save(input_color_dir / f"{stem}.png")
    Image.fromarray(recon_raw).save(recon_raw_dir / f"{stem}.png")
    Image.fromarray(recon_color).save(recon_color_dir / f"{stem}.png")
    Image.fromarray(panel).save(panel_dir / f"{stem}.png")


def _normalize_paths(mask_paths, batch_size):
    if mask_paths is None:
        return [None] * int(batch_size)
    if isinstance(mask_paths, (list, tuple)):
        return [None if value is None else str(value) for value in mask_paths]
    if isinstance(mask_paths, np.ndarray):
        return [None if value is None else str(value) for value in mask_paths.tolist()]
    return [str(mask_paths)] * int(batch_size)


def _build_sample_stem(global_index, mask_path):
    if mask_path is None:
        return f"{int(global_index):06}"
    stem = Path(str(mask_path)).stem.replace(" ", "_")
    return f"{int(global_index):06}_{stem}"


def _move_batch_tensors_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device=device, non_blocking=True)
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def main():
    args = parse_args()
    config = load_config(args.config)
    split_key = _resolve_split_key(config, args.split)
    ckpt_path = args.ckpt.resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Tokenizer checkpoint not found: {ckpt_path}")
    validate_ckpt_matches_config(args.config, ckpt_path)

    outdir = args.outdir.resolve()
    _prepare_outdir(outdir, overwrite=args.overwrite)
    torch.manual_seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, ckpt_path, device=device)
    ignore_index = _resolve_ignore_index(config)
    num_classes = int(model.num_classes)
    codebook_size = int(model.codebook_size)
    class_names = _resolve_label_names(config.data.params[split_key], num_classes=num_classes)

    dataset = instantiate_from_config(config.data.params[split_key])
    limit = min(len(dataset), int(args.n_samples))
    if limit <= 0:
        raise ValueError("The requested split is empty.")
    subset = torch.utils.data.Subset(dataset, list(range(limit)))
    dataloader = DataLoader(
        subset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
    )

    total_correct = 0
    total_valid = 0
    miou_values = []
    per_class_iou_values = []
    target_boundary_ratios = []
    recon_boundary_ratios = []
    target_small_region_presence = []
    recon_small_region_presence = []
    perplexity_values = []
    sample_rows = []
    sample_analysis_rows = []
    code_counts = np.zeros((codebook_size,), dtype=np.int64)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    global_index = 0
    for batch in dataloader:
        batch_on_device = _move_batch_tensors_to_device(batch, device=model.device)
        outputs = model(batch_on_device, sample_posterior=False)
        target_masks = outputs["mask_index"].detach().cpu().numpy()
        recon_masks = outputs["recon_mask_index"].detach().cpu().numpy()
        codes = outputs["codes"].detach().cpu().numpy()
        perplexity_values.append(float(outputs["quantizer_stats"]["perplexity"].detach().cpu().item()))
        code_counts += np.bincount(codes.reshape(-1), minlength=codebook_size).astype(np.int64, copy=False)
        mask_paths = _normalize_paths(batch.get("mask_path"), batch_size=target_masks.shape[0])

        for local_idx in range(target_masks.shape[0]):
            target_mask = np.asarray(target_masks[local_idx], dtype=np.int64)
            recon_mask = np.asarray(recon_masks[local_idx], dtype=np.int64)
            sample_codes = np.asarray(codes[local_idx], dtype=np.int64)
            stem = _build_sample_stem(global_index, mask_paths[local_idx])

            accuracy, correct, valid_count = _pixel_accuracy(recon_mask, target_mask, ignore_index=ignore_index)
            miou, per_class_ious = _mask_miou(
                recon_mask,
                target_mask,
                num_classes=num_classes,
                ignore_index=ignore_index,
            )
            target_boundary_ratio = _boundary_length_ratio(target_mask, ignore_index=ignore_index)
            recon_boundary_ratio = _boundary_length_ratio(recon_mask, ignore_index=ignore_index)
            target_small_presence = _small_region_presence(
                target_mask,
                num_classes=num_classes,
                ignore_index=ignore_index,
                threshold_ratio=args.small_region_threshold_ratio,
            )
            recon_small_presence = _small_region_presence(
                recon_mask,
                num_classes=num_classes,
                ignore_index=ignore_index,
                threshold_ratio=args.small_region_threshold_ratio,
            )
            small_region_gap = float(np.abs(target_small_presence - recon_small_presence).mean())
            target_class_ids = _present_class_ids(target_mask, num_classes=num_classes, ignore_index=ignore_index)
            present_class_pairs = [
                (class_id, float(per_class_ious[class_id]))
                for class_id in target_class_ids
            ]
            worst_class_id = None
            worst_class_iou = None
            if present_class_pairs:
                worst_class_id, worst_class_iou = min(present_class_pairs, key=lambda item: item[1])

            _save_reconstruction_triplet(
                target_mask=target_mask,
                recon_mask=recon_mask,
                outdir=outdir,
                stem=stem,
                num_classes=num_classes,
                ignore_index=ignore_index,
            )

            total_correct += int(correct)
            total_valid += int(valid_count)
            miou_values.append(float(miou))
            per_class_iou_values.append(np.asarray(per_class_ious, dtype=np.float64))
            target_boundary_ratios.append(float(target_boundary_ratio))
            recon_boundary_ratios.append(float(recon_boundary_ratio))
            target_small_region_presence.append(target_small_presence)
            recon_small_region_presence.append(recon_small_presence)
            confusion_matrix += _compute_confusion_matrix(
                recon_mask,
                target_mask,
                num_classes=num_classes,
                ignore_index=ignore_index,
            )

            sample_rows.append(
                {
                    "index": int(global_index),
                    "stem": stem,
                    "mask_path": None if mask_paths[local_idx] is None else str(mask_paths[local_idx]),
                    "pixel_accuracy": float(accuracy),
                    "miou": float(miou),
                    "boundary_length_ratio_input": float(target_boundary_ratio),
                    "boundary_length_ratio_recon": float(recon_boundary_ratio),
                    "boundary_length_ratio_gap": abs(float(recon_boundary_ratio) - float(target_boundary_ratio)),
                    "small_region_frequency_gap": float(small_region_gap),
                    "valid_pixels": int(valid_count),
                    "code_unique_count": int(np.unique(sample_codes).size),
                    "worst_class_name": None if worst_class_id is None else str(class_names[int(worst_class_id)]),
                    "worst_class_iou": None if worst_class_iou is None else float(worst_class_iou),
                }
            )
            sample_analysis_rows.append(
                {
                    "index": int(global_index),
                    "stem": stem,
                    "mask_path": None if mask_paths[local_idx] is None else str(mask_paths[local_idx]),
                    "miou": float(miou),
                    "pixel_accuracy": float(accuracy),
                    "boundary_length_ratio_gap": abs(float(recon_boundary_ratio) - float(target_boundary_ratio)),
                    "small_region_frequency_gap": float(small_region_gap),
                    "code_unique_count": int(np.unique(sample_codes).size),
                    "target_class_ids": [int(class_id) for class_id in target_class_ids],
                    "per_class_iou": [float(value) for value in per_class_ious],
                }
            )
            global_index += 1

    if per_class_iou_values:
        mean_per_class_iou = np.stack(per_class_iou_values, axis=0).mean(axis=0)
    else:
        mean_per_class_iou = np.zeros((num_classes,), dtype=np.float64)

    if target_small_region_presence:
        target_small_region_presence = np.stack(target_small_region_presence, axis=0)
        recon_small_region_presence = np.stack(recon_small_region_presence, axis=0)
        small_region_frequency_gap = float(
            np.abs(target_small_region_presence.mean(axis=0) - recon_small_region_presence.mean(axis=0)).mean()
        )
        input_small_region_frequency = target_small_region_presence.mean(axis=0)
        recon_small_region_frequency = recon_small_region_presence.mean(axis=0)
    else:
        small_region_frequency_gap = 0.0
        input_small_region_frequency = np.zeros((num_classes,), dtype=np.float64)
        recon_small_region_frequency = np.zeros((num_classes,), dtype=np.float64)

    pixel_accuracy = 0.0 if total_valid <= 0 else float(total_correct) / float(total_valid)
    miou_mean = 0.0 if not miou_values else float(np.mean(miou_values))
    boundary_length_ratio_input_mean = 0.0 if not target_boundary_ratios else float(np.mean(target_boundary_ratios))
    boundary_length_ratio_recon_mean = 0.0 if not recon_boundary_ratios else float(np.mean(recon_boundary_ratios))
    boundary_length_ratio_gap = abs(boundary_length_ratio_recon_mean - boundary_length_ratio_input_mean)
    code_prob = code_counts.astype(np.float64) / max(int(code_counts.sum()), 1)
    nonzero_prob = code_prob[code_prob > 0.0]
    code_perplexity_global = float(np.exp(-(nonzero_prob * np.log(nonzero_prob)).sum())) if nonzero_prob.size else 0.0
    used_code_count = int((code_counts > 0).sum())
    used_code_fraction = float(used_code_count) / float(max(codebook_size, 1))
    dead_code_fraction = 1.0 - used_code_fraction
    confusion_summary = _summarize_confusion(confusion_matrix, class_names=class_names)

    worst_k = max(int(args.worst_k), 0)
    worst_per_class_k = max(int(args.worst_per_class_k), 0)
    worst_by_miou = sorted(sample_analysis_rows, key=lambda row: (float(row["miou"]), int(row["index"])))[:worst_k]
    worst_per_class = {}
    for class_id in range(num_classes):
        class_name = str(class_names[int(class_id)])
        class_rows = [
            {
                "index": int(row["index"]),
                "stem": str(row["stem"]),
                "mask_path": row["mask_path"],
                "miou": float(row["miou"]),
                "pixel_accuracy": float(row["pixel_accuracy"]),
                "class_iou": float(row["per_class_iou"][class_id]),
                "boundary_length_ratio_gap": float(row["boundary_length_ratio_gap"]),
                "small_region_frequency_gap": float(row["small_region_frequency_gap"]),
                "code_unique_count": int(row["code_unique_count"]),
            }
            for row in sample_analysis_rows
            if int(class_id) in row["target_class_ids"]
        ]
        worst_per_class[class_name] = sorted(
            class_rows,
            key=lambda row: (float(row["class_iou"]), float(row["miou"]), int(row["index"])),
        )[:worst_per_class_k]

    analysis_dir = outdir / "analysis"
    confusion_json_path = analysis_dir / "confusion_matrix.json"
    worst_miou_json_path = analysis_dir / "worst_miou.json"
    worst_per_class_json_path = analysis_dir / "worst_per_class.json"
    worst_miou_panel_dir = analysis_dir / "worst_miou_panel"
    worst_per_class_panel_dir = analysis_dir / "worst_per_class_panel"
    exported_worst_miou_panels = _copy_ranked_panels(outdir / "panel", worst_miou_panel_dir, worst_by_miou)
    exported_worst_per_class_panels = {}
    for class_name, rows in worst_per_class.items():
        exported_worst_per_class_panels[class_name] = _copy_ranked_panels(
            outdir / "panel",
            worst_per_class_panel_dir / _safe_filename_component(class_name),
            rows,
        )

    summary = {
        "task": "semantic_mask_vq_tokenizer_reconstruction",
        "evaluation_type": "reconstruction_only_not_unconditional_generation",
        "config": str(args.config.resolve()),
        "checkpoint": str(ckpt_path),
        "split": split_key,
        "seed": int(args.seed),
        "num_samples": int(global_index),
        "batch_size": int(args.batch_size),
        "num_classes": int(num_classes),
        "ignore_index": None if ignore_index is None else int(ignore_index),
        "latent_channels": int(model.latent_channels),
        "latent_spatial_shape": [int(v) for v in model.latent_spatial_shape],
        "token_grid_shape": [int(v) for v in model.token_grid_shape],
        "token_sequence_length": int(np.prod(np.asarray(model.token_grid_shape, dtype=np.int64))),
        "codebook_size": int(codebook_size),
        "class_names": [str(class_names[class_id]) for class_id in range(num_classes)],
        "metrics": {
            "pixel_accuracy": float(pixel_accuracy),
            "miou": float(miou_mean),
            "boundary_length_ratio_input_mean": float(boundary_length_ratio_input_mean),
            "boundary_length_ratio_recon_mean": float(boundary_length_ratio_recon_mean),
            "boundary_length_ratio_gap": float(boundary_length_ratio_gap),
            "small_region_frequency_gap": float(small_region_frequency_gap),
            "mean_per_class_iou": [float(value) for value in mean_per_class_iou.tolist()],
            "aggregate_per_class_iou": [
                float(entry["iou"]) for entry in confusion_summary["per_class_metrics"]
            ],
            "input_small_region_frequency": [float(value) for value in input_small_region_frequency.tolist()],
            "recon_small_region_frequency": [float(value) for value in recon_small_region_frequency.tolist()],
        },
        "per_class_metrics": confusion_summary["per_class_metrics"],
        "codebook": {
            "perplexity_batch_mean": 0.0 if not perplexity_values else float(np.mean(perplexity_values)),
            "perplexity_global": float(code_perplexity_global),
            "used_code_count": int(used_code_count),
            "used_code_fraction": float(used_code_fraction),
            "dead_code_fraction": float(dead_code_fraction),
        },
        "analysis": {
            "worst_k": int(worst_k),
            "worst_per_class_k": int(worst_per_class_k),
            "worst_by_miou": worst_by_miou,
            "worst_per_class": worst_per_class,
            "exported_worst_miou_panels": exported_worst_miou_panels,
            "exported_worst_per_class_panels": exported_worst_per_class_panels,
            "confusion_matrix_json": str(confusion_json_path.resolve()),
            "worst_miou_json": str(worst_miou_json_path.resolve()),
            "worst_per_class_json": str(worst_per_class_json_path.resolve()),
        },
        "notes": {
            "protocol": "This script evaluates semantic_mask reconstruction from the discrete tokenizer code grid.",
            "non_goal": "It does not evaluate unconditional p(semantic_mask) generation or token priors.",
            "panel_columns": ["input_mask_color", "recon_mask_color", "boundary_compare"],
            "boundary_compare_legend": "red=input only, green=recon only, yellow=overlap",
        },
        "samples": sample_rows,
    }

    summary_row = {
        "config": str(args.config.resolve()),
        "checkpoint": str(ckpt_path),
        "split": split_key,
        "num_samples": int(global_index),
        "pixel_accuracy": float(pixel_accuracy),
        "miou": float(miou_mean),
        "boundary_length_ratio_gap": float(boundary_length_ratio_gap),
        "small_region_frequency_gap": float(small_region_frequency_gap),
        "used_code_fraction": float(used_code_fraction),
        "perplexity_global": float(code_perplexity_global),
    }

    summary_json_path = outdir / "summary.json"
    summary_csv_path = outdir / "summary.csv"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    confusion_json_path.write_text(json.dumps(confusion_summary, indent=2), encoding="utf-8")
    worst_miou_json_path.write_text(json.dumps(worst_by_miou, indent=2), encoding="utf-8")
    worst_per_class_json_path.write_text(json.dumps(worst_per_class, indent=2), encoding="utf-8")
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "config",
                "checkpoint",
                "split",
                "num_samples",
                "pixel_accuracy",
                "miou",
                "boundary_length_ratio_gap",
                "small_region_frequency_gap",
                "used_code_fraction",
                "perplexity_global",
            ],
        )
        writer.writeheader()
        writer.writerow(summary_row)

    print(f"Saved semantic_mask VQ tokenizer reconstructions to {outdir}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Confusion JSON: {confusion_json_path}")
    print(f"Worst-mIoU JSON: {worst_miou_json_path}")
    print(f"Worst-per-class JSON: {worst_per_class_json_path}")


if __name__ == "__main__":
    main()

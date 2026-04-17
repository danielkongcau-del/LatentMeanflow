import argparse
import csv
import json
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


DEFAULT_CONFIG = REPO_ROOT / "configs" / "semantic_mask_tokenizer_mid_plus_256.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the project-layer mask-only semantic tokenizer by reconstruction quality. "
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


def _resolve_ignore_index(config):
    return OmegaConf_select(config, "model.params.lossconfig.params.ignore_index", default=None)


def OmegaConf_select(config, key, default=None):
    from omegaconf import OmegaConf

    return OmegaConf.select(config, key, default=default)


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

    dataset = instantiate_from_config(config.data.params[split_key])
    limit = min(len(dataset), int(args.n_samples))
    if limit <= 0:
        raise ValueError("The requested split is empty.")
    indices = list(range(limit))
    subset = torch.utils.data.Subset(dataset, indices)
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
    sample_rows = []

    global_index = 0
    for batch in dataloader:
        outputs = model(batch, sample_posterior=False)
        target_masks = outputs["mask_index"].detach().cpu().numpy()
        recon_masks = outputs["recon_mask_index"].detach().cpu().numpy()
        mask_paths = _normalize_paths(batch.get("mask_path"), batch_size=target_masks.shape[0])

        for local_idx in range(target_masks.shape[0]):
            target_mask = np.asarray(target_masks[local_idx], dtype=np.int64)
            recon_mask = np.asarray(recon_masks[local_idx], dtype=np.int64)
            stem = _build_sample_stem(global_index, mask_paths[local_idx])

            accuracy, correct, valid_count = _pixel_accuracy(
                recon_mask,
                target_mask,
                ignore_index=ignore_index,
            )
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

    summary = {
        "task": "semantic_mask_tokenizer_reconstruction",
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
        "posterior_mode_decode": True,
        "small_region_threshold_ratio": float(args.small_region_threshold_ratio),
        "metrics": {
            "pixel_accuracy": float(pixel_accuracy),
            "miou": float(miou_mean),
            "boundary_length_ratio_input_mean": float(boundary_length_ratio_input_mean),
            "boundary_length_ratio_recon_mean": float(boundary_length_ratio_recon_mean),
            "boundary_length_ratio_gap": float(boundary_length_ratio_gap),
            "small_region_frequency_gap": float(small_region_frequency_gap),
            "mean_per_class_iou": [float(value) for value in mean_per_class_iou.tolist()],
            "input_small_region_frequency": [float(value) for value in input_small_region_frequency.tolist()],
            "recon_small_region_frequency": [float(value) for value in recon_small_region_frequency.tolist()],
        },
        "notes": {
            "protocol": "This script evaluates semantic_mask reconstruction from the tokenizer latent space.",
            "non_goal": "It does not evaluate unconditional p(semantic_mask) generation.",
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
    }

    summary_json_path = outdir / "summary.json"
    summary_csv_path = outdir / "summary.csv"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
            ],
        )
        writer.writeheader()
        writer.writerow(summary_row)

    print(f"Saved semantic_mask tokenizer reconstructions to {outdir}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()

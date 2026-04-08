import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
LDM_ROOT = REPO_ROOT / "third_party" / "latent-diffusion"
TAMING_ROOT = LDM_ROOT / "taming-transformers"

for path in (REPO_ROOT, LDM_ROOT, TAMING_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ldm.util import instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a semantic tokenizer checkpoint and optionally compare it "
            "against a reference tokenizer on the same split."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--reference-config", type=Path, default=None)
    parser.add_argument("--reference-ckpt", type=Path, default=None)
    parser.add_argument("--reference-name", type=str, default=None)
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "outputs" / "semantic_tokenizer_eval",
    )
    return parser.parse_args()


def _move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _load_config(config_path):
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)


def _load_model(config_path, ckpt_path, device):
    config = _load_config(config_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = instantiate_from_config(config.model)
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state["state_dict"] if "state_dict" in state else state
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return config, model


def _resolve_dataset_config(config, split):
    key = "validation" if split == "validation" else "train"
    dataset_config = OmegaConf.select(config, f"data.params.{key}")
    if dataset_config is None:
        raise KeyError(f"Config does not contain data.params.{key}")
    return dataset_config


def _resolve_label_names(dataset_config, num_classes):
    label_spec_path = OmegaConf.select(dataset_config, "params.gray_to_class_id")
    if label_spec_path is None:
        return {class_id: f"class_{class_id}" for class_id in range(num_classes)}
    label_spec_path = Path(label_spec_path)
    if not label_spec_path.is_absolute():
        label_spec_path = REPO_ROOT / label_spec_path
    if not label_spec_path.exists():
        return {class_id: f"class_{class_id}" for class_id in range(num_classes)}
    spec = OmegaConf.load(label_spec_path)
    class_names = OmegaConf.select(spec, "class_names", default={})
    resolved = {}
    for class_id in range(num_classes):
        value = class_names.get(class_id) if isinstance(class_names, dict) else None
        if value is None:
            value = class_names.get(str(class_id)) if isinstance(class_names, dict) else None
        resolved[class_id] = str(value) if value is not None else f"class_{class_id}"
    return resolved


def _build_dataloader(config, split, batch_size_override=None, num_workers_override=None):
    dataset_config = _resolve_dataset_config(config, split)
    dataset = instantiate_from_config(dataset_config)
    default_batch_size = int(OmegaConf.select(config, "data.params.batch_size", default=1))
    default_num_workers = int(OmegaConf.select(config, "data.params.num_workers", default=0))
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size_override or default_batch_size),
        shuffle=False,
        num_workers=int(default_num_workers if num_workers_override is None else num_workers_override),
    )
    return dataset_config, loader


def _safe_lpips():
    try:
        from taming.modules.losses.lpips import LPIPS
    except Exception:
        return None
    model = LPIPS().eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _compute_mask_intersection_union(pred_mask, target_mask, num_classes, ignore_index):
    valid_mask = torch.ones_like(target_mask, dtype=torch.bool)
    if ignore_index is not None:
        valid_mask = target_mask != int(ignore_index)
    pred_mask = pred_mask[valid_mask]
    target_mask = target_mask[valid_mask]

    intersections = torch.zeros(num_classes, dtype=torch.float64)
    unions = torch.zeros(num_classes, dtype=torch.float64)
    for class_id in range(num_classes):
        pred_class = pred_mask == class_id
        target_class = target_mask == class_id
        intersections[class_id] = torch.logical_and(pred_class, target_class).sum().item()
        unions[class_id] = torch.logical_or(pred_class, target_class).sum().item()
    return intersections, unions


def _finalize_moments(total_sum, total_sq_sum, total_count):
    if total_count <= 0:
        return 0.0, 0.0
    mean = total_sum / total_count
    variance = max(total_sq_sum / total_count - mean * mean, 0.0)
    return float(mean), float(math.sqrt(variance))


def _evaluate_one(name, config_path, ckpt_path, split, batch_size_override, num_workers_override, max_batches, device):
    config, model = _load_model(config_path, ckpt_path, device=device)
    dataset_config, loader = _build_dataloader(
        config,
        split=split,
        batch_size_override=batch_size_override,
        num_workers_override=num_workers_override,
    )
    lpips_model = _safe_lpips()
    if lpips_model is not None:
        lpips_model = lpips_model.to(device)

    ignore_index = getattr(model.loss, "ignore_index", None)
    num_classes = int(model.num_classes)
    class_names = _resolve_label_names(dataset_config, num_classes=num_classes)

    total_rgb_l1_sum = 0.0
    total_rgb_l1_count = 0
    total_lpips_sum = 0.0
    total_lpips_count = 0
    total_mask_ce_sum = 0.0
    total_mask_ce_weight = 0
    intersections = torch.zeros(num_classes, dtype=torch.float64)
    unions = torch.zeros(num_classes, dtype=torch.float64)

    latent_sum = 0.0
    latent_sq_sum = 0.0
    latent_count = 0
    latent_channel_sum = torch.zeros(model.latent_channels, dtype=torch.float64)
    latent_channel_sq_sum = torch.zeros(model.latent_channels, dtype=torch.float64)
    latent_channel_count = 0
    latent_norm_sum = 0.0
    latent_norm_sq_sum = 0.0
    latent_norm_count = 0
    image_shape = None
    latent_shape = tuple(int(v) for v in model.latent_spatial_shape)

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break

            batch = _move_to_device(batch, device)
            encoded = model.encode_batch(batch, sample_posterior=False)
            decoded = model.decode_latents(encoded["z"])

            image = encoded["image"]
            mask_index = encoded["mask_index"]
            mask_logits = decoded["mask_logits"]
            pred_mask = decoded["mask_index"]
            z = encoded["z"]

            if image_shape is None:
                image_shape = tuple(int(value) for value in image.shape[-2:])

            rgb_abs = torch.abs(decoded["rgb_recon"] - image)
            total_rgb_l1_sum += float(rgb_abs.sum().item())
            total_rgb_l1_count += int(rgb_abs.numel())

            if lpips_model is not None:
                batch_lpips = lpips_model(decoded["rgb_recon"].contiguous(), image.contiguous()).mean(dim=(1, 2, 3))
                total_lpips_sum += float(batch_lpips.sum().item())
                total_lpips_count += int(batch_lpips.numel())

            valid_mask = torch.ones_like(mask_index, dtype=torch.bool)
            if ignore_index is not None:
                valid_mask = mask_index != int(ignore_index)
            valid_pixel_count = int(valid_mask.sum().item())
            if valid_pixel_count > 0:
                safe_ignore_index = -100 if ignore_index is None else int(ignore_index)
                batch_mask_ce = F.cross_entropy(
                    mask_logits,
                    mask_index,
                    ignore_index=safe_ignore_index,
                    reduction="mean",
                )
                total_mask_ce_sum += float(batch_mask_ce.item()) * valid_pixel_count
                total_mask_ce_weight += valid_pixel_count

                batch_intersections, batch_unions = _compute_mask_intersection_union(
                    pred_mask=pred_mask,
                    target_mask=mask_index,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                )
                intersections += batch_intersections
                unions += batch_unions

            latent_sum += float(z.sum().item())
            latent_sq_sum += float((z * z).sum().item())
            latent_count += int(z.numel())

            channel_values = z.permute(1, 0, 2, 3).reshape(model.latent_channels, -1)
            latent_channel_sum += channel_values.sum(dim=1).double().cpu()
            latent_channel_sq_sum += (channel_values * channel_values).sum(dim=1).double().cpu()
            latent_channel_count += int(channel_values.shape[1])

            sample_norms = z.reshape(z.shape[0], -1).norm(dim=1)
            latent_norm_sum += float(sample_norms.sum().item())
            latent_norm_sq_sum += float((sample_norms * sample_norms).sum().item())
            latent_norm_count += int(sample_norms.numel())

    if image_shape is None:
        raise RuntimeError("No batches were evaluated. Check the dataset split or --max-batches.")

    per_class_iou = {}
    valid_ious = []
    for class_id in range(num_classes):
        union = float(unions[class_id].item())
        iou = None if union <= 0.0 else float(intersections[class_id].item() / union)
        per_class_iou[class_names[class_id]] = iou
        if iou is not None:
            valid_ious.append(iou)

    latent_mean, latent_std = _finalize_moments(latent_sum, latent_sq_sum, latent_count)
    latent_norm_mean, latent_norm_std = _finalize_moments(
        latent_norm_sum,
        latent_norm_sq_sum,
        latent_norm_count,
    )

    channel_stats = {}
    for channel_index in range(model.latent_channels):
        mean, std = _finalize_moments(
            float(latent_channel_sum[channel_index].item()),
            float(latent_channel_sq_sum[channel_index].item()),
            latent_channel_count,
        )
        channel_stats[f"channel_{channel_index}"] = {
            "mean": mean,
            "std": std,
        }

    downsample_factor_h = image_shape[0] // latent_shape[0]
    downsample_factor_w = image_shape[1] // latent_shape[1]

    return {
        "name": name,
        "config": str(Path(config_path).resolve()),
        "checkpoint": str(Path(ckpt_path).resolve()),
        "split": split,
        "num_classes": num_classes,
        "image_shape": list(image_shape),
        "latent_shape": [model.latent_channels, *latent_shape],
        "latent_spatial_shape": list(latent_shape),
        "downsample_factor": [int(downsample_factor_h), int(downsample_factor_w)],
        "rgb_l1": float(total_rgb_l1_sum / max(total_rgb_l1_count, 1)),
        "rgb_lpips": None if total_lpips_count <= 0 else float(total_lpips_sum / total_lpips_count),
        "mask_ce": float(total_mask_ce_sum / max(total_mask_ce_weight, 1)),
        "recon_miou": None if not valid_ious else float(sum(valid_ious) / len(valid_ious)),
        "per_class_iou": per_class_iou,
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "latent_l2_norm_mean": latent_norm_mean,
        "latent_l2_norm_std": latent_norm_std,
        "per_channel_stats": channel_stats,
    }


def _format_value(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _compare_summaries(candidate, reference):
    comparison = {
        "candidate": candidate["name"],
        "reference": reference["name"],
        "rgb_l1_delta": candidate["rgb_l1"] - reference["rgb_l1"],
        "mask_ce_delta": candidate["mask_ce"] - reference["mask_ce"],
        "recon_miou_delta": None,
        "latent_mean_delta": candidate["latent_mean"] - reference["latent_mean"],
        "latent_std_delta": candidate["latent_std"] - reference["latent_std"],
        "latent_l2_norm_mean_delta": candidate["latent_l2_norm_mean"] - reference["latent_l2_norm_mean"],
        "latent_shape_candidate": candidate["latent_shape"],
        "latent_shape_reference": reference["latent_shape"],
    }
    if candidate["rgb_lpips"] is not None and reference["rgb_lpips"] is not None:
        comparison["rgb_lpips_delta"] = candidate["rgb_lpips"] - reference["rgb_lpips"]
    if candidate["recon_miou"] is not None and reference["recon_miou"] is not None:
        comparison["recon_miou_delta"] = candidate["recon_miou"] - reference["recon_miou"]
    return comparison


def _write_summary_markdown(path, summaries, comparison):
    lines = ["# Semantic Tokenizer Evaluation", ""]
    for summary in summaries:
        lines.extend(
            [
                f"## {summary['name']}",
                "",
                f"- config: `{summary['config']}`",
                f"- checkpoint: `{summary['checkpoint']}`",
                f"- split: `{summary['split']}`",
                f"- latent shape: `{summary['latent_shape']}`",
                f"- downsample factor: `{summary['downsample_factor']}`",
                f"- RGB L1: `{_format_value(summary['rgb_l1'])}`",
                f"- RGB LPIPS: `{_format_value(summary['rgb_lpips'])}`",
                f"- mask CE: `{_format_value(summary['mask_ce'])}`",
                f"- recon mIoU: `{_format_value(summary['recon_miou'])}`",
                f"- latent mean/std: `{_format_value(summary['latent_mean'])}` / `{_format_value(summary['latent_std'])}`",
                f"- latent L2 norm mean/std: `{_format_value(summary['latent_l2_norm_mean'])}` / `{_format_value(summary['latent_l2_norm_std'])}`",
                "",
                "| Class | IoU |",
                "| --- | ---: |",
            ]
        )
        for class_name, iou in summary["per_class_iou"].items():
            lines.append(f"| {class_name} | {_format_value(iou)} |")
        lines.extend(["", "| Channel | Mean | Std |", "| --- | ---: | ---: |"])
        for channel_name, stats in summary["per_channel_stats"].items():
            lines.append(
                f"| {channel_name} | {_format_value(stats['mean'])} | {_format_value(stats['std'])} |"
            )
        lines.append("")

    if comparison is not None:
        lines.extend(
            [
                "## Comparison",
                "",
                f"- candidate: `{comparison['candidate']}`",
                f"- reference: `{comparison['reference']}`",
                f"- latent shape candidate: `{comparison['latent_shape_candidate']}`",
                f"- latent shape reference: `{comparison['latent_shape_reference']}`",
                f"- RGB L1 delta: `{_format_value(comparison['rgb_l1_delta'])}`",
                f"- RGB LPIPS delta: `{_format_value(comparison.get('rgb_lpips_delta'))}`",
                f"- mask CE delta: `{_format_value(comparison['mask_ce_delta'])}`",
                f"- recon mIoU delta: `{_format_value(comparison['recon_miou_delta'])}`",
                f"- latent mean delta: `{_format_value(comparison['latent_mean_delta'])}`",
                f"- latent std delta: `{_format_value(comparison['latent_std_delta'])}`",
                f"- latent L2 norm mean delta: `{_format_value(comparison['latent_l2_norm_mean_delta'])}`",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    torch.manual_seed(int(args.seed))
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summaries = []
    primary_name = args.name or args.config.stem
    summaries.append(
        _evaluate_one(
            name=primary_name,
            config_path=args.config,
            ckpt_path=args.ckpt,
            split=args.split,
            batch_size_override=args.batch_size,
            num_workers_override=args.num_workers,
            max_batches=args.max_batches,
            device=device,
        )
    )

    comparison = None
    if args.reference_config is not None or args.reference_ckpt is not None:
        if args.reference_config is None or args.reference_ckpt is None:
            raise ValueError("Both --reference-config and --reference-ckpt are required for comparison mode.")
        reference_name = args.reference_name or args.reference_config.stem
        reference_summary = _evaluate_one(
            name=reference_name,
            config_path=args.reference_config,
            ckpt_path=args.reference_ckpt,
            split=args.split,
            batch_size_override=args.batch_size,
            num_workers_override=args.num_workers,
            max_batches=args.max_batches,
            device=device,
        )
        summaries.append(reference_summary)
        comparison = _compare_summaries(candidate=summaries[0], reference=reference_summary)

    payload = {
        "seed": int(args.seed),
        "device": str(device),
        "split": args.split,
        "max_batches": None if args.max_batches is None else int(args.max_batches),
        "batch_size_override": None if args.batch_size is None else int(args.batch_size),
        "num_workers_override": None if args.num_workers is None else int(args.num_workers),
        "summaries": summaries,
        "comparison": comparison,
    }

    summary_json_path = outdir / "summary.json"
    summary_md_path = outdir / "summary.md"
    summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_summary_markdown(summary_md_path, summaries=summaries, comparison=comparison)

    print(f"Saved tokenizer evaluation JSON to {summary_json_path}")
    print(f"Saved tokenizer evaluation markdown to {summary_md_path}")


if __name__ == "__main__":
    main()

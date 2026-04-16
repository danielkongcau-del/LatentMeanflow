import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from ldm.util import instantiate_from_config

from latent_meanflow.utils import colorize_mask_index
from latent_meanflow.utils.latent_normalization import write_latent_stats_json


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)


def load_model(config_path, ckpt_path, device):
    config = load_config(config_path)
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = instantiate_from_config(config.model)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = state["state_dict"] if "state_dict" in state else state
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return config, model


def extract_tokenizer_config_metadata(config, model):
    model_params = OmegaConf.select(config, "model.params", default={})
    loss_params = OmegaConf.select(config, "model.params.lossconfig.params", default={})
    discriminator_config = OmegaConf.select(config, "model.params.discriminator_config", default=None)
    discriminator_params = {}
    discriminator_target = None
    if discriminator_config is not None:
        discriminator_target = OmegaConf.select(discriminator_config, "target", default=None)
        discriminator_params = dict(OmegaConf.select(discriminator_config, "params", default={}) or {})

    return {
        "model_target": str(OmegaConf.select(config, "model.target", default="")),
        "embed_dim": int(OmegaConf.select(model_params, "embed_dim", default=getattr(model, "embed_dim", 0))),
        "sample_posterior": bool(
            OmegaConf.select(model_params, "sample_posterior", default=getattr(model, "sample_posterior", True))
        ),
        "loss_target": str(OmegaConf.select(config, "model.params.lossconfig.target", default="")),
        "loss_weights": {
            "rgb_l1_weight": float(OmegaConf.select(loss_params, "rgb_l1_weight", default=1.0)),
            "rgb_lpips_weight": float(OmegaConf.select(loss_params, "rgb_lpips_weight", default=0.0)),
            "kl_weight": float(OmegaConf.select(loss_params, "kl_weight", default=0.0)),
            "latent_channel_std_floor_weight": float(
                OmegaConf.select(loss_params, "latent_channel_std_floor_weight", default=0.0)
            ),
            "latent_channel_std_floor": float(
                OmegaConf.select(loss_params, "latent_channel_std_floor", default=0.0)
            ),
            "latent_utilization_threshold": float(
                OmegaConf.select(loss_params, "latent_utilization_threshold", default=0.05)
            ),
        },
        "adversarial": {
            "enabled": bool(getattr(model, "use_discriminator", False)),
            "generator_adversarial_weight": float(
                OmegaConf.select(model_params, "generator_adversarial_weight", default=0.0)
            ),
            "discriminator_start_step": int(
                OmegaConf.select(model_params, "discriminator_start_step", default=0)
            ),
            "discriminator_learning_rate": OmegaConf.select(
                model_params, "discriminator_learning_rate", default=None
            ),
            "discriminator_target": discriminator_target,
            "discriminator_params": discriminator_params,
        },
    }


def resolve_dataset_config(config, split):
    key = "validation" if split == "validation" else "train"
    dataset_config = OmegaConf.select(config, f"data.params.{key}")
    if dataset_config is None:
        raise KeyError(f"Config does not contain data.params.{key}")
    return dataset_config


def build_dataloader(config, split, batch_size_override=None, num_workers_override=None):
    dataset_config = resolve_dataset_config(config, split)
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


def safe_lpips():
    try:
        from taming.modules.losses.lpips import LPIPS
    except Exception:
        return None
    model = LPIPS().eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def finalize_moments(total_sum, total_sq_sum, total_count):
    if total_count <= 0:
        return 0.0, 0.0
    mean = total_sum / total_count
    variance = max(total_sq_sum / total_count - mean * mean, 0.0)
    return float(mean), float(math.sqrt(variance))


def collapse_metrics_from_channel_stds(channel_stds, threshold):
    values = np.asarray(channel_stds, dtype=np.float64)
    if values.size == 0:
        return {
            "collapse_std_threshold": float(threshold),
            "collapsed_channel_count": 0,
            "collapsed_channel_fraction": 0.0,
            "min_channel_std": 0.0,
            "max_channel_std": 0.0,
            "mean_channel_std": 0.0,
            "channel_std_cv": 0.0,
            "channel_std_min_to_max_ratio": 0.0,
            "severity": "unknown",
            "is_channel_collapse_detected": False,
        }

    collapsed = values < float(threshold)
    collapsed_count = int(collapsed.sum())
    mean_std = float(values.mean())
    std_std = float(values.std())
    cv = 0.0 if mean_std <= 1.0e-12 else std_std / mean_std
    min_std = float(values.min())
    max_std = float(values.max())
    min_to_max = 0.0 if max_std <= 1.0e-12 else min_std / max_std
    fraction = float(collapsed_count / max(len(values), 1))

    if collapsed_count == 0:
        severity = "none"
    elif fraction < 0.20:
        severity = "mild"
    elif fraction < 0.50:
        severity = "moderate"
    else:
        severity = "severe"

    return {
        "collapse_std_threshold": float(threshold),
        "collapsed_channel_count": collapsed_count,
        "collapsed_channel_fraction": fraction,
        "min_channel_std": min_std,
        "max_channel_std": max_std,
        "mean_channel_std": mean_std,
        "channel_std_cv": float(cv),
        "channel_std_min_to_max_ratio": float(min_to_max),
        "severity": severity,
        "is_channel_collapse_detected": bool(collapsed_count > 0),
    }


def _clamp01(value):
    return float(max(0.0, min(1.0, value)))


def compute_downstream_readiness(summary):
    rgb_l1 = float(summary["rgb_l1"])
    rgb_lpips = summary["rgb_lpips"]
    if rgb_lpips is None:
        rgb_lpips = 0.35

    collapse = summary["channel_collapse"]
    latent_shape = summary["latent_spatial_shape"]
    image_shape = summary["image_shape"]

    l1_quality = _clamp01((0.10 - rgb_l1) / 0.08)
    lpips_quality = _clamp01((0.35 - float(rgb_lpips)) / 0.30)
    perceptual_quality = 0.40 * l1_quality + 0.60 * lpips_quality

    min_std_score = _clamp01((collapse["min_channel_std"] - 0.05) / 0.35)
    cv_score = _clamp01((0.60 - collapse["channel_std_cv"]) / 0.60)
    collapsed_fraction_penalty = 1.0 - min(1.0, collapse["collapsed_channel_fraction"] * 1.5)
    latent_health = (0.55 * min_std_score + 0.45 * cv_score) * collapsed_fraction_penalty

    image_area = max(int(image_shape[0]) * int(image_shape[1]), 1)
    latent_area = max(int(latent_shape[0]) * int(latent_shape[1]), 1)
    compression_ratio = float(image_area / latent_area)
    compactness_score = _clamp01((compression_ratio - 8.0) / 56.0)

    score = 100.0 * (
        0.55 * perceptual_quality
        + 0.35 * latent_health
        + 0.10 * compactness_score
    )
    return {
        "score": float(score),
        "perceptual_quality": float(perceptual_quality * 100.0),
        "latent_health": float(latent_health * 100.0),
        "compactness": float(compactness_score * 100.0),
        "notes": [
            "Perceptual quality uses RGB LPIPS and RGB L1.",
            "Latent health penalizes low per-channel std and uneven channel std spread.",
            "Compactness gives a small bonus to tighter latents; it never dominates the score.",
        ],
    }


def _normalize_rgb_for_save(image):
    tensor = image.detach().cpu().float()
    if tensor.ndim == 3 and tensor.shape[0] in {1, 3}:
        tensor = tensor.permute(1, 2, 0)
    image_np = tensor.numpy()
    image_np = np.clip((image_np + 1.0) / 2.0, 0.0, 1.0)
    return (image_np * 255.0).astype(np.uint8)


def _error_heatmap(error_map, scale_max):
    normalized = np.clip(error_map / max(float(scale_max), 1.0e-6), 0.0, 1.0)
    red = normalized
    green = np.sqrt(normalized)
    blue = 1.0 - normalized
    heatmap = np.stack([red, green, blue], axis=-1)
    return (heatmap * 255.0).astype(np.uint8)


def _mask_boundary(mask_index):
    boundary = np.zeros(mask_index.shape, dtype=bool)
    boundary[1:, :] |= mask_index[1:, :] != mask_index[:-1, :]
    boundary[:-1, :] |= mask_index[:-1, :] != mask_index[1:, :]
    boundary[:, 1:] |= mask_index[:, 1:] != mask_index[:, :-1]
    boundary[:, :-1] |= mask_index[:, :-1] != mask_index[:, 1:]
    return boundary


def _pool_score_map(array_2d, crop_size, stride):
    tensor = torch.from_numpy(array_2d.astype(np.float32, copy=False))[None, None]
    pooled = F.avg_pool2d(tensor, kernel_size=crop_size, stride=stride)
    return pooled[0, 0].cpu().numpy()


def _pick_box(score_map, crop_size, stride, image_h, image_w, existing_boxes):
    flat_indices = np.argsort(score_map.reshape(-1))[::-1]
    min_separation = max(crop_size // 3, 8)
    for flat_index in flat_indices.tolist():
        score = float(score_map.reshape(-1)[flat_index])
        if not math.isfinite(score):
            continue
        row, col = np.unravel_index(flat_index, score_map.shape)
        top = min(int(row * stride), max(image_h - crop_size, 0))
        left = min(int(col * stride), max(image_w - crop_size, 0))
        center_y = top + crop_size // 2
        center_x = left + crop_size // 2
        too_close = False
        for other in existing_boxes:
            other_center_y = other["top"] + crop_size // 2
            other_center_x = other["left"] + crop_size // 2
            if abs(center_y - other_center_y) < min_separation and abs(center_x - other_center_x) < min_separation:
                too_close = True
                break
        if too_close:
            continue
        return {"top": top, "left": left, "score": score}
    return None


def select_crop_boxes(mask_index, error_map, crop_size):
    image_h, image_w = mask_index.shape
    crop_size = int(max(16, min(crop_size, image_h, image_w)))
    stride = max(crop_size // 4, 8)

    boundary = _mask_boundary(mask_index).astype(np.float32)
    dilated_boundary = F.max_pool2d(
        torch.from_numpy(boundary)[None, None],
        kernel_size=5,
        stride=1,
        padding=2,
    )[0, 0].cpu().numpy()

    score_specs = [
        ("boundary_focus", boundary),
        ("edge_error_focus", error_map * dilated_boundary),
        ("error_focus", error_map),
    ]

    boxes = []
    for kind, score_source in score_specs:
        score_map = _pool_score_map(score_source, crop_size=crop_size, stride=stride)
        picked = _pick_box(
            score_map=score_map,
            crop_size=crop_size,
            stride=stride,
            image_h=image_h,
            image_w=image_w,
            existing_boxes=boxes,
        )
        if picked is None:
            continue
        picked.update({"kind": kind, "size": crop_size})
        boxes.append(picked)
    return boxes


def _draw_boxes(image_np, boxes):
    image = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image)
    colors = {
        "boundary_focus": (255, 0, 0),
        "edge_error_focus": (255, 255, 0),
        "error_focus": (0, 255, 255),
    }
    for box in boxes:
        color = colors.get(box["kind"], (255, 255, 255))
        top = int(box["top"])
        left = int(box["left"])
        size = int(box["size"])
        draw.rectangle((left, top, left + size, top + size), outline=color, width=2)
    return np.asarray(image)


def _sample_stem(batch, local_index, fallback_prefix):
    image_paths = batch.get("image_path")
    if isinstance(image_paths, list) and local_index < len(image_paths):
        try:
            return Path(image_paths[local_index]).stem
        except Exception:
            pass

    metadata = batch.get("metadata")
    if isinstance(metadata, dict):
        sample_stems = metadata.get("sample_stem")
        if isinstance(sample_stems, list) and local_index < len(sample_stems):
            return str(sample_stems[local_index])

    return f"{fallback_prefix}_{local_index:06d}"


def export_visual_diagnostics(
    *,
    batch,
    image_tensor,
    recon_tensor,
    sample_offset,
    outdir,
    crop_size,
    error_heatmap_max,
):
    outdir = Path(outdir)
    records = []
    batch_size = int(image_tensor.shape[0])
    error_tensor = torch.abs(recon_tensor - image_tensor).mean(dim=1)

    for local_index in range(batch_size):
        stem = _sample_stem(batch, local_index=local_index, fallback_prefix="sample")
        sample_name = f"{sample_offset + local_index:06d}_{stem}"
        sample_dir = outdir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        input_np = _normalize_rgb_for_save(image_tensor[local_index])
        recon_np = _normalize_rgb_for_save(recon_tensor[local_index])
        error_map = error_tensor[local_index].detach().cpu().numpy().astype(np.float32, copy=False)
        error_heatmap_np = _error_heatmap(error_map, scale_max=error_heatmap_max)

        Image.fromarray(input_np).save(sample_dir / "input.png")
        Image.fromarray(recon_np).save(sample_dir / "reconstruction.png")
        Image.fromarray(error_heatmap_np).save(sample_dir / "abs_error_heatmap.png")

        panel_tiles = [input_np, recon_np, error_heatmap_np]
        mask_index_batch = batch.get("mask_index")
        crop_boxes = []
        if isinstance(mask_index_batch, torch.Tensor):
            mask_index_np = mask_index_batch[local_index].detach().cpu().numpy().astype(np.int64, copy=False)
            mask_color_np = colorize_mask_index(mask_index_np, num_classes=int(batch["num_classes"][local_index]))
            Image.fromarray(mask_color_np).save(sample_dir / "mask_color.png")
            panel_tiles.append(mask_color_np)
            crop_boxes = select_crop_boxes(mask_index_np, error_map=error_map, crop_size=crop_size)
            overview_tiles = [
                _draw_boxes(input_np, crop_boxes),
                _draw_boxes(recon_np, crop_boxes),
                _draw_boxes(error_heatmap_np, crop_boxes),
                _draw_boxes(mask_color_np, crop_boxes),
            ]
            Image.fromarray(np.concatenate(overview_tiles, axis=1)).save(sample_dir / "overview_with_crops.png")

            for box in crop_boxes:
                top = int(box["top"])
                left = int(box["left"])
                size = int(box["size"])
                bottom = top + size
                right = left + size
                crop_panel = np.concatenate(
                    [
                        input_np[top:bottom, left:right],
                        recon_np[top:bottom, left:right],
                        error_heatmap_np[top:bottom, left:right],
                        mask_color_np[top:bottom, left:right],
                    ],
                    axis=1,
                )
                Image.fromarray(crop_panel).save(sample_dir / f"{box['kind']}_crop_panel.png")

        Image.fromarray(np.concatenate(panel_tiles, axis=1)).save(sample_dir / "panel.png")

        records.append(
            {
                "sample": sample_name,
                "input": str((sample_dir / "input.png").resolve()),
                "reconstruction": str((sample_dir / "reconstruction.png").resolve()),
                "abs_error_heatmap": str((sample_dir / "abs_error_heatmap.png").resolve()),
                "panel": str((sample_dir / "panel.png").resolve()),
                "crops": crop_boxes,
            }
        )
    return records


def evaluate_image_tokenizer(
    *,
    name,
    config_path,
    ckpt_path,
    split,
    batch_size_override,
    num_workers_override,
    max_batches,
    device,
    collapse_std_threshold=0.05,
    export_visuals=False,
    visual_outdir=None,
    visual_samples=0,
    crop_size=64,
    error_heatmap_max=0.20,
):
    config, model = load_model(config_path, ckpt_path, device=device)
    _, loader = build_dataloader(
        config,
        split=split,
        batch_size_override=batch_size_override,
        num_workers_override=num_workers_override,
    )
    lpips_model = safe_lpips()
    if lpips_model is not None:
        lpips_model = lpips_model.to(device)

    total_rgb_l1_sum = 0.0
    total_rgb_l1_count = 0
    total_lpips_sum = 0.0
    total_lpips_count = 0

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
    visual_records = []
    exported_samples = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break

            moved_batch = move_to_device(batch, device)
            encoded = model.encode_batch(moved_batch, sample_posterior=False)
            decoded = model.decode_latents(encoded["z"])

            image = encoded["image"]
            z = encoded["z"]
            recon = decoded["rgb_recon"]

            if image_shape is None:
                image_shape = tuple(int(value) for value in image.shape[-2:])

            rgb_abs = torch.abs(recon - image)
            total_rgb_l1_sum += float(rgb_abs.sum().item())
            total_rgb_l1_count += int(rgb_abs.numel())

            if lpips_model is not None:
                batch_lpips = lpips_model(recon.contiguous(), image.contiguous()).mean(dim=(1, 2, 3))
                total_lpips_sum += float(batch_lpips.sum().item())
                total_lpips_count += int(batch_lpips.numel())

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

            if export_visuals and exported_samples < int(visual_samples):
                take = min(int(visual_samples) - exported_samples, int(image.shape[0]))
                partial_batch = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        partial_batch[key] = value[:take]
                    elif isinstance(value, list):
                        partial_batch[key] = value[:take]
                    elif isinstance(value, dict):
                        partial_batch[key] = {
                            inner_key: inner_value[:take] if isinstance(inner_value, list) else inner_value
                            for inner_key, inner_value in value.items()
                        }
                    else:
                        partial_batch[key] = value
                visual_records.extend(
                    export_visual_diagnostics(
                        batch=partial_batch,
                        image_tensor=image[:take],
                        recon_tensor=recon[:take],
                        sample_offset=exported_samples,
                        outdir=visual_outdir,
                        crop_size=crop_size,
                        error_heatmap_max=error_heatmap_max,
                    )
                )
                exported_samples += take

    if image_shape is None:
        raise RuntimeError("No batches were evaluated. Check the dataset split or --max-batches.")

    latent_mean, latent_std = finalize_moments(latent_sum, latent_sq_sum, latent_count)
    latent_norm_mean, latent_norm_std = finalize_moments(
        latent_norm_sum,
        latent_norm_sq_sum,
        latent_norm_count,
    )

    channel_stats = {}
    channel_stds = []
    for channel_index in range(model.latent_channels):
        mean, std = finalize_moments(
            float(latent_channel_sum[channel_index].item()),
            float(latent_channel_sq_sum[channel_index].item()),
            latent_channel_count,
        )
        channel_stats[f"channel_{channel_index}"] = {
            "mean": mean,
            "std": std,
        }
        channel_stds.append(float(std))

    downsample_factor_h = image_shape[0] // latent_shape[0]
    downsample_factor_w = image_shape[1] // latent_shape[1]
    summary = {
        "name": name,
        "config": str(Path(config_path).resolve()),
        "checkpoint": str(Path(ckpt_path).resolve()),
        "split": split,
        "config_metadata": extract_tokenizer_config_metadata(config, model),
        "image_shape": list(image_shape),
        "latent_shape": [model.latent_channels, *latent_shape],
        "latent_spatial_shape": list(latent_shape),
        "downsample_factor": [int(downsample_factor_h), int(downsample_factor_w)],
        "rgb_l1": float(total_rgb_l1_sum / max(total_rgb_l1_count, 1)),
        "rgb_lpips": None if total_lpips_count <= 0 else float(total_lpips_sum / total_lpips_count),
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "latent_l2_norm_mean": latent_norm_mean,
        "latent_l2_norm_std": latent_norm_std,
        "per_channel_stats": channel_stats,
        "channel_collapse": collapse_metrics_from_channel_stds(
            channel_stds=channel_stds,
            threshold=collapse_std_threshold,
        ),
        "visual_diagnostics": {
            "exported_samples": int(exported_samples),
            "visual_root": None if visual_outdir is None else str(Path(visual_outdir).resolve()),
            "samples": visual_records,
        },
    }
    summary["downstream_readiness"] = compute_downstream_readiness(summary)
    return summary


def compare_summaries(candidate, reference):
    comparison = {
        "candidate": candidate["name"],
        "reference": reference["name"],
        "rgb_l1_delta": candidate["rgb_l1"] - reference["rgb_l1"],
        "latent_mean_delta": candidate["latent_mean"] - reference["latent_mean"],
        "latent_std_delta": candidate["latent_std"] - reference["latent_std"],
        "latent_l2_norm_mean_delta": candidate["latent_l2_norm_mean"] - reference["latent_l2_norm_mean"],
        "readiness_delta": candidate["downstream_readiness"]["score"] - reference["downstream_readiness"]["score"],
        "latent_shape_candidate": candidate["latent_shape"],
        "latent_shape_reference": reference["latent_shape"],
    }
    if candidate["rgb_lpips"] is not None and reference["rgb_lpips"] is not None:
        comparison["rgb_lpips_delta"] = candidate["rgb_lpips"] - reference["rgb_lpips"]
    return comparison


def format_value(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_eval_markdown(path, summaries, comparison=None):
    lines = ["# Image Tokenizer Evaluation", ""]
    for summary in summaries:
        collapse = summary["channel_collapse"]
        readiness = summary["downstream_readiness"]
        lines.extend(
            [
                f"## {summary['name']}",
                "",
                f"- config: `{summary['config']}`",
                f"- checkpoint: `{summary['checkpoint']}`",
                f"- split: `{summary['split']}`",
                f"- latent shape: `{summary['latent_shape']}`",
                f"- downsample factor: `{summary['downsample_factor']}`",
                f"- adversarial enabled: `{summary['config_metadata']['adversarial']['enabled']}`",
                f"- generator adversarial weight: `{format_value(summary['config_metadata']['adversarial']['generator_adversarial_weight'])}`",
                f"- RGB LPIPS weight: `{format_value(summary['config_metadata']['loss_weights']['rgb_lpips_weight'])}`",
                f"- latent std floor weight: `{format_value(summary['config_metadata']['loss_weights']['latent_channel_std_floor_weight'])}`",
                f"- latent std floor: `{format_value(summary['config_metadata']['loss_weights']['latent_channel_std_floor'])}`",
                f"- RGB L1: `{format_value(summary['rgb_l1'])}`",
                f"- RGB LPIPS: `{format_value(summary['rgb_lpips'])}`",
                f"- latent mean/std: `{format_value(summary['latent_mean'])}` / `{format_value(summary['latent_std'])}`",
                f"- latent L2 norm mean/std: `{format_value(summary['latent_l2_norm_mean'])}` / `{format_value(summary['latent_l2_norm_std'])}`",
                f"- collapse threshold: `{format_value(collapse['collapse_std_threshold'])}`",
                f"- collapsed channels: `{collapse['collapsed_channel_count']}` / `{len(summary['per_channel_stats'])}`",
                f"- min/max per-channel std: `{format_value(collapse['min_channel_std'])}` / `{format_value(collapse['max_channel_std'])}`",
                f"- channel-std CV: `{format_value(collapse['channel_std_cv'])}`",
                f"- readiness score: `{format_value(readiness['score'])}`",
                "",
                "| Channel | Mean | Std |",
                "| --- | ---: | ---: |",
            ]
        )
        for channel_name, stats in summary["per_channel_stats"].items():
            lines.append(f"| {channel_name} | {format_value(stats['mean'])} | {format_value(stats['std'])} |")
        if summary["visual_diagnostics"]["exported_samples"] > 0:
            lines.extend(
                [
                    "",
                    f"- visual diagnostics: `{summary['visual_diagnostics']['visual_root']}`",
                ]
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
                f"- RGB L1 delta: `{format_value(comparison['rgb_l1_delta'])}`",
                f"- RGB LPIPS delta: `{format_value(comparison.get('rgb_lpips_delta'))}`",
                f"- latent mean delta: `{format_value(comparison['latent_mean_delta'])}`",
                f"- latent std delta: `{format_value(comparison['latent_std_delta'])}`",
                f"- latent L2 norm mean delta: `{format_value(comparison['latent_l2_norm_mean_delta'])}`",
                f"- readiness delta: `{format_value(comparison['readiness_delta'])}`",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def flatten_summary(summary):
    collapse = summary["channel_collapse"]
    readiness = summary["downstream_readiness"]
    row = {
        "name": summary["name"],
        "config": summary["config"],
        "checkpoint": summary["checkpoint"],
        "split": summary["split"],
        "adversarial_enabled": summary["config_metadata"]["adversarial"]["enabled"],
        "generator_adversarial_weight": summary["config_metadata"]["adversarial"]["generator_adversarial_weight"],
        "rgb_lpips_weight": summary["config_metadata"]["loss_weights"]["rgb_lpips_weight"],
        "latent_channel_std_floor_weight": summary["config_metadata"]["loss_weights"]["latent_channel_std_floor_weight"],
        "latent_channel_std_floor": summary["config_metadata"]["loss_weights"]["latent_channel_std_floor"],
        "image_shape": json.dumps(summary["image_shape"]),
        "latent_shape": json.dumps(summary["latent_shape"]),
        "downsample_factor": json.dumps(summary["downsample_factor"]),
        "rgb_l1": summary["rgb_l1"],
        "rgb_lpips": summary["rgb_lpips"],
        "latent_mean": summary["latent_mean"],
        "latent_std": summary["latent_std"],
        "latent_l2_norm_mean": summary["latent_l2_norm_mean"],
        "latent_l2_norm_std": summary["latent_l2_norm_std"],
        "collapse_std_threshold": collapse["collapse_std_threshold"],
        "collapsed_channel_count": collapse["collapsed_channel_count"],
        "collapsed_channel_fraction": collapse["collapsed_channel_fraction"],
        "min_channel_std": collapse["min_channel_std"],
        "max_channel_std": collapse["max_channel_std"],
        "mean_channel_std": collapse["mean_channel_std"],
        "channel_std_cv": collapse["channel_std_cv"],
        "channel_std_min_to_max_ratio": collapse["channel_std_min_to_max_ratio"],
        "collapse_severity": collapse["severity"],
        "is_channel_collapse_detected": collapse["is_channel_collapse_detected"],
        "readiness_score": readiness["score"],
        "perceptual_quality_score": readiness["perceptual_quality"],
        "latent_health_score": readiness["latent_health"],
        "compactness_score": readiness["compactness"],
    }
    for channel_name, stats in summary["per_channel_stats"].items():
        row[f"{channel_name}_mean"] = stats["mean"]
        row[f"{channel_name}_std"] = stats["std"]
    return row


def write_summary_csv(path, summaries):
    rows = [flatten_summary(summary) for summary in summaries]
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)

    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latent_stats_artifacts(outdir, summaries, *, write_primary_alias=True):
    outdir = Path(outdir)
    stats_dir = outdir / "latent_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for summary in summaries:
        stats_path = stats_dir / f"{summary['name']}.json"
        write_latent_stats_json(stats_path, summary)
        written.append(
            {
                "name": summary["name"],
                "path": str(stats_path.resolve()),
            }
        )

    if write_primary_alias and summaries:
        primary_path = outdir / "latent_stats.json"
        write_latent_stats_json(primary_path, summaries[0])
        written.insert(
            0,
            {
                "name": summaries[0]["name"],
                "path": str(primary_path.resolve()),
            },
        )
    return written

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
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
from latent_meanflow.utils import colorize_mask_index, overlay_color_mask_on_image
from scripts.sample_latent_flow import load_config, load_model, save_pair, validate_ckpt_matches_config


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate paired RGB + semantic-mask generation under a fixed NFE "
            "sweep. This protocol reports RGB realism, teacher-aligned mask "
            "agreement, and pair consistency for few-step latent-flow runs."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--generated-root", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--nfe-values", type=int, nargs="+", default=[8, 4, 2, 1])
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--class-label", type=int, default=None)
    parser.add_argument("--overlay-alpha", type=float, default=0.4)
    parser.add_argument("--two-step-time", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-monitor", type=str, default="val/base_error_mean")
    parser.add_argument("--real-max-samples", type=int, default=None)
    parser.add_argument("--disable-fid-kid", action="store_true")
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
    parser.add_argument("--boundary-tolerance-px", type=int, default=2)
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
            f"Evaluation monitor mismatch: expected '{expected_monitor}', got '{configured_monitor}'. "
            "Use best checkpoints selected by val/base_error_mean for the paired-task protocol."
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


def _resolve_dataset_config(config, split):
    key = "validation" if split == "validation" else "train"
    dataset_config = OmegaConf.select(config, f"data.params.{key}")
    if dataset_config is None:
        raise KeyError(f"Config does not contain data.params.{key}")
    return dataset_config


def _build_real_loader(config, split, batch_size, num_workers=None):
    dataset_config = _resolve_dataset_config(config, split)
    dataset = instantiate_from_config(dataset_config)
    if num_workers is None:
        num_workers = int(OmegaConf.select(config, "data.params.num_workers", default=0))
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers))
    return dataset_config, dataset, loader


def _resolve_label_names(dataset_config, num_classes):
    label_spec_path = OmegaConf.select(dataset_config, "params.gray_to_class_id")
    if label_spec_path is None:
        return {class_id: f"class_{class_id}" for class_id in range(num_classes)}
    label_spec_path = Path(label_spec_path)
    if not label_spec_path.is_absolute():
        label_spec_path = REPO_ROOT / label_spec_path
    if not label_spec_path.exists():
        return {class_id: f"class_{class_id}" for class_id in range(num_classes)}
    label_spec = OmegaConf.load(label_spec_path)
    class_names = OmegaConf.select(label_spec, "class_names", default={})
    resolved = {}
    for class_id in range(num_classes):
        value = None
        if isinstance(class_names, dict):
            value = class_names.get(class_id, class_names.get(str(class_id)))
        resolved[class_id] = str(value) if value is not None else f"class_{class_id}"
    return resolved


def _batch_image_hwc_to_bchw_uint8(image_batch):
    if image_batch.ndim != 4:
        raise ValueError(f"Expected image batch with shape [B, H, W, C], got {tuple(image_batch.shape)}")
    image_batch = image_batch.permute(0, 3, 1, 2).contiguous()
    image_batch = torch.clamp((image_batch + 1.0) / 2.0, 0.0, 1.0)
    return (image_batch * 255.0).round().to(torch.uint8)


def _load_png_image_uint8(path):
    return torch.from_numpy(np.array(Image.open(path).convert("RGB"), dtype=np.uint8)).permute(2, 0, 1)


def _load_png_mask(path):
    return torch.from_numpy(np.array(Image.open(path), dtype=np.int64))


def _resolve_teacher_remap(path):
    if path is None:
        return None
    remap = json.loads(path.read_text(encoding="utf-8"))
    return {int(key): int(value) for key, value in remap.items()}


def _remap_mask(mask, remap):
    if remap is None:
        return mask
    result = mask.clone()
    for src_value, dst_value in remap.items():
        result[mask == int(src_value)] = int(dst_value)
    return result


def _sanitize_teacher_mask(mask, num_classes, remap):
    mask = _remap_mask(mask, remap)
    invalid = (mask < 0) | (mask >= int(num_classes))
    if torch.any(invalid):
        mask = mask.clone()
        mask[invalid] = -1
    return mask


def _boundary_map(mask):
    boundary = torch.zeros_like(mask, dtype=torch.bool)
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[1:, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    boundary[:, 1:] |= mask[:, :-1] != mask[:, 1:]
    boundary &= mask >= 0
    return boundary


def _dilate_boundary(boundary, radius):
    if radius <= 0:
        return boundary
    tensor = boundary.float().unsqueeze(0).unsqueeze(0)
    dilated = F.max_pool2d(tensor, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return dilated.squeeze(0).squeeze(0).bool()


class BoundaryF1Meter:
    def __init__(self, tolerance_px):
        self.tolerance_px = int(tolerance_px)
        self.matched_pred = 0.0
        self.total_pred = 0.0
        self.matched_target = 0.0
        self.total_target = 0.0

    def update(self, pred_mask, target_mask):
        pred_boundary = _boundary_map(pred_mask)
        target_boundary = _boundary_map(target_mask)

        self.total_pred += float(pred_boundary.sum().item())
        self.total_target += float(target_boundary.sum().item())

        target_dilated = _dilate_boundary(target_boundary, self.tolerance_px)
        pred_dilated = _dilate_boundary(pred_boundary, self.tolerance_px)

        self.matched_pred += float((pred_boundary & target_dilated).sum().item())
        self.matched_target += float((target_boundary & pred_dilated).sum().item())

    def compute(self):
        if self.total_pred <= 0.0 and self.total_target <= 0.0:
            return 1.0
        precision = 0.0 if self.total_pred <= 0.0 else self.matched_pred / self.total_pred
        recall = 0.0 if self.total_target <= 0.0 else self.matched_target / self.total_target
        if precision + recall <= 0.0:
            return 0.0
        return float(2.0 * precision * recall / (precision + recall))


class HFSegmentationTeacher:
    def __init__(self, model_id_or_path, device):
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        self.processor = AutoImageProcessor.from_pretrained(model_id_or_path)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_id_or_path)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def predict(self, rgb_uint8_batch):
        images = [image.permute(1, 2, 0).cpu().numpy() for image in rgb_uint8_batch]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits
        logits = F.interpolate(
            logits,
            size=(rgb_uint8_batch.shape[-2], rgb_uint8_batch.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        return logits.argmax(dim=1).cpu()


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


def _save_teacher_pair(teacher_mask, rgb_uint8, outdir, stem, num_classes, overlay_alpha):
    raw_dir = outdir / "teacher_mask_raw"
    color_dir = outdir / "teacher_mask_color"
    overlay_dir = outdir / "teacher_overlay"
    raw_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    teacher_mask_np = teacher_mask.cpu().numpy().astype(np.uint16)
    color_mask_np = colorize_mask_index(teacher_mask_np.astype(np.int64), num_classes=num_classes)
    overlay_np = overlay_color_mask_on_image(
        rgb_uint8.permute(1, 2, 0).cpu().numpy().astype(np.uint8),
        color_mask_np,
        alpha=overlay_alpha,
    )

    Image.fromarray(teacher_mask_np).save(raw_dir / f"{stem}.png")
    Image.fromarray(color_mask_np).save(color_dir / f"{stem}.png")
    Image.fromarray(overlay_np).save(overlay_dir / f"{stem}.png")


def _create_fid_kid_metrics(real_images, device, subset_size_limit):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except Exception as exc:
        raise RuntimeError(
            "FID/KID dependencies are unavailable. Install torchmetrics/torch-fidelity or pass --disable-fid-kid."
        ) from exc

    real_tensor = torch.stack(real_images, dim=0).to(device)
    subset_size = max(1, min(int(subset_size_limit), real_tensor.shape[0]))
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid_metric = KernelInceptionDistance(feature=2048, subset_size=subset_size, normalize=False).to(device)
    fid_metric.update(real_tensor, real=True)
    kid_metric.update(real_tensor, real=True)
    return fid_metric, kid_metric


def _build_real_distribution(config, split, batch_size, real_max_samples):
    dataset_config, dataset, loader = _build_real_loader(config, split, batch_size=batch_size)
    real_images = []
    real_count = 0
    for batch in loader:
        batch_uint8 = _batch_image_hwc_to_bchw_uint8(batch["image"])
        for image in batch_uint8:
            real_images.append(image)
            real_count += 1
            if real_max_samples is not None and real_count >= int(real_max_samples):
                break
        if real_max_samples is not None and real_count >= int(real_max_samples):
            break

    if not real_images:
        raise RuntimeError(f"No real images found for split '{split}'")

    return dataset_config, int(dataset.num_classes), real_images


def _count_pngs(path):
    if not path.exists():
        return 0
    return len(list(path.glob("*.png")))


def _collect_generated_stems(nfe_dir, n_samples):
    image_dir = nfe_dir / "image"
    mask_dir = nfe_dir / "mask_raw"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Missing generated pair directories under {nfe_dir}")
    stems = sorted(path.stem for path in image_dir.glob("*.png"))
    if not stems:
        raise RuntimeError(f"No generated images found under {image_dir}")
    if n_samples is not None:
        stems = stems[: int(n_samples)]
    return stems


def _compute_mask_metrics(fake_masks, teacher_masks, num_classes, class_names, boundary_tolerance_px):
    if len(fake_masks) != len(teacher_masks):
        raise ValueError(
            f"Teacher/generated mask count mismatch: {len(fake_masks)} generated masks vs {len(teacher_masks)} teacher masks."
        )

    intersections = torch.zeros(num_classes, dtype=torch.float64)
    unions = torch.zeros(num_classes, dtype=torch.float64)
    matched_pixels = 0.0
    total_pixels = 0.0
    boundary_meter = BoundaryF1Meter(tolerance_px=boundary_tolerance_px)

    for fake_mask, teacher_mask in zip(fake_masks, teacher_masks):
        valid_mask = teacher_mask >= 0
        matched_pixels += float((fake_mask[valid_mask] == teacher_mask[valid_mask]).sum().item())
        total_pixels += float(valid_mask.sum().item())

        fake_valid = fake_mask[valid_mask]
        teacher_valid = teacher_mask[valid_mask]
        for class_id in range(num_classes):
            fake_class = fake_valid == class_id
            teacher_class = teacher_valid == class_id
            intersections[class_id] += float(torch.logical_and(fake_class, teacher_class).sum().item())
            unions[class_id] += float(torch.logical_or(fake_class, teacher_class).sum().item())

        fake_for_boundary = fake_mask.clone()
        teacher_for_boundary = teacher_mask.clone()
        fake_for_boundary[~valid_mask] = -1
        teacher_for_boundary[~valid_mask] = -1
        boundary_meter.update(fake_for_boundary, teacher_for_boundary)

    per_class_iou = {}
    iou_values = []
    for class_id in range(num_classes):
        union = float(unions[class_id].item())
        iou = None if union <= 0.0 else float(intersections[class_id].item() / union)
        per_class_iou[class_names[class_id]] = iou
        if iou is not None:
            iou_values.append(iou)

    return {
        "mask_miou": None if not iou_values else float(sum(iou_values) / len(iou_values)),
        "mask_per_class_iou": per_class_iou,
        "mask_boundary_f1": float(boundary_meter.compute()),
        "pair_pixel_accuracy": None if total_pixels <= 0.0 else float(matched_pixels / total_pixels),
    }


def _generate_samples(args, config, ckpt_path, outdir, device):
    validate_ckpt_matches_config(args.config, ckpt_path)
    model = load_model(config, ckpt_path, device=device)
    if args.class_label is not None and not getattr(model, "use_class_condition", False):
        raise ValueError("--class-label was provided, but this checkpoint is configured as unconditional.")
    if args.two_step_time is not None and hasattr(model, "sampler") and hasattr(model.sampler, "two_step_time"):
        model.sampler.two_step_time = float(args.two_step_time)

    latent_shape = (model.latent_channels, *model.latent_spatial_shape)
    noise_bank = _make_fixed_noise_bank(args.n_samples, latent_shape=latent_shape, seed=args.seed)

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
            remaining -= batch_size
            index += batch_size
    return int(model.num_classes)


def _evaluate_nfe_dir(
    nfe_dir,
    nfe,
    n_samples,
    num_classes,
    class_names,
    device,
    real_images,
    disable_fid_kid,
    teacher_model,
    teacher_mask_root,
    teacher_remap,
    overlay_alpha,
    boundary_tolerance_px,
):
    stems = _collect_generated_stems(nfe_dir, n_samples=n_samples)

    if disable_fid_kid:
        fid_metric = None
        kid_metric = None
    else:
        fid_metric, kid_metric = _create_fid_kid_metrics(
            real_images=real_images,
            device=device,
            subset_size_limit=min(50, len(stems)),
        )

    fake_masks = []
    teacher_masks = []

    batch_rgb_uint8 = []
    batch_stems = []

    def flush_teacher_batch():
        nonlocal batch_rgb_uint8, batch_stems, teacher_masks
        if not batch_rgb_uint8 or teacher_model is None:
            return
        rgb_tensor = torch.stack(batch_rgb_uint8, dim=0)
        predicted_masks = teacher_model.predict(rgb_tensor.to(device))
        for stem, rgb_uint8, teacher_mask in zip(batch_stems, batch_rgb_uint8, predicted_masks):
            teacher_mask = _sanitize_teacher_mask(teacher_mask, num_classes=num_classes, remap=teacher_remap)
            teacher_masks.append(teacher_mask)
            _save_teacher_pair(
                teacher_mask=teacher_mask,
                rgb_uint8=rgb_uint8,
                outdir=nfe_dir,
                stem=stem,
                num_classes=num_classes,
                overlay_alpha=overlay_alpha,
            )
        batch_rgb_uint8 = []
        batch_stems = []

    if teacher_mask_root is not None:
        teacher_masks = _load_teacher_predictions_from_root(
            root=teacher_mask_root,
            nfe=nfe,
            stems=stems,
            num_classes=num_classes,
            remap=teacher_remap,
        )

    for stem in stems:
        rgb_uint8 = _load_png_image_uint8(nfe_dir / "image" / f"{stem}.png")
        if fid_metric is not None and kid_metric is not None:
            fake_tensor = rgb_uint8.unsqueeze(0).to(device)
            fid_metric.update(fake_tensor, real=False)
            kid_metric.update(fake_tensor, real=False)
        fake_masks.append(_load_png_mask(nfe_dir / "mask_raw" / f"{stem}.png"))
        if teacher_model is not None:
            batch_rgb_uint8.append(rgb_uint8)
            batch_stems.append(stem)
            if len(batch_rgb_uint8) >= 8:
                flush_teacher_batch()

    flush_teacher_batch()

    mask_metrics = _compute_mask_metrics(
        fake_masks=fake_masks,
        teacher_masks=teacher_masks,
        num_classes=num_classes,
        class_names=class_names,
        boundary_tolerance_px=boundary_tolerance_px,
    )

    fid_value = None
    kid_mean = None
    kid_std = None
    if fid_metric is not None and kid_metric is not None:
        fid_value = float(fid_metric.compute().item())
        kid_mean_tensor, kid_std_tensor = kid_metric.compute()
        kid_mean = float(kid_mean_tensor.item())
        kid_std = float(kid_std_tensor.item())

    return {
        "nfe": int(nfe),
        "outdir": str(nfe_dir.resolve()),
        "image_count": _count_pngs(nfe_dir / "image"),
        "mask_raw_count": _count_pngs(nfe_dir / "mask_raw"),
        "mask_color_count": _count_pngs(nfe_dir / "mask_color"),
        "overlay_count": _count_pngs(nfe_dir / "overlay"),
        "teacher_mask_count": _count_pngs(nfe_dir / "teacher_mask_raw"),
        "teacher_mask_color_count": _count_pngs(nfe_dir / "teacher_mask_color"),
        "teacher_overlay_count": _count_pngs(nfe_dir / "teacher_overlay"),
        "fid": fid_value,
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "mask_per_class_iou_json": json.dumps(mask_metrics["mask_per_class_iou"], ensure_ascii=True, sort_keys=True),
        **mask_metrics,
    }


def _write_summary_csv(path, results):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "nfe",
                "outdir",
                "image_count",
                "mask_raw_count",
                "mask_color_count",
                "overlay_count",
                "teacher_mask_count",
                "teacher_mask_color_count",
                "teacher_overlay_count",
                "fid",
                "kid_mean",
                "kid_std",
                "mask_miou",
                "mask_boundary_f1",
                "pair_pixel_accuracy",
                "mask_per_class_iou_json",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})


@torch.no_grad()
def main():
    args = parse_args()
    if (args.teacher_hf_model is None) == (args.teacher_mask_root is None):
        raise ValueError(
            "Pass exactly one teacher source: either --teacher-hf-model or --teacher-mask-root."
        )

    config = load_config(args.config)
    monitor = _check_monitor(config, expected_monitor=args.expected_monitor)
    outdir = args.outdir.resolve()
    _prepare_outdir(outdir, overwrite=args.overwrite)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))

    dataset_config, dataset_num_classes, real_images = _build_real_distribution(
        config=config,
        split=args.split,
        batch_size=args.batch_size,
        real_max_samples=args.real_max_samples,
    )
    class_names = _resolve_label_names(dataset_config, num_classes=dataset_num_classes)

    if args.generated_root is None:
        if args.ckpt is None:
            raise ValueError("--ckpt is required unless --generated-root is provided.")
        ckpt_path = args.ckpt.resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        num_classes = _generate_samples(args, config, ckpt_path, outdir=outdir, device=device)
        generated_root = outdir
    else:
        generated_root = args.generated_root.resolve()
        if not generated_root.exists():
            raise FileNotFoundError(f"Generated root not found: {generated_root}")
        ckpt_path = None if args.ckpt is None else args.ckpt.resolve()
        num_classes = dataset_num_classes

    teacher_remap = _resolve_teacher_remap(args.teacher_remap_json)
    teacher_model = None
    if args.teacher_hf_model is not None:
        teacher_model = HFSegmentationTeacher(args.teacher_hf_model, device=device)
    teacher_mask_root = None if args.teacher_mask_root is None else args.teacher_mask_root.resolve()

    summary = {
        "config": str(args.config.resolve()),
        "checkpoint": None if ckpt_path is None else str(ckpt_path),
        "generated_root": str(generated_root),
        "monitor": monitor,
        "split": args.split,
        "seed": int(args.seed),
        "n_samples": int(args.n_samples),
        "real_image_count": int(len(real_images)),
        "nfe_values": [int(value) for value in args.nfe_values],
        "teacher_hf_model": args.teacher_hf_model,
        "teacher_mask_root": None if teacher_mask_root is None else str(teacher_mask_root),
        "teacher_remap_json": None if args.teacher_remap_json is None else str(args.teacher_remap_json.resolve()),
        "boundary_tolerance_px": int(args.boundary_tolerance_px),
        "protocol_notes": {
            "checkpoint_rule": "Use best checkpoints selected by val/base_error_mean.",
            "mask_metrics": "For unconditional paired generation, mask metrics are teacher-aligned metrics comparing M_hat with S(I_hat), not sample-wise GT metrics.",
            "teacher_training": "Loading a fixed external teacher or precomputed teacher masks is supported. Teacher training inside this repository is not implemented yet.",
            "nfe_rule": "Do not report only NFE=1. Use the fixed NFE=8/4/2/1 sweep.",
        },
        "results": [],
    }

    for nfe in args.nfe_values:
        nfe_dir = generated_root / f"nfe{int(nfe)}"
        result = _evaluate_nfe_dir(
            nfe_dir=nfe_dir,
            nfe=nfe,
            n_samples=args.n_samples,
            num_classes=num_classes,
            class_names=class_names,
            device=device,
            real_images=real_images,
            disable_fid_kid=args.disable_fid_kid,
            teacher_model=teacher_model,
            teacher_mask_root=teacher_mask_root,
            teacher_remap=teacher_remap,
            overlay_alpha=args.overlay_alpha,
            boundary_tolerance_px=args.boundary_tolerance_px,
        )
        summary["results"].append(result)

    summary_json_path = outdir / "summary.json"
    summary_csv_path = outdir / "summary.csv"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_csv(summary_csv_path, summary["results"])

    print(f"Saved paired semantic evaluation JSON to {summary_json_path}")
    print(f"Saved paired semantic evaluation CSV to {summary_csv_path}")


if __name__ == "__main__":
    main()

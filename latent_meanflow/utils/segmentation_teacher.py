import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from latent_meanflow.utils.palette import (
    UNDEFINED_CLASS_ID,
    build_lookup_table,
    colorize_mask_index,
    infer_num_classes,
    overlay_color_mask_on_image,
    resolve_gray_to_class_id,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SEGMENTATION_ROOT = REPO_ROOT / "third_party" / "segmentation"
SUPPORTED_TEACHER_CANDIDATES = ("deeplabv3-resnet", "csnet", "unet")


def ensure_segmentation_vendor_on_path():
    segmentation_root_str = str(SEGMENTATION_ROOT.resolve())
    if segmentation_root_str not in sys.path:
        sys.path.insert(0, segmentation_root_str)


def resolve_label_spec_metadata(label_spec_path):
    label_spec_path = Path(label_spec_path).resolve()
    spec = OmegaConf.to_container(OmegaConf.load(label_spec_path), resolve=True)
    gray_to_class_id, ignore_index = resolve_gray_to_class_id(label_spec_path, ignore_index=None)
    num_classes = int(infer_num_classes(gray_to_class_id, ignore_index=ignore_index))
    class_names_config = spec.get("class_names", {}) if isinstance(spec, dict) else {}
    class_names = {}
    for class_id in range(num_classes):
        value = None
        if isinstance(class_names_config, dict):
            value = class_names_config.get(class_id, class_names_config.get(str(class_id)))
        class_names[int(class_id)] = str(value) if value is not None else f"class_{class_id}"
    return {
        "label_spec_path": str(label_spec_path),
        "gray_to_class_id": {int(key): int(value) for key, value in gray_to_class_id.items()},
        "ignore_index": None if ignore_index is None else int(ignore_index),
        "num_classes": num_classes,
        "class_names": class_names,
        "lookup": build_lookup_table(gray_to_class_id, undefined_value=UNDEFINED_CLASS_ID),
    }


def _resolve_split_dir(dataset_root, split):
    dataset_root = Path(dataset_root).resolve()
    candidate = dataset_root / split
    return candidate if candidate.exists() else dataset_root


def _resolve_image_and_mask_dirs(split_dir):
    image_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        alt_image_dir = split_dir / "image"
        alt_mask_dir = split_dir / "mask"
        if alt_image_dir.is_dir() and alt_mask_dir.is_dir():
            image_dir = alt_image_dir
            mask_dir = alt_mask_dir
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Expected images/masks or image/mask under split directory: {split_dir}"
        )
    return image_dir, mask_dir


def collect_image_mask_pairs(dataset_root, split):
    split_dir = _resolve_split_dir(dataset_root, split)
    image_dir, mask_dir = _resolve_image_and_mask_dirs(split_dir)
    image_files = [path for path in image_dir.iterdir() if path.is_file()]
    image_map = {}
    for path in image_files:
        if path.stem not in image_map:
            image_map[path.stem] = path
    pairs = []
    missing_images = []
    for mask_path in sorted(path for path in mask_dir.iterdir() if path.is_file()):
        image_path = image_map.get(mask_path.stem)
        if image_path is None:
            missing_images.append(mask_path.name)
            continue
        pairs.append({"stem": mask_path.stem, "image_path": image_path, "mask_path": mask_path})
    if missing_images:
        preview = ", ".join(missing_images[:5])
        raise FileNotFoundError(
            f"Missing paired images for {len(missing_images)} masks under {split_dir}. Example(s): {preview}"
        )
    if not pairs:
        raise FileNotFoundError(f"No image/mask pairs found under {split_dir}")
    return pairs


def _safe_rmtree(path):
    path = Path(path).resolve()
    if path == path.anchor:
        raise ValueError(f"Refusing to remove filesystem root: {path}")
    if path.exists():
        shutil.rmtree(path)


def _link_or_copy_file(src, dst, mode):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    resolved_mode = str(mode).lower()
    if resolved_mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    if resolved_mode == "hardlink":
        os.link(src, dst)
        return "hardlink"
    if resolved_mode == "auto":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"
    raise ValueError(f"Unsupported link mode: {mode}")


def decode_mask_to_index(mask_path, lookup, num_classes):
    mask_image = Image.open(mask_path).convert("L")
    mask_raw = np.asarray(mask_image, dtype=np.uint8)
    if int(mask_raw.max()) <= int(num_classes) - 1 and int(mask_raw.min()) >= 0:
        return mask_raw.astype(np.int64, copy=False)
    mask_index = lookup[mask_raw].astype(np.int64, copy=False)
    invalid = mask_index == UNDEFINED_CLASS_ID
    if np.any(invalid):
        unknown_values = sorted(int(value) for value in np.unique(mask_raw[invalid]).tolist())
        raise ValueError(
            f"Mask {mask_path} contains gray values missing from the label spec: {unknown_values}"
        )
    return mask_index


def prepare_segmentation_teacher_dataset(
    *,
    src_root,
    dst_root,
    label_spec_path,
    splits,
    link_mode="auto",
    overwrite=False,
):
    label_metadata = resolve_label_spec_metadata(label_spec_path)
    dst_root = Path(dst_root).resolve()
    if label_metadata["num_classes"] > 255:
        raise ValueError(
            "The vendored segmentation harness expects 8-bit class-index masks. "
            f"Got {label_metadata['num_classes']} classes."
        )
    if dst_root.exists() and overwrite:
        _safe_rmtree(dst_root)
    elif dst_root.exists() and any(dst_root.iterdir()):
        raise FileExistsError(
            f"Destination directory already exists and is not empty: {dst_root}. "
            "Pass --overwrite to rebuild the remapped dataset view."
        )
    dst_root.mkdir(parents=True, exist_ok=True)

    split_summaries = []
    image_materialization = None
    for split in splits:
        pairs = collect_image_mask_pairs(src_root, split)
        split_dir = dst_root / split
        image_dir = split_dir / "images"
        mask_dir = split_dir / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for pair in pairs:
            materialization = _link_or_copy_file(
                pair["image_path"],
                image_dir / pair["image_path"].name,
                mode=link_mode,
            )
            if image_materialization is None:
                image_materialization = materialization
            mask_index = decode_mask_to_index(
                pair["mask_path"],
                lookup=label_metadata["lookup"],
                num_classes=label_metadata["num_classes"],
            )
            Image.fromarray(mask_index.astype(np.uint8), mode="L").save(mask_dir / f"{pair['stem']}.png")
        split_summaries.append(
            {
                "split": str(split),
                "pair_count": int(len(pairs)),
                "image_dir": str(image_dir),
                "mask_dir": str(mask_dir),
            }
        )

    manifest = {
        "source_root": str(Path(src_root).resolve()),
        "prepared_root": str(dst_root),
        "label_spec_path": label_metadata["label_spec_path"],
        "gray_to_class_id": label_metadata["gray_to_class_id"],
        "ignore_index": label_metadata["ignore_index"],
        "num_classes": label_metadata["num_classes"],
        "class_names": label_metadata["class_names"],
        "splits": split_summaries,
        "image_materialization": image_materialization,
    }
    (dst_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return manifest


def resolve_teacher_checkpoint(run_dir, checkpoint_path=None):
    run_dir = Path(run_dir).resolve()
    train_args_path = run_dir / "train_args.json"
    if not train_args_path.exists():
        raise FileNotFoundError(f"Missing train_args.json under teacher run directory: {run_dir}")
    train_args = json.loads(train_args_path.read_text(encoding="utf-8"))
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path, train_args
    net_name = str(train_args["net_name"])
    best_path = run_dir / f"{net_name}_best.pt"
    if not best_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {best_path}. The bakeoff workflow expects the vendored trainer's "
            "explicit val-mIoU winner checkpoint."
        )
    return best_path.resolve(), train_args


def _strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if not all(str(key).startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {str(key)[7:]: value for key, value in state_dict.items()}


def load_teacher_model(*, run_dir, device, checkpoint_path=None):
    checkpoint_path, train_args = resolve_teacher_checkpoint(run_dir, checkpoint_path=checkpoint_path)
    ensure_segmentation_vendor_on_path()
    from choices import choose_net

    net_name = str(train_args["net_name"])
    out_channels = int(train_args["out_channels"])
    height = int(train_args["height"])
    width = int(train_args["width"])
    img_size = height if height == width else height
    model = choose_net(net_name, out_channels=out_channels, img_size=img_size)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint payload for {checkpoint_path}")
    state_dict = _strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    metadata = {
        "run_dir": str(Path(run_dir).resolve()),
        "checkpoint_path": str(checkpoint_path),
        "net_name": net_name,
        "out_channels": out_channels,
        "height": height,
        "width": width,
        "save_dir": train_args.get("save_dir"),
        "epoch_budget": None if train_args.get("epoch") is None else int(train_args["epoch"]),
        "batch_size": None if train_args.get("batch_size") is None else int(train_args["batch_size"]),
        "train_args": train_args,
    }
    return model, metadata


def _resize_mask_index(mask_index, size_hw):
    width = int(size_hw[1])
    height = int(size_hw[0])
    resized = Image.fromarray(mask_index.astype(np.uint8), mode="L").resize((width, height), resample=Image.NEAREST)
    return np.asarray(resized, dtype=np.int64)


def load_ground_truth_mask(mask_path, lookup, target_size_hw):
    max_class_id = int(np.max(lookup[lookup != UNDEFINED_CLASS_ID]))
    mask_index = decode_mask_to_index(mask_path, lookup, num_classes=max_class_id + 1)
    if target_size_hw is not None:
        mask_index = _resize_mask_index(mask_index, target_size_hw)
    return torch.from_numpy(mask_index.astype(np.int64, copy=False))


def _load_image_uint8(path):
    return Image.open(path).convert("RGB")


def _preprocess_pil_image(image, size_hw):
    tensor = transforms.Compose(
        [
            transforms.Resize((int(size_hw[0]), int(size_hw[1]))),
            transforms.ToTensor(),
        ]
    )(image)
    return tensor


@torch.no_grad()
def predict_masks_for_paths(*, image_paths, model, input_size_hw, device, batch_size, output_size_mode="input"):
    predictions = []
    batch_tensors = []
    batch_output_sizes = []
    for image_path in image_paths:
        image = _load_image_uint8(image_path)
        original_size_hw = (image.height, image.width)
        batch_tensors.append(_preprocess_pil_image(image, input_size_hw))
        if output_size_mode == "input":
            batch_output_sizes.append(tuple(int(value) for value in input_size_hw))
        elif output_size_mode == "original":
            batch_output_sizes.append(original_size_hw)
        else:
            raise ValueError(f"Unsupported output_size_mode: {output_size_mode}")
        if len(batch_tensors) >= int(batch_size):
            predictions.extend(
                _forward_teacher_batch(
                    model=model,
                    device=device,
                    batch_tensors=batch_tensors,
                    batch_output_sizes=batch_output_sizes,
                )
            )
            batch_tensors = []
            batch_output_sizes = []
    if batch_tensors:
        predictions.extend(
            _forward_teacher_batch(
                model=model,
                device=device,
                batch_tensors=batch_tensors,
                batch_output_sizes=batch_output_sizes,
            )
        )
    return predictions


def _forward_teacher_batch(*, model, device, batch_tensors, batch_output_sizes):
    batch = torch.stack(batch_tensors, dim=0).to(device)
    logits = model(batch)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
    outputs = []
    for pred_mask, output_size_hw in zip(pred, batch_output_sizes):
        if tuple(pred_mask.shape[-2:]) != tuple(int(value) for value in output_size_hw):
            pred_mask = _resize_mask_index(pred_mask.astype(np.int64), output_size_hw).astype(np.uint8)
        outputs.append(torch.from_numpy(pred_mask.astype(np.int64, copy=False)))
    return outputs


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


def infer_focus_class_ids(class_names):
    focus_class_ids = []
    keywords = ("road", "ditch", "canal", "channel", "boundary", "edge", "waterway")
    for class_id, class_name in class_names.items():
        name_lower = str(class_name).lower()
        if any(keyword in name_lower for keyword in keywords):
            focus_class_ids.append(int(class_id))
    return sorted(set(focus_class_ids))


def compute_segmentation_metrics(
    *,
    pred_masks,
    target_masks,
    class_names,
    boundary_tolerance_px,
    small_class_threshold_ratio,
    focus_class_ids=None,
):
    if len(pred_masks) != len(target_masks):
        raise ValueError(
            f"Prediction/target count mismatch: {len(pred_masks)} predictions vs {len(target_masks)} targets."
        )
    num_classes = int(len(class_names))
    intersections = torch.zeros(num_classes, dtype=torch.float64)
    unions = torch.zeros(num_classes, dtype=torch.float64)
    gt_pixel_counts = torch.zeros(num_classes, dtype=torch.float64)
    matched_pixels = 0.0
    total_pixels = 0.0
    boundary_meter = BoundaryF1Meter(tolerance_px=boundary_tolerance_px)

    for pred_mask, target_mask in zip(pred_masks, target_masks):
        valid_mask = target_mask >= 0
        matched_pixels += float((pred_mask[valid_mask] == target_mask[valid_mask]).sum().item())
        total_pixels += float(valid_mask.sum().item())
        pred_valid = pred_mask[valid_mask]
        target_valid = target_mask[valid_mask]
        for class_id in range(num_classes):
            pred_class = pred_valid == class_id
            target_class = target_valid == class_id
            intersections[class_id] += float(torch.logical_and(pred_class, target_class).sum().item())
            unions[class_id] += float(torch.logical_or(pred_class, target_class).sum().item())
            gt_pixel_counts[class_id] += float(target_class.sum().item())
        pred_for_boundary = pred_mask.clone()
        target_for_boundary = target_mask.clone()
        pred_for_boundary[~valid_mask] = -1
        target_for_boundary[~valid_mask] = -1
        boundary_meter.update(pred_for_boundary, target_for_boundary)

    total_gt_pixels = float(gt_pixel_counts.sum().item())
    per_class_iou = {}
    per_class_pixel_ratio = {}
    valid_iou_values = []
    for class_id in range(num_classes):
        union = float(unions[class_id].item())
        iou = None if union <= 0.0 else float(intersections[class_id].item() / union)
        ratio = 0.0 if total_gt_pixels <= 0.0 else float(gt_pixel_counts[class_id].item() / total_gt_pixels)
        per_class_iou[int(class_id)] = iou
        per_class_pixel_ratio[int(class_id)] = ratio
        if iou is not None:
            valid_iou_values.append(iou)

    focus_class_ids = (
        sorted(set(int(value) for value in focus_class_ids))
        if focus_class_ids
        else infer_focus_class_ids(class_names)
    )
    small_class_ids = [
        int(class_id)
        for class_id, ratio in per_class_pixel_ratio.items()
        if ratio > 0.0 and ratio <= float(small_class_threshold_ratio)
    ]

    def _mean_over_class_ids(class_ids):
        values = [
            per_class_iou[int(class_id)]
            for class_id in class_ids
            if per_class_iou.get(int(class_id)) is not None
        ]
        return None if not values else float(sum(values) / len(values))

    worst_class_id = None
    worst_class_iou = None
    sortable_iou = [
        (int(class_id), float(iou))
        for class_id, iou in per_class_iou.items()
        if iou is not None
    ]
    if sortable_iou:
        worst_class_id, worst_class_iou = min(sortable_iou, key=lambda item: item[1])

    return {
        "miou": None if not valid_iou_values else float(sum(valid_iou_values) / len(valid_iou_values)),
        "pixel_accuracy": None if total_pixels <= 0.0 else float(matched_pixels / total_pixels),
        "boundary_f1": float(boundary_meter.compute()),
        "per_class_iou": per_class_iou,
        "per_class_iou_by_name": {
            str(class_names[int(class_id)]): value for class_id, value in per_class_iou.items()
        },
        "per_class_pixel_ratio": per_class_pixel_ratio,
        "per_class_pixel_ratio_by_name": {
            str(class_names[int(class_id)]): value for class_id, value in per_class_pixel_ratio.items()
        },
        "small_class_threshold_ratio": float(small_class_threshold_ratio),
        "small_class_ids": small_class_ids,
        "small_class_names": [str(class_names[int(class_id)]) for class_id in small_class_ids],
        "small_class_miou": _mean_over_class_ids(small_class_ids),
        "focus_class_ids": focus_class_ids,
        "focus_class_names": [str(class_names[int(class_id)]) for class_id in focus_class_ids],
        "focus_class_mean_iou": _mean_over_class_ids(focus_class_ids),
        "worst_class_id": None if worst_class_id is None else int(worst_class_id),
        "worst_class_name": None if worst_class_id is None else str(class_names[int(worst_class_id)]),
        "worst_class_iou": None if worst_class_iou is None else float(worst_class_iou),
    }


def evaluate_teacher_on_dataset(
    *,
    run_dir,
    dataset_root,
    split,
    label_spec_path,
    device,
    checkpoint_path=None,
    batch_size=4,
    boundary_tolerance_px=2,
    small_class_threshold_ratio=0.02,
    focus_class_ids=None,
    max_samples=None,
):
    label_metadata = resolve_label_spec_metadata(label_spec_path)
    model, teacher_metadata = load_teacher_model(
        run_dir=run_dir,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    target_size_hw = (int(teacher_metadata["height"]), int(teacher_metadata["width"]))
    pairs = collect_image_mask_pairs(dataset_root, split)
    if max_samples is not None:
        pairs = pairs[: int(max_samples)]
    image_paths = [pair["image_path"] for pair in pairs]
    pred_masks = predict_masks_for_paths(
        image_paths=image_paths,
        model=model,
        input_size_hw=target_size_hw,
        device=device,
        batch_size=batch_size,
        output_size_mode="input",
    )
    target_masks = [
        load_ground_truth_mask(
            pair["mask_path"],
            lookup=label_metadata["lookup"],
            target_size_hw=target_size_hw,
        )
        for pair in pairs
    ]
    metrics = compute_segmentation_metrics(
        pred_masks=pred_masks,
        target_masks=target_masks,
        class_names=label_metadata["class_names"],
        boundary_tolerance_px=boundary_tolerance_px,
        small_class_threshold_ratio=small_class_threshold_ratio,
        focus_class_ids=focus_class_ids,
    )
    metrics.update(
        {
            "sample_count": int(len(pairs)),
            "split": str(split),
            "input_height": int(target_size_hw[0]),
            "input_width": int(target_size_hw[1]),
            "num_classes": int(label_metadata["num_classes"]),
            "teacher_out_channels": int(teacher_metadata["out_channels"]),
            "class_names": label_metadata["class_names"],
            "label_spec_path": label_metadata["label_spec_path"],
            "checkpoint_path": teacher_metadata["checkpoint_path"],
            "net_name": teacher_metadata["net_name"],
            "run_dir": teacher_metadata["run_dir"],
            "epoch_budget": teacher_metadata["epoch_budget"],
            "train_batch_size": teacher_metadata["batch_size"],
        }
    )
    return metrics


def write_teacher_mask_triplet(*, mask_index, rgb_uint8, outdir, stem, num_classes, overlay_alpha):
    raw_dir = Path(outdir) / "teacher_mask_raw"
    color_dir = Path(outdir) / "teacher_mask_color"
    overlay_dir = Path(outdir) / "teacher_overlay"
    raw_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_np = np.asarray(mask_index, dtype=np.int64)
    color_mask = colorize_mask_index(mask_np, num_classes=num_classes)
    overlay = overlay_color_mask_on_image(np.asarray(rgb_uint8, dtype=np.uint8), color_mask, alpha=overlay_alpha)
    Image.fromarray(mask_np.astype(np.uint16)).save(raw_dir / f"{stem}.png")
    Image.fromarray(color_mask).save(color_dir / f"{stem}.png")
    Image.fromarray(overlay).save(overlay_dir / f"{stem}.png")

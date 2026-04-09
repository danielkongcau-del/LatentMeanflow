import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
LDM_ROOT = REPO_ROOT / "third_party" / "latent-diffusion"
TAMING_ROOT = LDM_ROOT / "taming-transformers"

for path in (REPO_ROOT, LDM_ROOT, TAMING_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ldm.util import instantiate_from_config
from latent_meanflow.utils import (
    UNDEFINED_CLASS_ID,
    colorize_mask_index,
    overlay_color_mask_on_image,
    resolve_gray_to_class_id,
)
from latent_meanflow.utils.palette import build_lookup_table, infer_num_classes
from scripts.sample_latent_flow import (
    find_latest_flow_ckpt,
    load_config,
    load_model,
    validate_ckpt_matches_config,
)


DEFAULT_CONFIG = REPO_ROOT / "configs" / "latent_alphaflow_mask2image_unet.yaml"
DEFAULT_NFE_VALUES = [8, 4, 2, 1]
MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample project-layer mask-conditioned image generation p(image | semantic mask)."
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
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="validation", help="Dataset split key: train, validation, or val.")
    parser.add_argument("--mask-dir", type=Path, default=None, help="Optional standalone semantic mask directory.")
    parser.add_argument("--image-dir", type=Path, default=None, help="Optional GT image directory for standalone mask mode.")
    parser.add_argument(
        "--label-spec",
        type=Path,
        default=REPO_ROOT / "configs" / "label_specs" / "remote_semantic.yaml",
        help="Gray-to-class label spec used when reading standalone masks.",
    )
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--nfe-values", type=int, nargs="+", default=DEFAULT_NFE_VALUES)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--overlay-alpha", type=float, default=0.4)
    parser.add_argument("--two-step-time", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def apply_tokenizer_overrides(config, *, tokenizer_config=None, tokenizer_ckpt=None):
    if tokenizer_config is not None:
        OmegaConf.update(
            config,
            "model.params.tokenizer_config_path",
            str(tokenizer_config.resolve()),
            merge=False,
        )
    if tokenizer_ckpt is not None:
        OmegaConf.update(
            config,
            "model.params.tokenizer_ckpt_path",
            str(tokenizer_ckpt.resolve()),
            merge=False,
        )
    return config


def _normalize_rgb_for_save(image):
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu().float()
        if tensor.ndim == 3 and tensor.shape[0] in {1, 3}:
            tensor = tensor.permute(1, 2, 0)
        image_np = tensor.numpy()
    else:
        image_np = np.asarray(image, dtype=np.float32)
    image_np = np.clip((image_np + 1.0) / 2.0, 0.0, 1.0)
    return (image_np * 255.0).astype(np.uint8)


def _load_rgb_image(image_path, size):
    image = Image.open(image_path).convert("RGB")
    if size is not None:
        image = image.resize((size, size), resample=Image.BICUBIC)
    image_np = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return image_np


def _build_onehot(mask_index, num_classes, ignore_index=None):
    onehot = np.zeros(mask_index.shape + (num_classes,), dtype=np.float32)
    if ignore_index is None:
        valid_mask = np.ones(mask_index.shape, dtype=bool)
    else:
        valid_mask = mask_index != int(ignore_index)
    if np.any(valid_mask):
        flat_onehot = onehot.reshape(-1, num_classes)
        flat_index = mask_index.reshape(-1)
        valid_positions = np.nonzero(valid_mask.reshape(-1))[0]
        flat_onehot[valid_positions, flat_index.reshape(-1)[valid_positions]] = 1.0
    return onehot


def _load_mask_index(mask_path, *, label_spec_path, size):
    gray_to_class_id, ignore_index = resolve_gray_to_class_id(label_spec_path, ignore_index=None)
    num_classes = infer_num_classes(gray_to_class_id, ignore_index=ignore_index)
    lookup = build_lookup_table(gray_to_class_id, undefined_value=UNDEFINED_CLASS_ID)

    mask_image = Image.open(mask_path)
    if mask_image.mode != "L":
        mask_image = mask_image.convert("L")
    if size is not None:
        mask_image = mask_image.resize((size, size), resample=Image.NEAREST)
    mask_raw = np.asarray(mask_image, dtype=np.uint8)
    mask_index = lookup[mask_raw].astype(np.int64, copy=False)
    undefined_mask = mask_index == UNDEFINED_CLASS_ID
    if np.any(undefined_mask):
        unknown_values = sorted(int(value) for value in np.unique(mask_raw[undefined_mask]).tolist())
        raise ValueError(
            f"Mask {mask_path} contains gray values missing from {label_spec_path}: {unknown_values}"
        )
    return mask_index, num_classes, ignore_index


def _resolve_dataset_size(config, split_key):
    dataset_cfg = OmegaConf.select(config, f"data.params.{split_key}.params")
    if dataset_cfg is None:
        return None
    size = OmegaConf.select(dataset_cfg, "size")
    return None if size is None else int(size)


def _resolve_split_key(config, split_name):
    normalized = split_name.lower()
    if normalized == "val":
        normalized = "validation"
    dataset_cfg = OmegaConf.select(config, f"data.params.{normalized}")
    if dataset_cfg is None:
        raise KeyError(f"Config does not define data.params.{normalized}")
    return normalized


def _load_examples_from_dataset(config, split, n_samples):
    dataset_cfg = OmegaConf.select(config, f"data.params.{split}")
    dataset = instantiate_from_config(dataset_cfg)
    examples = []
    limit = min(len(dataset), int(n_samples))
    for index in range(limit):
        sample = dataset[index]
        examples.append(
            {
                "stem": Path(sample["image_path"]).stem,
                "mask_index": np.asarray(sample["mask_index"], dtype=np.int64),
                "mask_onehot": np.asarray(sample["mask_onehot"], dtype=np.float32),
                "ground_truth_image": np.asarray(sample["image"], dtype=np.float32),
                "image_path": sample.get("image_path"),
                "mask_path": sample.get("mask_path"),
            }
        )
    return examples


def _find_paired_image(mask_path, image_dir):
    if image_dir is None:
        return None
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{mask_path.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _load_examples_from_mask_dir(mask_dir, *, image_dir, label_spec, size, n_samples):
    mask_dir = mask_dir.resolve()
    image_dir = None if image_dir is None else image_dir.resolve()
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    mask_paths = []
    for ext in MASK_EXTS:
        mask_paths.extend(mask_dir.glob(f"*{ext}"))
    mask_paths = sorted(mask_paths)[: int(n_samples)]
    if not mask_paths:
        raise FileNotFoundError(f"No mask files found under {mask_dir}")

    examples = []
    for mask_path in mask_paths:
        mask_index, num_classes, ignore_index = _load_mask_index(
            mask_path,
            label_spec_path=label_spec,
            size=size,
        )
        gt_image = None
        image_path = _find_paired_image(mask_path, image_dir=image_dir)
        if image_path is not None:
            gt_image = _load_rgb_image(image_path, size=size)
        examples.append(
            {
                "stem": mask_path.stem,
                "mask_index": mask_index,
                "mask_onehot": _build_onehot(mask_index, num_classes=num_classes, ignore_index=ignore_index),
                "ground_truth_image": gt_image,
                "image_path": None if image_path is None else str(image_path),
                "mask_path": str(mask_path),
            }
        )
    return examples


def load_examples(config, *, split, mask_dir, image_dir, label_spec, n_samples):
    if mask_dir is None:
        split_key = _resolve_split_key(config, split)
        return _load_examples_from_dataset(config, split=split_key, n_samples=n_samples), split_key
    size = _resolve_dataset_size(config, "validation")
    return (
        _load_examples_from_mask_dir(
            mask_dir,
            image_dir=image_dir,
            label_spec=label_spec,
            size=size,
            n_samples=n_samples,
        ),
        "mask_dir",
    )


def _prepare_outdir(path, overwrite):
    if path.exists():
        existing = list(path.rglob("*"))
        if existing and not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {path}. "
                "Use a fresh outdir or pass --overwrite."
            )
    path.mkdir(parents=True, exist_ok=True)


def _make_panel(mask_color, generated_image, ground_truth_image, overlay):
    tiles = [mask_color]
    if ground_truth_image is not None:
        tiles.append(ground_truth_image)
    tiles.append(generated_image)
    tiles.append(overlay)
    return np.concatenate(tiles, axis=1)


def save_mask_conditioned_sample(
    *,
    outdir,
    index,
    stem,
    mask_index,
    generated_image,
    num_classes,
    overlay_alpha,
    ground_truth_image=None,
):
    input_mask_raw_dir = outdir / "input_mask_raw"
    input_mask_color_dir = outdir / "input_mask_color"
    generated_image_dir = outdir / "generated_image"
    ground_truth_image_dir = outdir / "ground_truth_image"
    overlay_dir = outdir / "overlay"
    panel_dir = outdir / "panel"
    for directory in (
        input_mask_raw_dir,
        input_mask_color_dir,
        generated_image_dir,
        overlay_dir,
        panel_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if ground_truth_image is not None:
        ground_truth_image_dir.mkdir(parents=True, exist_ok=True)

    mask_index_np = np.asarray(mask_index, dtype=np.int64)
    raw_mask_np = mask_index_np.copy()
    raw_mask_np[raw_mask_np < 0] = 65535
    raw_mask_np = raw_mask_np.astype(np.uint16)
    mask_color_np = colorize_mask_index(mask_index_np, num_classes=num_classes)
    generated_image_np = _normalize_rgb_for_save(generated_image)
    ground_truth_image_np = (
        None if ground_truth_image is None else _normalize_rgb_for_save(ground_truth_image)
    )
    overlay_np = overlay_color_mask_on_image(generated_image_np, mask_color_np, alpha=overlay_alpha)
    panel_np = _make_panel(mask_color_np, generated_image_np, ground_truth_image_np, overlay_np)

    file_stem = f"{index:06}_{stem}"
    Image.fromarray(raw_mask_np).save(input_mask_raw_dir / f"{file_stem}.png")
    Image.fromarray(mask_color_np).save(input_mask_color_dir / f"{file_stem}.png")
    Image.fromarray(generated_image_np).save(generated_image_dir / f"{file_stem}.png")
    Image.fromarray(overlay_np).save(overlay_dir / f"{file_stem}.png")
    Image.fromarray(panel_np).save(panel_dir / f"{file_stem}.png")
    if ground_truth_image_np is not None:
        Image.fromarray(ground_truth_image_np).save(ground_truth_image_dir / f"{file_stem}.png")


@torch.no_grad()
def generate_mask_conditioned_sweep(
    *,
    model,
    examples,
    outdir,
    nfe_values,
    seed,
    batch_size,
    overlay_alpha,
):
    device = model.device
    latent_shape = (model.latent_channels, *model.latent_spatial_shape)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise_bank = torch.randn((len(examples), *latent_shape), generator=generator)

    summary_rows = []
    for nfe in nfe_values:
        nfe = int(nfe)
        nfe_dir = outdir / f"nfe{nfe}"
        _prepare_outdir(nfe_dir, overwrite=True)

        for start in range(0, len(examples), int(batch_size)):
            batch_examples = examples[start : start + int(batch_size)]
            batch_noise = noise_bank[start : start + len(batch_examples)].to(device)
            batch_mask_onehot = torch.from_numpy(
                np.stack([example["mask_onehot"] for example in batch_examples], axis=0)
            )
            condition = model.build_condition_from_mask_onehot(
                batch_mask_onehot,
                device=device,
                dtype=torch.float32,
            )
            latents = model.sample_latents(
                batch_size=len(batch_examples),
                nfe=nfe,
                device=device,
                condition=condition,
                noise=batch_noise,
            )
            decoded = model.decode_latents(latents)
            for local_idx, example in enumerate(batch_examples):
                save_mask_conditioned_sample(
                    outdir=nfe_dir,
                    index=start + local_idx,
                    stem=example["stem"],
                    mask_index=example["mask_index"],
                    generated_image=decoded["rgb_recon"][local_idx],
                    ground_truth_image=example["ground_truth_image"],
                    num_classes=model.num_classes,
                    overlay_alpha=overlay_alpha,
                )

        summary_rows.append(
            {
                "nfe": nfe,
                "outdir": str(nfe_dir),
                "input_mask_raw_count": len(list((nfe_dir / "input_mask_raw").glob("*.png"))),
                "input_mask_color_count": len(list((nfe_dir / "input_mask_color").glob("*.png"))),
                "generated_image_count": len(list((nfe_dir / "generated_image").glob("*.png"))),
                "ground_truth_image_count": len(list((nfe_dir / "ground_truth_image").glob("*.png"))),
                "overlay_count": len(list((nfe_dir / "overlay").glob("*.png"))),
                "panel_count": len(list((nfe_dir / "panel").glob("*.png"))),
            }
        )
    return summary_rows


@torch.no_grad()
def main():
    args = parse_args()
    config = load_config(args.config)
    apply_tokenizer_overrides(
        config,
        tokenizer_config=args.tokenizer_config,
        tokenizer_ckpt=args.tokenizer_ckpt,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))

    ckpt_path = args.ckpt or find_latest_flow_ckpt(args.config, config)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError("Mask-conditioned checkpoint not found. Pass --ckpt explicitly.")
    validate_ckpt_matches_config(args.config, ckpt_path)

    outdir = args.outdir.resolve()
    _prepare_outdir(outdir, overwrite=args.overwrite)

    model = load_model(config, ckpt_path, device=device)
    if not hasattr(model, "build_condition_from_mask_onehot"):
        raise TypeError(
            f"Checkpoint/config '{args.config.name}' is not a mask-conditioned image route."
        )
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
    if examples:
        condition_channels = int(np.asarray(examples[0]["mask_onehot"]).shape[-1])
        if int(model.num_classes) != condition_channels:
            raise ValueError(
                f"Mask class count mismatch: examples carry {condition_channels} one-hot channels, "
                f"but the checkpoint expects {model.num_classes} spatial condition channels."
            )
    summary_rows = generate_mask_conditioned_sweep(
        model=model,
        examples=examples,
        outdir=outdir,
        nfe_values=args.nfe_values,
        seed=args.seed,
        batch_size=args.batch_size,
        overlay_alpha=args.overlay_alpha,
    )
    backbone = getattr(model, "backbone", None)

    summary = {
        "config": str(args.config.resolve()),
        "checkpoint": str(ckpt_path.resolve()),
        "tokenizer_config": str(Path(config.model.params.tokenizer_config_path).resolve()),
        "tokenizer_checkpoint": str(Path(config.model.params.tokenizer_ckpt_path).resolve()),
        "source_mode": source_mode,
        "split": args.split,
        "mask_dir": None if args.mask_dir is None else str(args.mask_dir.resolve()),
        "image_dir": None if args.image_dir is None else str(args.image_dir.resolve()),
        "label_spec": str(args.label_spec.resolve()),
        "seed": int(args.seed),
        "n_samples": int(len(examples)),
        "batch_size": int(args.batch_size),
        "nfe_values": [int(value) for value in args.nfe_values],
        "task": "p(image | semantic_mask)",
        "condition_mode": None if backbone is None else getattr(backbone, "condition_mode", None),
        "condition_source": None if backbone is None else getattr(backbone, "condition_source", None),
        "use_boundary_condition": None if backbone is None else bool(getattr(backbone, "use_boundary_condition", False)),
        "boundary_mode": None if backbone is None else getattr(backbone, "boundary_mode", None),
        "use_semantic_condition_encoder": None
        if backbone is None
        else bool(getattr(backbone, "use_semantic_condition_encoder", False)),
        "results": summary_rows,
    }

    summary_json_path = outdir / "summary.json"
    summary_csv_path = outdir / "summary.csv"
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
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved mask-conditioned image sweep to {outdir}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()

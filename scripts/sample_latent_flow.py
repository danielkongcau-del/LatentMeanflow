import argparse
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
from latent_meanflow.utils import colorize_mask_index, overlay_color_mask_on_image


DEFAULT_CONFIG = REPO_ROOT / "configs" / "latent_alphaflow_semantic_256_unet.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample paired RGB image and semantic mask outputs from latent flow priors."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=REPO_ROOT / "outputs" / "latent_flow_samples")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--nfe",
        type=int,
        default=1,
        help="Number of function evaluations. Common few-step sweeps use 8, 4, 2, and 1.",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--class-label", type=int, default=None, help="Optional image-level condition.")
    parser.add_argument("--overlay-alpha", type=float, default=0.4)
    parser.add_argument("--two-step-time", type=float, default=None, help="Optional midpoint override for 2-step interval sampling.")
    return parser.parse_args()


def load_config(config_path, overrides=None):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = OmegaConf.load(config_path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    return config


def resolve_run_tag(config_path, config):
    stem = config_path.stem.strip().lower()
    if stem:
        return stem
    objective_name = OmegaConf.select(config, "model.params.objective_name")
    if objective_name is not None:
        return str(objective_name).lower()
    return "latent_flow"


def resolve_objective_name(config):
    objective_name = OmegaConf.select(config, "model.params.objective_name")
    if objective_name is not None:
        return str(objective_name)
    stem = Path(OmegaConf.select(config, "model.params.tokenizer_config_path", default="latent_flow")).stem
    return stem.lower()


def find_latest_flow_ckpt(config_path, config):
    run_tag = resolve_run_tag(config_path, config)
    objective_name = resolve_objective_name(config)
    candidates = sorted(
        REPO_ROOT.glob("logs/**/checkpoints/last.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        candidate_run_dir = candidate.parent.parent.name.lower()
        if candidate_run_dir == run_tag or candidate_run_dir.endswith(f"_{run_tag}"):
            return candidate
    for candidate in candidates:
        candidate_text = str(candidate).lower()
        if objective_name in candidate_text:
            return candidate
    for candidate in candidates:
        if "latent_" in str(candidate).lower():
            return candidate
    return None


def validate_ckpt_matches_config(config_path, ckpt_path):
    run_tag = config_path.stem.strip().lower()
    if not run_tag:
        return
    run_dir_name = ckpt_path.parent.parent.name.lower()
    if run_dir_name != run_tag and not run_dir_name.endswith(f"_{run_tag}"):
        raise ValueError(
            f"Checkpoint/config mismatch: config '{config_path.name}' expects a run path containing "
            f"the exact run tag '{run_tag}', but got '{ckpt_path}'. This usually means a tiny/debug checkpoint is being "
            "mixed with a baseline config. Use scripts/find_checkpoint.py to select the correct checkpoint."
        )


def load_checkpoint_state(ckpt_path):
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def _normalize_contract_value(field_name, value):
    if value in {None, ""}:
        return None
    if field_name.endswith("_path"):
        return str(Path(str(value)).expanduser().resolve())
    if isinstance(value, bool):
        return bool(value)
    return str(value)


def validate_ckpt_matches_resolved_config(
    config,
    ckpt_path,
    *,
    config_path=None,
    checkpoint_state=None,
    fields=None,
):
    if config_path is not None:
        validate_ckpt_matches_config(config_path, ckpt_path)

    state = checkpoint_state if checkpoint_state is not None else load_checkpoint_state(ckpt_path)
    if not isinstance(state, dict):
        return state
    hyper_parameters = state.get("hyper_parameters", {})
    if not isinstance(hyper_parameters, dict):
        return state

    compare_fields = tuple(
        fields
        or (
            "monitor",
            "objective_name",
            "tokenizer_config_path",
            "tokenizer_ckpt_path",
            "freeze_tokenizer",
            "tokenizer_sample_posterior",
        )
    )
    mismatches = []
    for field_name in compare_fields:
        runtime_value = OmegaConf.select(config, f"model.params.{field_name}", default=None)
        ckpt_value = hyper_parameters.get(field_name)
        if runtime_value in {None, ""} or ckpt_value in {None, ""}:
            continue
        runtime_norm = _normalize_contract_value(field_name, runtime_value)
        ckpt_norm = _normalize_contract_value(field_name, ckpt_value)
        if runtime_norm != ckpt_norm:
            mismatches.append(
                f"{field_name}: resolved runtime value {runtime_norm!r} != checkpoint value {ckpt_norm!r}"
            )
    if mismatches:
        raise ValueError(
            "Checkpoint/resolved-config mismatch after runtime overrides. "
            "The loaded checkpoint does not match the final OmegaConf config:\n- "
            + "\n- ".join(mismatches)
        )
    return state


def load_model(config, ckpt_path, device, checkpoint_state=None):
    model = instantiate_from_config(config.model)
    state = checkpoint_state if checkpoint_state is not None else load_checkpoint_state(ckpt_path)
    state_dict = state["state_dict"] if "state_dict" in state else state
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def save_pair(rgb_tensor, mask_index_tensor, outdir, index, num_classes, overlay_alpha):
    image_dir = outdir / "image"
    raw_mask_dir = outdir / "mask_raw"
    overlay_dir = outdir / "overlay"
    color_mask_dir = outdir / "mask_color"
    image_dir.mkdir(parents=True, exist_ok=True)
    raw_mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    color_mask_dir.mkdir(parents=True, exist_ok=True)

    rgb = torch.clamp((rgb_tensor + 1.0) / 2.0, 0.0, 1.0)
    rgb_np = (rgb.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    mask_index_np = mask_index_tensor.cpu().numpy().astype(np.uint16)
    color_mask_np = colorize_mask_index(mask_index_np.astype(np.int64), num_classes=num_classes)
    overlay_np = overlay_color_mask_on_image(rgb_np, color_mask_np, alpha=overlay_alpha)

    stem = f"{index:06}"
    Image.fromarray(rgb_np).save(image_dir / f"{stem}.png")
    Image.fromarray(mask_index_np).save(raw_mask_dir / f"{stem}.png")
    Image.fromarray(color_mask_np).save(color_mask_dir / f"{stem}.png")
    Image.fromarray(overlay_np).save(overlay_dir / f"{stem}.png")


@torch.no_grad()
def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    ckpt_path = args.ckpt or find_latest_flow_ckpt(args.config, config)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError("Latent flow checkpoint not found. Pass --ckpt explicitly.")
    validate_ckpt_matches_config(args.config, ckpt_path)

    model = load_model(config, ckpt_path, device)
    if args.class_label is not None and not getattr(model, "use_class_condition", False):
        raise ValueError(
            "--class-label was provided, but this latent flow checkpoint is configured as unconditional."
        )
    if args.two_step_time is not None and hasattr(model, "sampler") and hasattr(model.sampler, "two_step_time"):
        model.sampler.two_step_time = float(args.two_step_time)

    remaining = args.n_samples
    index = 0
    while remaining > 0:
        batch_size = min(args.batch_size, remaining)
        condition = None
        if args.class_label is not None:
            condition = torch.full((batch_size,), args.class_label, device=device, dtype=torch.long)

        latents = model.sample_latents(
            batch_size=batch_size,
            nfe=args.nfe,
            device=device,
            condition=condition,
        )
        decoded = model.decode_latents(latents)

        for sample_idx in range(batch_size):
            save_pair(
                decoded["rgb_recon"][sample_idx],
                decoded["mask_index"][sample_idx],
                outdir=args.outdir,
                index=index,
                num_classes=model.num_classes,
                overlay_alpha=args.overlay_alpha,
            )
            index += 1
        remaining -= batch_size


if __name__ == "__main__":
    main()

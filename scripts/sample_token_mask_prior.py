import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
LDM_ROOT = REPO_ROOT / "third_party" / "latent-diffusion"
TAMING_ROOT = LDM_ROOT / "taming-transformers"

for path in (REPO_ROOT, LDM_ROOT, TAMING_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from latent_meanflow.utils import colorize_mask_index
from scripts.sample_latent_flow import (
    load_config,
    load_checkpoint_state,
    load_model,
    validate_ckpt_matches_resolved_config,
)
from scripts.sample_mask_prior import _prepare_outdir


DEFAULT_CONFIG = REPO_ROOT / "configs" / "token_mask_prior_vq_sit.yaml"
DEFAULT_TOKENIZER_CONFIG = REPO_ROOT / "configs" / "semantic_mask_vq_tokenizer_main_balanced_256.yaml"
DEFAULT_NFE_VALUES = [8, 4, 2, 1]
TOKEN_MASK_PRIOR_TARGETS = {
    "latent_meanflow.trainers.token_mask_prior_trainer.TokenMaskPriorTrainer",
    "latent_meanflow.trainers.token_code_autoregressive_prior_trainer.TokenCodeAutoregressivePriorTrainer",
    "latent_meanflow.trainers.token_code_maskgit_prior_trainer.TokenCodeMaskGitPriorTrainer",
}
TOKEN_CODE_AR_TARGET = "latent_meanflow.trainers.token_code_autoregressive_prior_trainer.TokenCodeAutoregressivePriorTrainer"
TOKEN_CODE_MASKGIT_TARGET = "latent_meanflow.trainers.token_code_maskgit_prior_trainer.TokenCodeMaskGitPriorTrainer"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sample unconditional tokenizer code grids from the project-layer token-code p(mask) route, "
            "then decode them through the frozen balanced VQ tokenizer into semantic masks."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, default=DEFAULT_TOKENIZER_CONFIG)
    parser.add_argument("--tokenizer-ckpt", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nfe-values", type=int, nargs="+", default=DEFAULT_NFE_VALUES)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Extra OmegaConf dotlist override. Repeat as needed, without a leading --.",
    )
    return parser.parse_args()


def apply_tokenizer_overrides(config, *, tokenizer_config, tokenizer_ckpt):
    OmegaConf.update(
        config,
        "model.params.tokenizer_config_path",
        str(tokenizer_config.resolve()),
        merge=False,
    )
    OmegaConf.update(
        config,
        "model.params.tokenizer_ckpt_path",
        str(tokenizer_ckpt.resolve()),
        merge=False,
    )
    return config


def is_token_mask_prior_route(config):
    return str(OmegaConf.select(config, "model.target", default="")) in TOKEN_MASK_PRIOR_TARGETS


def resolve_configured_tokenizer_artifacts(config, *, route_name):
    tokenizer_config_value = OmegaConf.select(config, "model.params.tokenizer_config_path", default=None)
    tokenizer_ckpt_value = OmegaConf.select(config, "model.params.tokenizer_ckpt_path", default=None)
    if tokenizer_config_value in {None, ""}:
        raise ValueError(f"{route_name} requires model.params.tokenizer_config_path to be set in the resolved config.")
    if tokenizer_ckpt_value in {None, ""}:
        raise ValueError(f"{route_name} requires model.params.tokenizer_ckpt_path to be set in the resolved config.")

    tokenizer_config_path = Path(str(tokenizer_config_value)).expanduser().resolve()
    tokenizer_ckpt_path = Path(str(tokenizer_ckpt_value)).expanduser().resolve()
    if not tokenizer_config_path.exists():
        raise FileNotFoundError(f"{route_name} tokenizer config file not found: {tokenizer_config_path}")
    if not tokenizer_ckpt_path.exists():
        raise FileNotFoundError(f"{route_name} tokenizer checkpoint not found: {tokenizer_ckpt_path}")
    return tokenizer_config_path, tokenizer_ckpt_path


def validate_token_mask_prior_checkpoint_contract(config, ckpt_path, *, config_path=None, checkpoint_state=None):
    return validate_ckpt_matches_resolved_config(
        config,
        ckpt_path,
        config_path=config_path,
        checkpoint_state=checkpoint_state,
        fields=(
            "monitor",
            "objective_name",
            "tokenizer_config_path",
            "tokenizer_ckpt_path",
            "freeze_tokenizer",
            "tokenizer_sample_posterior",
        ),
    )


def extract_token_mask_prior_route_metadata(*, config, model=None):
    route_target = str(OmegaConf.select(config, "model.target", default=""))
    if route_target == TOKEN_CODE_AR_TARGET or getattr(model, "route_family", None) == "autoregressive":
        return {
            "corruption_mode": "next_token_teacher_forcing",
            "full_mask_batch_fraction": 0.0,
            "high_mask_batch_fraction": 0.0,
            "high_mask_min_ratio": 0.0,
            "refinement_mode": "autoregressive",
            "final_full_reveal": True,
            "min_keep_fraction": 1.0,
            "lock_noise_scale": 0.0,
            "reveal_noise_scale": 0.0,
            "sample_temperature": float(
                getattr(
                    model,
                    "sample_temperature",
                    OmegaConf.select(config, "model.params.sample_temperature", default=1.0),
                )
            ),
            "context_length": int(
                getattr(
                    model,
                    "context_length",
                    OmegaConf.select(config, "model.params.backbone_config.params.block_size", default=0),
                )
            ),
            "permuter": str(
                getattr(
                    model,
                    "permuter_name",
                    str(
                        OmegaConf.select(
                            config,
                            "model.params.permuter_config.target",
                            default="taming.modules.transformer.permuter.Identity",
                        )
                    ).rsplit(".", 1)[-1],
                )
            ),
            "nfe_ignored": True,
        }
    if route_target == TOKEN_CODE_MASKGIT_TARGET or getattr(model, "route_family", None) == "maskgit":
        backbone_params = OmegaConf.select(config, "model.params.backbone_config.params", default={}) or {}
        mask_schedule_type = str(backbone_params.get("mask_schedule_type", "cosine"))
        return {
            "corruption_mode": "masked_token_ce",
            "full_mask_batch_fraction": 0.0,
            "high_mask_batch_fraction": 0.0,
            "high_mask_min_ratio": 0.0,
            "refinement_mode": "canonical_maskgit",
            "final_full_reveal": True,
            "min_keep_fraction": 0.0,
            "lock_noise_scale": 0.0,
            "reveal_noise_scale": 0.0,
            "sample_temperature": float(
                getattr(
                    model,
                    "sample_temperature",
                    OmegaConf.select(config, "model.params.sample_temperature", default=1.0),
                )
            ),
            "sample_top_k": (
                None
                if getattr(model, "sample_top_k", OmegaConf.select(config, "model.params.sample_top_k", default=None))
                in {None, ""}
                else int(
                    getattr(
                        model,
                        "sample_top_k",
                        OmegaConf.select(config, "model.params.sample_top_k", default=None),
                    )
                )
            ),
            "base_gumbel_temp": float(
                getattr(
                    model,
                    "sample_base_gumbel_temp",
                    OmegaConf.select(config, "model.params.sample_base_gumbel_temp", default=4.5),
                )
            ),
            "mask_schedule_type": str(getattr(getattr(model, "backbone", None), "mask_schedule_type", mask_schedule_type)),
            "permuter": str(
                getattr(
                    model,
                    "permuter_name",
                    str(
                        OmegaConf.select(
                            config,
                            "model.params.permuter_config.target",
                            default="taming.modules.transformer.permuter.Identity",
                        )
                    ).rsplit(".", 1)[-1],
                )
            ),
            "nfe_ignored": False,
        }

    objective_cfg = OmegaConf.select(config, "model.params.objective_config.params", default={}) or {}
    sampler_cfg = OmegaConf.select(config, "model.params.sampler_config.params", default={}) or {}
    route = {
        "corruption_mode": str(objective_cfg.get("corruption_mode", "bernoulli")),
        "full_mask_batch_fraction": float(objective_cfg.get("full_mask_batch_fraction", 0.0)),
        "high_mask_batch_fraction": float(objective_cfg.get("high_mask_batch_fraction", 0.0)),
        "high_mask_min_ratio": float(objective_cfg.get("high_mask_min_ratio", 0.0)),
        "refinement_mode": str(sampler_cfg.get("refinement_mode", "progressive_reveal")),
        "final_full_reveal": bool(sampler_cfg.get("final_full_reveal", True)),
        "min_keep_fraction": float(sampler_cfg.get("min_keep_fraction", 0.0)),
        "lock_noise_scale": float(sampler_cfg.get("lock_noise_scale", sampler_cfg.get("reveal_noise_scale", 0.0))),
        "reveal_noise_scale": float(sampler_cfg.get("reveal_noise_scale", 0.0)),
        "sample_temperature": float(sampler_cfg.get("sample_temperature", 1.0)),
    }
    if model is not None:
        sampler = getattr(model, "sampler", None)
        objective = getattr(model, "objective", None)
        if sampler is not None:
            route["refinement_mode"] = str(getattr(sampler, "refinement_mode", route["refinement_mode"]))
            route["final_full_reveal"] = bool(getattr(sampler, "final_full_reveal", route["final_full_reveal"]))
            route["min_keep_fraction"] = float(getattr(sampler, "min_keep_fraction", route["min_keep_fraction"]))
            route["lock_noise_scale"] = float(getattr(sampler, "lock_noise_scale", route["lock_noise_scale"]))
            route["reveal_noise_scale"] = float(getattr(sampler, "reveal_noise_scale", route["reveal_noise_scale"]))
            route["sample_temperature"] = float(getattr(sampler, "sample_temperature", route["sample_temperature"]))
        if objective is not None:
            route["corruption_mode"] = str(getattr(objective, "corruption_mode", route["corruption_mode"]))
            route["full_mask_batch_fraction"] = float(
                getattr(objective, "full_mask_batch_fraction", route["full_mask_batch_fraction"])
            )
            route["high_mask_batch_fraction"] = float(
                getattr(objective, "high_mask_batch_fraction", route["high_mask_batch_fraction"])
            )
            route["high_mask_min_ratio"] = float(
                getattr(objective, "high_mask_min_ratio", route["high_mask_min_ratio"])
            )
    return route


def _write_markdown_report(path, summary):
    route = summary.get("route_metadata", {})
    lines = [
        "# Token Mask Prior Sampling Summary",
        "",
        f"- task: `{summary['task']}`",
        f"- config: `{summary['config']}`",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- tokenizer config: `{summary['tokenizer_config']}`",
        f"- tokenizer checkpoint: `{summary['tokenizer_checkpoint']}`",
        f"- refinement mode: `{route.get('refinement_mode', 'unknown')}`",
        f"- corruption mode: `{route.get('corruption_mode', 'unknown')}`",
        f"- final full reveal: `{route.get('final_full_reveal', 'unknown')}`",
        f"- min keep fraction: `{route.get('min_keep_fraction', 'unknown')}`",
        f"- lock noise scale: `{route.get('lock_noise_scale', 'unknown')}`",
        f"- reveal noise scale: `{route.get('reveal_noise_scale', 'unknown')}`",
        f"- sample temperature: `{route.get('sample_temperature', 'unknown')}`",
        "",
        "| NFE | active codes | active code fraction | code perplexity | unique codes / sample |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in summary["results"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(result["nfe"])),
                    str(int(result["active_code_count"])),
                    f"{float(result['active_code_fraction']):.4f}",
                    f"{float(result['code_perplexity']):.2f}",
                    f"{float(result['unique_code_count_mean']):.2f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _boundary_map(mask_index):
    mask_index = np.asarray(mask_index, dtype=np.int64)
    boundary = np.zeros(mask_index.shape, dtype=np.uint8)
    vertical = mask_index[1:, :] != mask_index[:-1, :]
    horizontal = mask_index[:, 1:] != mask_index[:, :-1]
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    return (boundary * 255).astype(np.uint8)


def _make_mask_panel(mask_index, num_classes):
    mask_index = np.asarray(mask_index, dtype=np.int64)
    raw = np.zeros(mask_index.shape, dtype=np.uint8)
    denom = max(int(num_classes) - 1, 1)
    raw = np.clip(np.round(mask_index.astype(np.float32) * (255.0 / float(denom))), 0.0, 255.0).astype(np.uint8)
    raw_rgb = np.repeat(raw[:, :, None], 3, axis=2)
    color = colorize_mask_index(mask_index, num_classes=num_classes)
    boundary = np.repeat(_boundary_map(mask_index)[:, :, None], 3, axis=2)
    panel = np.concatenate([raw_rgb, color, boundary], axis=1)
    return raw_rgb, color, boundary, panel


def _save_sample(*, codes, mask_index, outdir, index, num_classes):
    code_raw_dir = outdir / "code_raw"
    mask_raw_dir = outdir / "mask_raw"
    mask_color_dir = outdir / "mask_color"
    boundary_dir = outdir / "boundary"
    panel_dir = outdir / "panel"
    for directory in (code_raw_dir, mask_raw_dir, mask_color_dir, boundary_dir, panel_dir):
        directory.mkdir(parents=True, exist_ok=True)

    stem = f"{int(index):06}"
    Image.fromarray(np.asarray(codes, dtype=np.uint16)).save(code_raw_dir / f"{stem}.png")
    Image.fromarray(np.asarray(mask_index, dtype=np.uint16)).save(mask_raw_dir / f"{stem}.png")
    _, color, boundary, panel = _make_mask_panel(mask_index, num_classes=num_classes)
    Image.fromarray(color).save(mask_color_dir / f"{stem}.png")
    Image.fromarray(boundary).save(boundary_dir / f"{stem}.png")
    Image.fromarray(panel).save(panel_dir / f"{stem}.png")


def _summarize_codes(code_grids, *, codebook_size):
    flat = np.concatenate([grid.reshape(-1) for grid in code_grids], axis=0)
    counts = np.bincount(flat.astype(np.int64), minlength=int(codebook_size)).astype(np.float64)
    probs = counts / max(1.0, float(counts.sum()))
    unique_per_sample = np.asarray([np.unique(grid).size for grid in code_grids], dtype=np.float64)
    return {
        "active_code_count": int((counts > 0).sum()),
        "active_code_fraction": float((counts > 0).sum() / float(max(int(codebook_size), 1))),
        "code_perplexity": float(np.exp(-(probs * np.log(np.clip(probs, 1.0e-10, 1.0))).sum())),
        "unique_code_count_mean": float(unique_per_sample.mean()),
        "unique_code_count_std": float(unique_per_sample.std()),
    }


@torch.no_grad()
def generate_token_mask_prior_sweep(*, model, outdir, nfe_values, seed, n_samples, batch_size):
    device = model.device
    latent_shape = (model.latent_channels, *model.latent_spatial_shape)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise_bank = torch.randn((int(n_samples), *latent_shape), generator=generator)

    summary_rows = []
    for nfe in nfe_values:
        nfe = int(nfe)
        nfe_dir = outdir / f"nfe{nfe}"
        _prepare_outdir(nfe_dir, overwrite=True)
        collected_codes = []

        for start in range(0, int(n_samples), int(batch_size)):
            current_batch = min(int(batch_size), int(n_samples) - start)
            noise = noise_bank[start : start + current_batch].to(device)
            sampled_codes = model.sample_latents(
                batch_size=current_batch,
                nfe=nfe,
                device=device,
                noise=noise,
            )
            decoded = model.decode_latents(sampled_codes)
            for local_idx in range(current_batch):
                code_grid = sampled_codes[local_idx].detach().cpu().numpy().astype(np.int64, copy=False)
                collected_codes.append(code_grid)
                _save_sample(
                    codes=code_grid,
                    mask_index=decoded["mask_index"][local_idx].detach().cpu().numpy().astype(np.int64, copy=False),
                    outdir=nfe_dir,
                    index=start + local_idx,
                    num_classes=model.num_classes,
                )

        code_summary = _summarize_codes(collected_codes, codebook_size=model.codebook_size)
        summary_rows.append(
            {
                "nfe": nfe,
                "outdir": str(nfe_dir),
                "code_raw_count": len(list((nfe_dir / "code_raw").glob("*.png"))),
                "mask_raw_count": len(list((nfe_dir / "mask_raw").glob("*.png"))),
                "mask_color_count": len(list((nfe_dir / "mask_color").glob("*.png"))),
                "boundary_count": len(list((nfe_dir / "boundary").glob("*.png"))),
                "panel_count": len(list((nfe_dir / "panel").glob("*.png"))),
                "active_code_count": int(code_summary["active_code_count"]),
                "active_code_fraction": float(code_summary["active_code_fraction"]),
                "code_perplexity": float(code_summary["code_perplexity"]),
                "unique_code_count_mean": float(code_summary["unique_code_count_mean"]),
                "unique_code_count_std": float(code_summary["unique_code_count_std"]),
            }
        )
    return summary_rows


@torch.no_grad()
def main():
    args = parse_args()
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Token-mask prior checkpoint not found: {args.ckpt}")
    if not args.tokenizer_ckpt.exists():
        raise FileNotFoundError(f"Frozen tokenizer checkpoint not found: {args.tokenizer_ckpt}")

    config = load_config(args.config, overrides=args.overrides)
    apply_tokenizer_overrides(
        config,
        tokenizer_config=args.tokenizer_config,
        tokenizer_ckpt=args.tokenizer_ckpt,
    )
    resolved_tokenizer_config_path, resolved_tokenizer_ckpt_path = resolve_configured_tokenizer_artifacts(
        config,
        route_name="Token-mask prior sampling",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))

    checkpoint_state = load_checkpoint_state(args.ckpt.resolve())
    validate_token_mask_prior_checkpoint_contract(
        config,
        args.ckpt.resolve(),
        config_path=args.config,
        checkpoint_state=checkpoint_state,
    )
    outdir = args.outdir.resolve()
    _prepare_outdir(outdir, overwrite=args.overwrite)
    model = load_model(config, args.ckpt.resolve(), device=device, checkpoint_state=checkpoint_state)
    route_metadata = extract_token_mask_prior_route_metadata(config=config, model=model)

    summary_rows = generate_token_mask_prior_sweep(
        model=model,
        outdir=outdir,
        nfe_values=args.nfe_values,
        seed=args.seed,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
    )

    summary = {
        "task": "p(token_codes) -> frozen tokenizer decode -> semantic_mask",
        "config": str(args.config.resolve()),
        "checkpoint": str(args.ckpt.resolve()),
        "tokenizer_config": str(resolved_tokenizer_config_path),
        "tokenizer_checkpoint": str(resolved_tokenizer_ckpt_path),
        "config_overrides": list(args.overrides),
        "seed": int(args.seed),
        "n_samples": int(args.n_samples),
        "batch_size": int(args.batch_size),
        "nfe_values": [int(value) for value in args.nfe_values],
        "monitor": getattr(model, "monitor", None),
        "num_classes": int(model.num_classes),
        "codebook_size": int(model.codebook_size),
        "token_spatial_shape": [int(v) for v in model.token_spatial_shape],
        "mask_spatial_shape": [int(v) for v in model.mask_spatial_shape],
        "route_metadata": route_metadata,
        "results": summary_rows,
    }

    summary_json_path = outdir / "summary.json"
    summary_csv_path = outdir / "summary.csv"
    summary_md_path = outdir / "summary.md"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "nfe",
                "outdir",
                "code_raw_count",
                "mask_raw_count",
                "mask_color_count",
                "boundary_count",
                "panel_count",
                "active_code_count",
                "active_code_fraction",
                "code_perplexity",
                "unique_code_count_mean",
                "unique_code_count_std",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    _write_markdown_report(summary_md_path, summary)

    print(f"Saved token-mask prior sweep to {outdir}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Summary markdown: {summary_md_path}")


if __name__ == "__main__":
    main()

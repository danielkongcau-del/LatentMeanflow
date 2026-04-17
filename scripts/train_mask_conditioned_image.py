import argparse
import os
import sys
from pathlib import Path

from _launch_utils import parse_bool_arg, run_managed_subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_LATENT_FLOW = REPO_ROOT / "scripts" / "train_latent_meanflow.py"
DEFAULT_CONFIGS = {
    "fm": REPO_ROOT / "configs" / "latent_fm_mask2image_unet.yaml",
    "meanflow": REPO_ROOT / "configs" / "latent_meanflow_mask2image_unet.yaml",
    "alphaflow": REPO_ROOT / "configs" / "ablations" / "latent_alphaflow_mask2image_unet_fullres_pyramid_boundary_encoder.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the project-layer mask-conditioned image route p(image | semantic mask). "
            "This wrapper keeps the unconditional paired route separate."
        )
    )
    parser.add_argument("--objective", choices=["fm", "meanflow", "alphaflow"], default="alphaflow")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--tokenizer-config", type=Path, default=None)
    parser.add_argument("--tokenizer-ckpt", type=Path, default=None)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--scale-lr",
        type=parse_bool_arg,
        nargs="?",
        const=True,
        default=True,
        help=(
            "Pass an explicit scale_lr value through to train_latent_meanflow.py. "
            "true reproduces the legacy effective-lr scaling by accumulate_grad_batches * ngpu * batch_size."
        ),
    )
    parser.add_argument("--run-test", action="store_true")
    parser.add_argument("--enable-image-logger", action="store_true")
    parser.add_argument("--image-log-frequency", type=int, default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Extra OmegaConf dotlist override. Repeat as needed.",
    )
    return parser.parse_args()


def build_env():
    return os.environ.copy()


def resolve_config(args):
    if args.config is not None:
        return args.config
    return DEFAULT_CONFIGS[args.objective]


def build_command(args, config_path):
    cmd = [
        sys.executable,
        str(TRAIN_LATENT_FLOW),
        "--objective",
        str(args.objective),
        "--config",
        str(config_path.resolve()),
    ]
    if args.tokenizer_config is not None:
        cmd.extend(["--tokenizer-config", str(args.tokenizer_config.resolve())])
    if args.tokenizer_ckpt is not None:
        cmd.extend(["--tokenizer-ckpt", str(args.tokenizer_ckpt.resolve())])
    if args.gpus is not None:
        cmd.extend(["--gpus", str(args.gpus)])
    if args.max_epochs is not None:
        cmd.extend(["--max-epochs", str(args.max_epochs)])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.resume is not None:
        cmd.extend(["--resume", str(args.resume.resolve())])
    cmd.extend(["--scale-lr", str(bool(args.scale_lr)).lower()])
    if args.run_test:
        cmd.append("--run-test")
    if args.enable_image_logger:
        cmd.append("--enable-image-logger")
    if args.image_log_frequency is not None:
        cmd.extend(["--image-log-frequency", str(args.image_log_frequency)])
    for override in args.overrides:
        cmd.extend(["--set", override])
    return cmd


def main():
    args = parse_args()
    config_path = resolve_config(args)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cmd = build_command(args, config_path)
    print("Running:", " ".join(cmd))
    run_managed_subprocess(cmd, cwd=REPO_ROOT, env=build_env())


if __name__ == "__main__":
    main()

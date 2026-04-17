import argparse
import os
import sys
from pathlib import Path

from _launch_utils import normalize_gpus_arg, run_managed_subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
LDM_ROOT = REPO_ROOT / "third_party" / "latent-diffusion"
TAMING_ROOT = LDM_ROOT / "taming-transformers"
LDM_LAUNCHER = SCRIPT_ROOT / "launch_ldm_main.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "latent_diffusion_mask_prior_sit.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the project-layer unconditional semantic-mask diffusion baseline "
            "with a SiT-style transformer backbone."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpus", type=str, default=None, help='Examples: "0" or "0,1".')
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="Opt in to Trainer.test() after fit. Disabled by default for this baseline.",
    )
    parser.add_argument("--image-log-frequency", type=int, default=None)
    parser.add_argument("--enable-image-logger", action="store_true")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Extra OmegaConf dotlist override. Repeat as needed, without a leading --.",
    )
    return parser.parse_args()


def build_env():
    env = os.environ.copy()
    pythonpath = [str(REPO_ROOT), str(LDM_ROOT), str(TAMING_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def resolve_run_name(config_path):
    stem = config_path.stem.strip()
    return stem if stem else "latent_diffusion_mask_prior_sit"


def build_command(args):
    cmd = [sys.executable, str(LDM_LAUNCHER), "-t"]
    if args.resume is None:
        cmd.extend(["--name", resolve_run_name(args.config), "--base", str(args.config.resolve())])
    else:
        cmd.extend(["--base", str(args.config.resolve()), "--resume", str(args.resume.resolve())])
    if not args.run_test:
        cmd.append("--no-test")
    if args.gpus is not None:
        cmd.extend(["--gpus", normalize_gpus_arg(args.gpus)])
    if args.max_epochs is not None:
        cmd.extend(["--max_epochs", str(args.max_epochs)])
    if args.batch_size is not None:
        cmd.append(f"data.params.batch_size={args.batch_size}")
    if args.enable_image_logger:
        cmd.append("lightning.callbacks.image_logger.params.disabled=False")
    if args.image_log_frequency is not None:
        cmd.append("lightning.callbacks.image_logger.params.disabled=False")
        cmd.append(f"lightning.callbacks.image_logger.params.batch_frequency={args.image_log_frequency}")
    cmd.extend(args.overrides)
    return cmd


def main():
    args = parse_args()
    if not LDM_ROOT.exists():
        raise FileNotFoundError(f"latent-diffusion vendor directory not found: {LDM_ROOT}")
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    cmd = build_command(args)
    print("Running:", " ".join(cmd))
    run_managed_subprocess(cmd, cwd=REPO_ROOT, env=build_env())


if __name__ == "__main__":
    main()

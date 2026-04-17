import argparse
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

from _launch_utils import (
    count_devices_for_lr_scaling,
    normalize_gpus_arg,
    parse_bool_arg,
    resolve_resume_logdir,
    resume_run_matches_config,
    run_managed_subprocess,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
LDM_ROOT = REPO_ROOT / "third_party" / "latent-diffusion"
TAMING_ROOT = LDM_ROOT / "taming-transformers"
LDM_LAUNCHER = SCRIPT_ROOT / "launch_ldm_main.py"
DEFAULT_TOKENIZER_CONFIG = REPO_ROOT / "configs" / "autoencoder_semantic_pair_256.yaml"
DEFAULT_TOKENIZER_CKPT = REPO_ROOT / "logs" / "autoencoder" / "checkpoints" / "last.ckpt"
DEFAULT_CONFIGS = {
    "fm": REPO_ROOT / "configs" / "latent_fm_semantic_256.yaml",
    "meanflow": REPO_ROOT / "configs" / "latent_meanflow_semantic_256_unet.yaml",
    "alphaflow": REPO_ROOT / "configs" / "latent_alphaflow_semantic_256_unet.yaml",
}
RUN_NAMES = {
    "fm": "latent_fm",
    "meanflow": "latent_meanflow",
    "alphaflow": "latent_alphaflow",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train latent flow priors. Recommended default: U-Net AlphaFlow project baseline."
    )
    parser.add_argument("--objective", choices=["fm", "meanflow", "alphaflow"], default="alphaflow")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--tokenizer-config", type=Path, default=None)
    parser.add_argument("--tokenizer-ckpt", type=Path, default=None)
    parser.add_argument("--gpus", type=str, default=None, help='Examples: "0" or "0,1".')
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
            "Pass an explicit scale_lr value to the vendored launcher. "
            "true reproduces the legacy effective-lr scaling by accumulate_grad_batches * ngpu * batch_size."
        ),
    )
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="Opt in to Trainer.test() after fit. Disabled by default for project-layer trainers.",
    )
    parser.add_argument(
        "--allow-config-override",
        action="store_true",
        help="Dangerous: allow --config to override the saved resume config.",
    )
    parser.add_argument(
        "--force-tokenizer-config",
        action="store_true",
        help="Dangerous: inject tokenizer_config_path even during resume.",
    )
    parser.add_argument(
        "--force-tokenizer-ckpt",
        action="store_true",
        help="Dangerous: inject tokenizer_ckpt_path even during resume.",
    )
    parser.add_argument(
        "--allow-dotlist-override",
        action="store_true",
        help="Dangerous: allow --set and wrapper-managed dotlist overrides during resume.",
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


def resolve_tokenizer_ckpt(user_path):
    if user_path is not None:
        return user_path
    if DEFAULT_TOKENIZER_CKPT.exists():
        return DEFAULT_TOKENIZER_CKPT
    raise FileNotFoundError(
        "Tokenizer checkpoint not found. Pass --tokenizer-ckpt or train the tokenizer first."
    )


def resolve_config(args):
    if args.config is not None:
        return args.config
    return DEFAULT_CONFIGS[args.objective]


def load_train_config(config_path):
    return OmegaConf.load(config_path)


def _resolve_repo_path(value):
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_tokenizer_config_path(args, config):
    if args.tokenizer_config is not None:
        return args.tokenizer_config
    if config is None:
        return DEFAULT_TOKENIZER_CONFIG
    config_value = OmegaConf.select(config, "model.params.tokenizer_config_path")
    resolved = _resolve_repo_path(config_value)
    if resolved is not None:
        return resolved
    return DEFAULT_TOKENIZER_CONFIG


def _find_latest_ckpt_for_config(tokenizer_config_path):
    run_tag = tokenizer_config_path.stem.lower()
    candidates = sorted(
        REPO_ROOT.glob("logs/**/checkpoints/last.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        run_dir_name = candidate.parent.parent.name.lower()
        if run_dir_name == run_tag or run_dir_name.endswith(f"_{run_tag}"):
            return candidate
    return None


def resolve_tokenizer_ckpt_path(args, config, tokenizer_config_path):
    if args.tokenizer_ckpt is not None:
        return args.tokenizer_ckpt

    config_ckpt = None if config is None else _resolve_repo_path(
        OmegaConf.select(config, "model.params.tokenizer_ckpt_path")
    )
    if config_ckpt is not None and config_ckpt.exists():
        return config_ckpt

    latest_ckpt = _find_latest_ckpt_for_config(tokenizer_config_path)
    if latest_ckpt is not None:
        return latest_ckpt

    if config_ckpt is not None:
        return config_ckpt
    return resolve_tokenizer_ckpt(None)


def is_resume_mode(args):
    return args.resume is not None


def resolve_run_name(args, config_path):
    stem = config_path.stem.strip()
    if stem:
        return stem
    return RUN_NAMES[args.objective]


def should_pass_run_name(args):
    # Vendored latent-diffusion forbids combining --name with --resume.
    # Fresh runs keep the config-stem naming behavior; resume runs must omit --name.
    return args.resume is None


def should_pass_base_config(args):
    if not is_resume_mode(args):
        return True
    return bool(args.allow_config_override)


def should_inject_tokenizer_config(args):
    if not is_resume_mode(args):
        return True
    return bool(args.force_tokenizer_config)


def should_inject_tokenizer_ckpt(args):
    if not is_resume_mode(args):
        return True
    return bool(args.force_tokenizer_ckpt)


def should_pass_dotlist_overrides(args):
    if not is_resume_mode(args):
        return True
    return bool(args.allow_dotlist_override)


def has_wrapper_dotlist_flags(args):
    return (
        args.batch_size is not None
        or args.enable_image_logger
        or args.image_log_frequency is not None
    )


def has_any_dotlist_style_override(args):
    return bool(args.overrides) or has_wrapper_dotlist_flags(args)


def validate_resume_request(args, config_path=None, resume_logdir=None):
    if not is_resume_mode(args):
        return

    if args.allow_config_override and args.config is None:
        raise ValueError("--allow-config-override requires an explicit --config.")
    if args.allow_dotlist_override and not has_any_dotlist_style_override(args):
        raise ValueError(
            "--allow-dotlist-override requires at least one dotlist-style override "
            "(--set, --batch-size, --enable-image-logger, or --image-log-frequency)."
        )
    if has_any_dotlist_style_override(args) and not args.allow_dotlist_override:
        raise ValueError(
            "Resume with dotlist-style overrides is not allowed by default. Safe resume should pass only "
            "--resume. This includes --set, --batch-size, --enable-image-logger, and "
            "--image-log-frequency. If you truly want to inject dotlist overrides into a resumed run, add "
            "--allow-dotlist-override explicitly."
        )

    if args.config is not None and not args.allow_config_override:
        if config_path is None or resume_logdir is None:
            raise ValueError("Internal error: config and resume_logdir must be resolved before validation.")
        if not resume_run_matches_config(resume_logdir, config_path):
            raise ValueError(
                f"Resume/config mismatch: resume run '{resume_logdir.name}' does not match config "
                f"'{Path(config_path).name}'. Safe resume should pass only --resume. If you truly want "
                "to override the saved config, add --allow-config-override explicitly."
            )


def build_command(args, config_path, tokenizer_config_path, tokenizer_ckpt):
    cmd = [
        sys.executable,
        str(LDM_LAUNCHER),
        "-t",
    ]
    if not args.run_test:
        cmd.append("--no-test")
    if should_pass_base_config(args):
        if config_path is None:
            raise ValueError("config_path is required when building a fresh run or overriding resume config.")
        cmd.extend(["--base", str(config_path.resolve())])
    if should_pass_run_name(args):
        if config_path is None:
            raise ValueError("config_path is required when naming a fresh run.")
        cmd[3:3] = ["--name", resolve_run_name(args, config_path)]
    if args.resume:
        cmd.extend(["--resume", str(args.resume.resolve())])
    cmd.extend(["--scale_lr", str(bool(args.scale_lr)).lower()])
    if args.gpus:
        cmd.extend(["--gpus", normalize_gpus_arg(args.gpus)])
    if args.max_epochs is not None:
        cmd.extend(["--max_epochs", str(args.max_epochs)])
    if args.batch_size is not None and should_pass_dotlist_overrides(args):
        cmd.append(f"data.params.batch_size={args.batch_size}")
    if args.enable_image_logger and should_pass_dotlist_overrides(args):
        cmd.append("lightning.callbacks.image_logger.params.disabled=False")
    if args.image_log_frequency is not None and should_pass_dotlist_overrides(args):
        cmd.append("lightning.callbacks.image_logger.params.disabled=False")
        cmd.append(f"lightning.callbacks.image_logger.params.batch_frequency={args.image_log_frequency}")
    if should_inject_tokenizer_config(args):
        cmd.append(f"model.params.tokenizer_config_path={tokenizer_config_path.resolve()}")
    if should_inject_tokenizer_ckpt(args):
        if tokenizer_ckpt is None:
            raise ValueError("tokenizer_ckpt is required when tokenizer ckpt override injection is enabled.")
        cmd.append(f"model.params.tokenizer_ckpt_path={tokenizer_ckpt.resolve()}")
    if should_pass_dotlist_overrides(args):
        cmd.extend(args.overrides)
    return cmd


def print_learning_rate_summary(*, config, config_path, args, note_prefix="[train_latent_meanflow]"):
    if config is None or config_path is None:
        print(
            f"{note_prefix} scale_lr={bool(args.scale_lr)}. "
            "Safe resume is reusing the saved run config, so effective_lr is determined from that saved config."
        )
        return

    batch_size = int(args.batch_size) if args.batch_size is not None else int(config.data.params.batch_size)
    accumulate_grad_batches = int(
        OmegaConf.select(config, "lightning.trainer.accumulate_grad_batches", default=1)
    )
    ngpu = count_devices_for_lr_scaling(args.gpus)
    base_lr = float(config.model.base_learning_rate)
    effective_lr = (
        float(base_lr)
        if not bool(args.scale_lr)
        else float(base_lr) * float(batch_size) * float(accumulate_grad_batches) * float(ngpu)
    )
    print(
        f"{note_prefix} base_lr={base_lr:.2e}, scale_lr={bool(args.scale_lr)}, "
        f"batch_size={batch_size}, accumulate_grad_batches={accumulate_grad_batches}, "
        f"num_devices={ngpu}, effective_lr={effective_lr:.2e}, config={config_path.resolve()}"
    )


def main():
    args = parse_args()
    if not LDM_ROOT.exists():
        raise FileNotFoundError(f"latent-diffusion vendor directory not found: {LDM_ROOT}")

    resume_logdir = resolve_resume_logdir(args.resume) if is_resume_mode(args) else None

    config_path = None
    if args.config is not None or not is_resume_mode(args):
        config_path = resolve_config(args)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

    validate_resume_request(args, config_path=config_path, resume_logdir=resume_logdir)

    config = load_train_config(config_path) if config_path is not None else None
    tokenizer_config_path = (
        resolve_tokenizer_config_path(args, config) if should_inject_tokenizer_config(args) else None
    )
    if tokenizer_config_path is not None and not tokenizer_config_path.exists():
        raise FileNotFoundError(f"Tokenizer config file not found: {tokenizer_config_path}")

    tokenizer_ckpt = (
        resolve_tokenizer_ckpt_path(args, config, tokenizer_config_path)
        if should_inject_tokenizer_ckpt(args)
        else None
    )
    if tokenizer_ckpt is not None and not tokenizer_ckpt.exists():
        raise FileNotFoundError(
            f"Tokenizer checkpoint not found: {tokenizer_ckpt}. "
            "Pass --tokenizer-ckpt explicitly or train the referenced tokenizer first."
        )
    print_learning_rate_summary(config=config, config_path=config_path, args=args)
    cmd = build_command(args, config_path, tokenizer_config_path, tokenizer_ckpt)
    print("Running:", " ".join(cmd))
    run_managed_subprocess(cmd, cwd=REPO_ROOT, env=build_env())


if __name__ == "__main__":
    main()

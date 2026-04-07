import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import train_latent_fm
import train_latent_meanflow
import train_semantic_autoencoder


def assert_in(token, cmd, context):
    if token not in cmd:
        raise AssertionError(f"Expected '{token}' in {context}: {cmd}")


def assert_not_in(token, cmd, context):
    if token in cmd:
        raise AssertionError(f"Did not expect '{token}' in {context}: {cmd}")


def assert_uses_project_launcher(cmd, context):
    if len(cmd) < 2 or not str(cmd[1]).endswith("launch_ldm_main.py"):
        raise AssertionError(f"Expected project launcher in {context}: {cmd}")


def assert_gpu_arg(cmd, expected, context):
    if "--gpus" not in cmd:
        raise AssertionError(f"Expected --gpus in {context}: {cmd}")
    idx = cmd.index("--gpus")
    if cmd[idx + 1] != expected:
        raise AssertionError(f"Expected --gpus {expected!r} in {context}, got {cmd[idx + 1]!r}: {cmd}")


def main():
    tokenizer_config = REPO_ROOT / "configs" / "autoencoder_semantic_pair_256.yaml"
    tokenizer_ckpt = REPO_ROOT / "logs" / "autoencoder" / "checkpoints" / "last.ckpt"

    semantic_args = argparse.Namespace(
        config=REPO_ROOT / "configs" / "semantic_tokenizer_tiny_256.yaml",
        gpus="2",
        max_epochs=1,
        batch_size=None,
        resume=None,
        run_test=False,
        image_log_frequency=None,
        enable_image_logger=False,
        overrides=[],
    )
    semantic_cmd = train_semantic_autoencoder.build_command(semantic_args)
    assert_uses_project_launcher(semantic_cmd, "semantic autoencoder default command")
    assert_gpu_arg(semantic_cmd, "0,1", "semantic autoencoder default command")
    assert_in("--no-test", semantic_cmd, "semantic autoencoder default command")
    semantic_args.run_test = True
    semantic_cmd_with_test = train_semantic_autoencoder.build_command(semantic_args)
    assert_not_in("--no-test", semantic_cmd_with_test, "semantic autoencoder run-test command")

    fm_args = argparse.Namespace(
        config=REPO_ROOT / "configs" / "latent_fm_semantic_256_tiny.yaml",
        tokenizer_config=tokenizer_config,
        tokenizer_ckpt=tokenizer_ckpt,
        gpus="2",
        max_epochs=1,
        batch_size=None,
        resume=None,
        run_test=False,
        image_log_frequency=None,
        enable_image_logger=False,
        overrides=[],
    )
    fm_cmd = train_latent_fm.build_command(fm_args, tokenizer_ckpt)
    assert_uses_project_launcher(fm_cmd, "latent fm default command")
    assert_gpu_arg(fm_cmd, "0,1", "latent fm default command")
    assert_in("--no-test", fm_cmd, "latent fm default command")
    fm_args.run_test = True
    fm_cmd_with_test = train_latent_fm.build_command(fm_args, tokenizer_ckpt)
    assert_not_in("--no-test", fm_cmd_with_test, "latent fm run-test command")

    meanflow_args = argparse.Namespace(
        objective="meanflow",
        config=REPO_ROOT / "configs" / "latent_meanflow_semantic_256.yaml",
        tokenizer_config=tokenizer_config,
        tokenizer_ckpt=tokenizer_ckpt,
        gpus="2",
        max_epochs=1,
        batch_size=None,
        resume=None,
        run_test=False,
        allow_config_override=False,
        force_tokenizer_config=False,
        force_tokenizer_ckpt=False,
        allow_dotlist_override=False,
        image_log_frequency=None,
        enable_image_logger=False,
        overrides=[],
    )
    meanflow_cmd = train_latent_meanflow.build_command(meanflow_args, meanflow_args.config, tokenizer_ckpt)
    assert_uses_project_launcher(meanflow_cmd, "latent meanflow default command")
    assert_gpu_arg(meanflow_cmd, "0,1", "latent meanflow default command")
    assert_in("--no-test", meanflow_cmd, "latent meanflow default command")
    meanflow_args.run_test = True
    meanflow_cmd_with_test = train_latent_meanflow.build_command(
        meanflow_args,
        meanflow_args.config,
        tokenizer_ckpt,
    )
    assert_not_in("--no-test", meanflow_cmd_with_test, "latent meanflow run-test command")

    print("Project-layer training wrappers disable post-fit test by default")
    print("Passing --run-test opts back into Trainer.test() explicitly")


if __name__ == "__main__":
    main()

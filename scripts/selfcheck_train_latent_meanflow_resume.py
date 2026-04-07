import argparse
from pathlib import Path

from train_latent_meanflow import build_command


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_args(resume=None):
    return argparse.Namespace(
        objective="meanflow",
        config=REPO_ROOT / "configs" / "latent_meanflow_semantic_256.yaml",
        tokenizer_config=REPO_ROOT / "configs" / "autoencoder_semantic_pair_256.yaml",
        tokenizer_ckpt=REPO_ROOT / "logs" / "autoencoder" / "checkpoints" / "last.ckpt",
        gpus="0",
        max_epochs=None,
        batch_size=None,
        resume=resume,
        image_log_frequency=None,
        enable_image_logger=False,
        overrides=[],
    )


def assert_contains_once(cmd, token):
    count = cmd.count(token)
    if count != 1:
        raise AssertionError(f"Expected token '{token}' exactly once, got {count}: {cmd}")


def main():
    config_path = REPO_ROOT / "configs" / "latent_meanflow_semantic_256.yaml"
    tokenizer_ckpt = REPO_ROOT / "logs" / "autoencoder" / "checkpoints" / "last.ckpt"

    fresh_cmd = build_command(make_args(resume=None), config_path, tokenizer_ckpt)
    assert_contains_once(fresh_cmd, "--name")
    if "--resume" in fresh_cmd:
        raise AssertionError(f"Fresh run command should not contain --resume: {fresh_cmd}")

    name_idx = fresh_cmd.index("--name")
    expected_name = config_path.stem
    if fresh_cmd[name_idx + 1] != expected_name:
        raise AssertionError(
            f"Fresh run should use config stem '{expected_name}', got '{fresh_cmd[name_idx + 1]}'"
        )

    resume_path = REPO_ROOT / "logs" / "2026-04-07T12-00-00_latent_meanflow_semantic_256" / "checkpoints" / "last.ckpt"
    resume_cmd = build_command(make_args(resume=resume_path), config_path, tokenizer_ckpt)
    if "--name" in resume_cmd:
        raise AssertionError(f"Resume command must omit --name: {resume_cmd}")
    assert_contains_once(resume_cmd, "--resume")

    resume_idx = resume_cmd.index("--resume")
    if Path(resume_cmd[resume_idx + 1]) != resume_path.resolve():
        raise AssertionError(
            f"Resume command should forward the resolved checkpoint path '{resume_path.resolve()}', "
            f"got '{resume_cmd[resume_idx + 1]}'"
        )

    print("fresh command keeps --name and omits --resume")
    print("resume command keeps --resume and omits --name")


if __name__ == "__main__":
    main()

import argparse
import tempfile
from pathlib import Path

from train_latent_meanflow import (
    build_command,
    resolve_resume_logdir,
    validate_resume_request,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_args(
    resume=None,
    config=None,
    objective="meanflow",
    tokenizer_config=None,
    tokenizer_ckpt=None,
    allow_config_override=False,
    allow_dotlist_override=False,
    force_tokenizer_config=False,
    force_tokenizer_ckpt=False,
    overrides=None,
):
    return argparse.Namespace(
        objective=objective,
        config=config,
        tokenizer_config=tokenizer_config or (REPO_ROOT / "configs" / "autoencoder_semantic_pair_256.yaml"),
        tokenizer_ckpt=tokenizer_ckpt,
        gpus="0",
        max_epochs=None,
        batch_size=None,
        resume=resume,
        allow_config_override=allow_config_override,
        allow_dotlist_override=allow_dotlist_override,
        force_tokenizer_config=force_tokenizer_config,
        force_tokenizer_ckpt=force_tokenizer_ckpt,
        image_log_frequency=None,
        enable_image_logger=False,
        overrides=list(overrides or []),
    )


def assert_contains_once(cmd, token):
    count = cmd.count(token)
    if count != 1:
        raise AssertionError(f"Expected token '{token}' exactly once, got {count}: {cmd}")


def main():
    config_path = REPO_ROOT / "configs" / "latent_meanflow_semantic_256.yaml"
    tokenizer_ckpt = REPO_ROOT / "logs" / "autoencoder" / "checkpoints" / "last.ckpt"
    tokenizer_config = REPO_ROOT / "configs" / "autoencoder_semantic_pair_256.yaml"

    fresh_cmd = build_command(
        make_args(
            resume=None,
            config=config_path,
            tokenizer_config=tokenizer_config,
            tokenizer_ckpt=tokenizer_ckpt,
        ),
        config_path,
        tokenizer_ckpt,
    )
    assert_contains_once(fresh_cmd, "--name")
    assert_contains_once(fresh_cmd, "--base")
    if "--resume" in fresh_cmd:
        raise AssertionError(f"Fresh run command should not contain --resume: {fresh_cmd}")
    if not any(item.startswith("--model.params.tokenizer_config_path=") for item in fresh_cmd):
        raise AssertionError(f"Fresh run should inject tokenizer_config_path: {fresh_cmd}")
    if not any(item.startswith("--model.params.tokenizer_ckpt_path=") for item in fresh_cmd):
        raise AssertionError(f"Fresh run should inject tokenizer_ckpt_path: {fresh_cmd}")

    name_idx = fresh_cmd.index("--name")
    expected_name = config_path.stem
    if fresh_cmd[name_idx + 1] != expected_name:
        raise AssertionError(
            f"Fresh run should use config stem '{expected_name}', got '{fresh_cmd[name_idx + 1]}'"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        matching_logdir = tmp_root / "2026-04-07T12-00-00_latent_meanflow_semantic_256"
        mismatched_logdir = tmp_root / "2026-04-07T12-00-00_latent_alphaflow_semantic_256"
        for logdir in (matching_logdir, mismatched_logdir):
            (logdir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (logdir / "configs").mkdir(parents=True, exist_ok=True)
            (logdir / "configs" / "000000-project.yaml").write_text("model:\n  target: dummy\n", encoding="utf-8")
            (logdir / "configs" / "000000-lightning.yaml").write_text("lightning:\n  trainer: {}\n", encoding="utf-8")
            (logdir / "checkpoints" / "last.ckpt").write_text("stub", encoding="utf-8")

        resume_path = matching_logdir / "checkpoints" / "last.ckpt"
        resume_logdir = resolve_resume_logdir(resume_path)
        validate_resume_request(
            make_args(resume=resume_path),
            config_path=None,
            resume_logdir=resume_logdir,
        )
        resume_cmd = build_command(
            make_args(resume=resume_path),
            config_path=None,
            tokenizer_ckpt=None,
        )
        if "--name" in resume_cmd:
            raise AssertionError(f"Resume command must omit --name: {resume_cmd}")
        if "--base" in resume_cmd:
            raise AssertionError(f"Bare resume should not inject --base: {resume_cmd}")
        assert_contains_once(resume_cmd, "--resume")
        if any(item.startswith("--model.params.tokenizer_config_path=") for item in resume_cmd):
            raise AssertionError(f"Bare resume should not inject tokenizer_config_path: {resume_cmd}")
        if any(item.startswith("--model.params.tokenizer_ckpt_path=") for item in resume_cmd):
            raise AssertionError(f"Bare resume should not inject tokenizer_ckpt_path: {resume_cmd}")

        try:
            validate_resume_request(
                make_args(
                    resume=resume_path,
                    overrides=["--model.params.foo=bar"],
                ),
                config_path=None,
                resume_logdir=resume_logdir,
            )
        except ValueError as exc:
            if "--allow-dotlist-override" not in str(exc):
                raise AssertionError(f"Unexpected dotlist error message: {exc}") from exc
        else:
            raise AssertionError("Bare resume with --set should fail by default.")

        matched_resume_args = make_args(
            resume=resume_path,
            config=config_path,
        )
        validate_resume_request(
            matched_resume_args,
            config_path=config_path,
            resume_logdir=resume_logdir,
        )
        matched_resume_cmd = build_command(
            matched_resume_args,
            config_path=config_path,
            tokenizer_ckpt=None,
        )
        if "--base" in matched_resume_cmd:
            raise AssertionError(f"Resume with matching config should still avoid injecting --base: {matched_resume_cmd}")

        mismatched_resume_path = mismatched_logdir / "checkpoints" / "last.ckpt"
        mismatched_resume_logdir = resolve_resume_logdir(mismatched_resume_path)
        try:
            validate_resume_request(
                make_args(
                    resume=mismatched_resume_path,
                    config=config_path,
                ),
                config_path=config_path,
                resume_logdir=mismatched_resume_logdir,
            )
        except ValueError as exc:
            if "--allow-config-override" not in str(exc):
                raise AssertionError(f"Unexpected mismatch error message: {exc}") from exc
        else:
            raise AssertionError("Resume with mismatched config should fail by default.")

        forced_resume_cmd = build_command(
            make_args(
                resume=resume_path,
                config=config_path,
                tokenizer_config=tokenizer_config,
                tokenizer_ckpt=tokenizer_ckpt,
                allow_config_override=True,
                force_tokenizer_config=True,
                force_tokenizer_ckpt=True,
            ),
            config_path=config_path,
            tokenizer_ckpt=tokenizer_ckpt,
        )
        assert_contains_once(forced_resume_cmd, "--base")
        if not any(item.startswith("--model.params.tokenizer_config_path=") for item in forced_resume_cmd):
            raise AssertionError(f"Forced resume should inject tokenizer_config_path: {forced_resume_cmd}")
        if not any(item.startswith("--model.params.tokenizer_ckpt_path=") for item in forced_resume_cmd):
            raise AssertionError(f"Forced resume should inject tokenizer_ckpt_path: {forced_resume_cmd}")

        forced_dotlist_args = make_args(
            resume=resume_path,
            allow_dotlist_override=True,
            overrides=["--model.params.foo=bar"],
        )
        validate_resume_request(
            forced_dotlist_args,
            config_path=None,
            resume_logdir=resume_logdir,
        )
        forced_dotlist_cmd = build_command(
            forced_dotlist_args,
            config_path=None,
            tokenizer_ckpt=None,
        )
        if "--model.params.foo=bar" not in forced_dotlist_cmd:
            raise AssertionError(f"Resume with --allow-dotlist-override should keep dotlist overrides: {forced_dotlist_cmd}")

    print("fresh command keeps --name and omits --resume")
    print("bare resume omits --name, --base, and tokenizer overrides")
    print("bare resume rejects --set unless --allow-dotlist-override is used")
    print("resume with matching config is allowed without injecting a new base config")
    print("resume with mismatched config fails unless --allow-config-override is used")
    print("resume injects tokenizer overrides only when the force flags are enabled")


if __name__ == "__main__":
    main()

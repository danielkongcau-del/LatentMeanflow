import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import train_latent_fm
import train_latent_meanflow
import train_semantic_autoencoder


def assert_in(token, seq, label):
    if token not in seq:
        raise AssertionError(f"Expected '{token}' in {label}: {seq}")


def assert_no_prefixed_dotlists(seq, label):
    bad = [
        item for item in seq
        if item.startswith("--data.")
        or item.startswith("--model.")
        or item.startswith("--lightning.")
    ]
    if bad:
        raise AssertionError(f"{label} contains malformed dotlist overrides: {bad}")


def main():
    tokenizer_config = REPO_ROOT / "configs" / "autoencoder_semantic_pair_256.yaml"
    tokenizer_ckpt = REPO_ROOT / "logs" / "autoencoder" / "checkpoints" / "last.ckpt"

    semantic_args = argparse.Namespace(
        config=REPO_ROOT / "configs" / "autoencoder_semantic_pair_24gb_256.yaml",
        gpus="1",
        max_epochs=1,
        batch_size=4,
        resume=None,
        run_test=False,
        image_log_frequency=50,
        enable_image_logger=False,
        overrides=["model.params.foo=bar"],
    )
    semantic_cmd = train_semantic_autoencoder.build_command(semantic_args)
    assert_in("data.params.batch_size=4", semantic_cmd, "semantic tokenizer command")
    assert_in("lightning.callbacks.image_logger.params.disabled=False", semantic_cmd, "semantic tokenizer command")
    assert_in("lightning.callbacks.image_logger.params.batch_frequency=50", semantic_cmd, "semantic tokenizer command")
    assert_in("model.params.foo=bar", semantic_cmd, "semantic tokenizer command")
    assert_no_prefixed_dotlists(semantic_cmd, "semantic tokenizer command")

    fm_args = argparse.Namespace(
        config=REPO_ROOT / "configs" / "latent_fm_semantic_256_tiny.yaml",
        tokenizer_config=tokenizer_config,
        tokenizer_ckpt=tokenizer_ckpt,
        gpus="1",
        max_epochs=1,
        batch_size=4,
        resume=None,
        run_test=False,
        image_log_frequency=50,
        enable_image_logger=False,
        overrides=["model.params.foo=bar"],
    )
    fm_cmd = train_latent_fm.build_command(fm_args, tokenizer_ckpt)
    assert_in("data.params.batch_size=4", fm_cmd, "latent fm command")
    assert_in("lightning.callbacks.image_logger.params.disabled=False", fm_cmd, "latent fm command")
    assert_in("lightning.callbacks.image_logger.params.batch_frequency=50", fm_cmd, "latent fm command")
    assert_in(f"model.params.tokenizer_config_path={tokenizer_config.resolve()}", fm_cmd, "latent fm command")
    assert_in(f"model.params.tokenizer_ckpt_path={tokenizer_ckpt.resolve()}", fm_cmd, "latent fm command")
    assert_in("model.params.foo=bar", fm_cmd, "latent fm command")
    assert_no_prefixed_dotlists(fm_cmd, "latent fm command")

    meanflow_args = argparse.Namespace(
        objective="meanflow",
        config=REPO_ROOT / "configs" / "latent_meanflow_semantic_256.yaml",
        tokenizer_config=tokenizer_config,
        tokenizer_ckpt=tokenizer_ckpt,
        gpus="1",
        max_epochs=1,
        batch_size=4,
        resume=None,
        run_test=False,
        allow_config_override=False,
        force_tokenizer_config=False,
        force_tokenizer_ckpt=False,
        allow_dotlist_override=False,
        image_log_frequency=50,
        enable_image_logger=False,
        overrides=["model.params.foo=bar"],
    )
    meanflow_cmd = train_latent_meanflow.build_command(meanflow_args, meanflow_args.config, tokenizer_ckpt)
    assert_in("data.params.batch_size=4", meanflow_cmd, "latent meanflow command")
    assert_in("lightning.callbacks.image_logger.params.disabled=False", meanflow_cmd, "latent meanflow command")
    assert_in("lightning.callbacks.image_logger.params.batch_frequency=50", meanflow_cmd, "latent meanflow command")
    assert_in(f"model.params.tokenizer_config_path={tokenizer_config.resolve()}", meanflow_cmd, "latent meanflow command")
    assert_in(f"model.params.tokenizer_ckpt_path={tokenizer_ckpt.resolve()}", meanflow_cmd, "latent meanflow command")
    assert_in("model.params.foo=bar", meanflow_cmd, "latent meanflow command")
    assert_no_prefixed_dotlists(meanflow_cmd, "latent meanflow command")

    merged = OmegaConf.merge(
        OmegaConf.load(REPO_ROOT / "configs" / "autoencoder_semantic_pair_24gb_256.yaml"),
        OmegaConf.from_dotlist([
            "data.params.batch_size=4",
            "lightning.callbacks.image_logger.params.batch_frequency=50",
        ]),
    )
    if merged.data.params.batch_size != 4:
        raise AssertionError(f"Expected merged batch size 4, got {merged.data.params.batch_size}")
    if merged.lightning.callbacks.image_logger.params.batch_frequency != 50:
        raise AssertionError(
            "Expected merged image logger frequency 50, "
            f"got {merged.lightning.callbacks.image_logger.params.batch_frequency}"
        )

    print("Project-layer wrappers emit OmegaConf dotlist overrides without a leading --")
    print("Batch-size and image-logger overrides now merge into the intended config keys")


if __name__ == "__main__":
    main()

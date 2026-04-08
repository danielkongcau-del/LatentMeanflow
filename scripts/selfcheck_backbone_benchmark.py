import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
LDM_ROOT = REPO_ROOT / "third_party" / "latent-diffusion"
TAMING_ROOT = LDM_ROOT / "taming-transformers"
for path in (REPO_ROOT, LDM_ROOT, TAMING_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ldm.util import instantiate_from_config


def _write_temp_tokenizer(temp_root):
    tokenizer_config = OmegaConf.load(REPO_ROOT / "configs" / "autoencoder_semantic_pair_256.yaml")
    tokenizer_model = instantiate_from_config(tokenizer_config.model)

    config_path = temp_root / "autoencoder_semantic_pair_256.yaml"
    ckpt_path = temp_root / "autoencoder" / "checkpoints" / "last.ckpt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(tokenizer_config, config_path)
    torch.save({"state_dict": tokenizer_model.state_dict()}, ckpt_path)
    return config_path, ckpt_path


def _write_temp_flow_run(temp_root, tokenizer_config_path, tokenizer_ckpt_path):
    config_stem = "latent_meanflow_semantic_256_unet"
    flow_config = OmegaConf.load(REPO_ROOT / "configs" / f"{config_stem}.yaml")
    flow_config.model.params.tokenizer_config_path = str(tokenizer_config_path)
    flow_config.model.params.tokenizer_ckpt_path = str(tokenizer_ckpt_path)

    config_path = temp_root / f"{config_stem}.yaml"
    OmegaConf.save(flow_config, config_path)

    model = instantiate_from_config(flow_config.model)
    run_dir = temp_root / "logs" / f"1970-01-01T00-00-00_{config_stem}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "epoch=000001.ckpt"
    last_ckpt_path = ckpt_dir / "last.ckpt"

    base_state = {"state_dict": model.state_dict()}
    torch.save(base_state, best_ckpt_path)
    callback_state = {
        "callbacks": {
            "ModelCheckpoint{monitor='val/base_error_mean'}": {
                "monitor": "val/base_error_mean",
                "best_model_score": torch.tensor(0.1234),
                "best_model_path": str(best_ckpt_path),
            }
        }
    }
    torch.save({**base_state, **callback_state}, last_ckpt_path)
    return config_path, run_dir, best_ckpt_path


def main():
    temp_parent = REPO_ROOT / "outputs" / "_tmp_selfcheck"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_parent) as temp_dir:
        temp_root = Path(temp_dir)
        tokenizer_config_path, tokenizer_ckpt_path = _write_temp_tokenizer(temp_root)
        flow_config_path, run_dir, best_ckpt_path = _write_temp_flow_run(
            temp_root,
            tokenizer_config_path,
            tokenizer_ckpt_path,
        )

        find_command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "find_checkpoint.py"),
            "--run-dir",
            str(run_dir),
            "--selection",
            "best",
            "--monitor",
            "val/base_error_mean",
        ]
        resolved_best = subprocess.check_output(find_command, cwd=REPO_ROOT, text=True).strip()
        if Path(resolved_best).resolve() != best_ckpt_path.resolve():
            raise AssertionError(f"find_checkpoint.py returned {resolved_best}, expected {best_ckpt_path}")

        outdir = temp_root / "outputs" / "benchmark"
        eval_command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "eval_backbone_nfe_sweep.py"),
            "--config",
            str(flow_config_path),
            "--ckpt",
            str(best_ckpt_path),
            "--outdir",
            str(outdir),
            "--n-samples",
            "2",
            "--batch-size",
            "1",
            "--nfe-values",
            "4",
            "2",
            "1",
            "--seed",
            "23",
        ]
        subprocess.run(eval_command, check=True, cwd=REPO_ROOT)

        summary = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
        expected_nfe_values = [4, 2, 1]
        if summary["nfe_values"] != expected_nfe_values:
            raise AssertionError(f"Unexpected nfe_values: {summary['nfe_values']}")
        for result in summary["results"]:
            if result["image_count"] != 2 or result["mask_raw_count"] != 2 or result["mask_color_count"] != 2 or result["overlay_count"] != 2:
                raise AssertionError(f"Incomplete sweep result: {result}")
        print(f"backbone benchmark selfcheck ok: {outdir}")

    shutil.rmtree(temp_parent, ignore_errors=True)


if __name__ == "__main__":
    main()

import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
TAMING_DIR = ROOT / "taming-transformers"
if TAMING_DIR.exists():
    sys.path.insert(0, str(TAMING_DIR))
CONFIG = "configs/mask_image/ldm_4ch_256.yaml"

# Set to an explicit AE checkpoint path to skip auto-discovery.
AE_CKPT = "/root/autodl-tmp/latent-diffusion/logs/autoencoder/checkpoints/last.ckpt"

# Set to None or "" to run on CPU.
GPUS = "1"

# Training epochs.
MAX_EPOCHS = 3000

# Batch size override (set to None to use config default).
BATCH_SIZE = 4

# Set to a logdir or checkpoint path to resume; leave empty to train from scratch.
RESUME_PATH = "/root/autodl-tmp/latent-diffusion/logs/ldm/checkpoints/last.ckpt"



def find_latest_ae_ckpt(logs_dir):
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        return None
    candidates = []
    for ckpt in logs_path.rglob("checkpoints/last.ckpt"):
        run_dir = ckpt.parents[1]
        cfg_dir = run_dir / "configs"
        if not cfg_dir.exists():
            continue
        cfg_files = sorted(cfg_dir.glob("*-project.yaml"), reverse=True)
        if not cfg_files:
            continue
        cfg = OmegaConf.load(cfg_files[0])
        target = str(cfg.get("model", {}).get("target", ""))
        if "AutoencoderKL" in target:
            candidates.append((ckpt.stat().st_mtime, ckpt))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def build_command(ckpt_path):
    cmd = [sys.executable, "main.py", "-t", "--base", CONFIG]
    if RESUME_PATH:
        cmd.extend(["--resume", RESUME_PATH])
    cmd.append(f"--model.params.first_stage_config.params.ckpt_path={ckpt_path}")
    if GPUS:
        cmd.extend(["--gpus", GPUS])
    if MAX_EPOCHS is not None:
        cmd.extend(["--max_epochs", str(MAX_EPOCHS)])
    if BATCH_SIZE is not None:
        cmd.append(f"--data.params.batch_size={BATCH_SIZE}")
    return cmd


def main():
    ckpt_path = Path(AE_CKPT) if AE_CKPT else find_latest_ae_ckpt(ROOT / "logs")
    if not ckpt_path or not Path(ckpt_path).exists():
        raise FileNotFoundError(
            "Autoencoder checkpoint not found. Set AE_CKPT in scripts/run_train_ldm.py "
            "or make sure an Autoencoder run exists under logs/."
        )
    cmd = build_command(str(ckpt_path))
    cmd.append("--lightning.callbacks.image_logger.params.disabled=False")
    cmd.append("--lightning.callbacks.image_logger.params.batch_frequency=1")
    print("Using AE checkpoint:", ckpt_path)
    print("Running:", " ".join(cmd))
    env = os.environ.copy()
    if TAMING_DIR.exists():
        env["PYTHONPATH"] = str(TAMING_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=str(ROOT), check=True, env=env)


if __name__ == "__main__":
    main()

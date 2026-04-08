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
from latent_meanflow.models.backbones.latent_interval_unet import LatentIntervalUNet


UNET_CONFIGS = [
    "latent_meanflow_semantic_256_unet_tiny.yaml",
    "latent_meanflow_semantic_256_unet.yaml",
    "latent_meanflow_semantic_256_unet_large.yaml",
]


def _instantiate_backbone(config_name):
    cfg = OmegaConf.load(REPO_ROOT / "configs" / config_name)
    model = LatentIntervalUNet(in_channels=4, **cfg.model.params.backbone_config.params)
    batch_size = 2
    z_t = torch.randn(batch_size, 4, 64, 64)
    t = torch.rand(batch_size)
    r = torch.rand(batch_size) * 0.5
    out = model(z_t, t=t, r=r)
    assert out.shape == z_t.shape, f"{config_name}: expected {tuple(z_t.shape)}, got {tuple(out.shape)}"
    params = sum(parameter.numel() for parameter in model.parameters())
    print(f"{config_name}: output shape ok {tuple(out.shape)}, params={params}")


def _write_temp_tokenizer_artifacts(temp_root):
    tokenizer_config = OmegaConf.load(REPO_ROOT / "configs" / "autoencoder_semantic_pair_256.yaml")
    tokenizer_model = instantiate_from_config(tokenizer_config.model)

    tokenizer_config_path = temp_root / "autoencoder_semantic_pair_256.yaml"
    tokenizer_ckpt_path = temp_root / "autoencoder" / "checkpoints" / "last.ckpt"
    tokenizer_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(tokenizer_config, tokenizer_config_path)
    torch.save({"state_dict": tokenizer_model.state_dict()}, tokenizer_ckpt_path)
    return tokenizer_config_path, tokenizer_ckpt_path


def _write_temp_flow_artifacts(temp_root, tokenizer_config_path, tokenizer_ckpt_path):
    config_name = "latent_meanflow_semantic_256_unet_tiny"
    flow_config = OmegaConf.load(REPO_ROOT / "configs" / f"{config_name}.yaml")
    flow_config.model.params.tokenizer_config_path = str(tokenizer_config_path)
    flow_config.model.params.tokenizer_ckpt_path = str(tokenizer_ckpt_path)

    flow_config_path = temp_root / f"{config_name}.yaml"
    flow_run_dir = temp_root / "logs" / f"1970-01-01T00-00-00_{config_name}"
    flow_ckpt_path = flow_run_dir / "checkpoints" / "last.ckpt"
    flow_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(flow_config, flow_config_path)
    flow_model = instantiate_from_config(flow_config.model)
    torch.save({"state_dict": flow_model.state_dict()}, flow_ckpt_path)
    return flow_config_path, flow_ckpt_path


def _run_sample_smoke(temp_root, flow_config_path, flow_ckpt_path):
    outdir = temp_root / "outputs" / "meanflow_unet_tiny_nfe2"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "sample_latent_flow.py"),
        "--config",
        str(flow_config_path),
        "--ckpt",
        str(flow_ckpt_path),
        "--outdir",
        str(outdir),
        "--n-samples",
        "2",
        "--batch-size",
        "1",
        "--nfe",
        "2",
        "--seed",
        "23",
    ]
    subprocess.run(command, check=True, cwd=REPO_ROOT)

    expected_dirs = ["image", "mask_raw", "mask_color", "overlay"]
    for name in expected_dirs:
        files = sorted((outdir / name).glob("*.png"))
        assert len(files) == 2, f"{name}: expected 2 png files, found {len(files)}"
    print(f"sample_latent_flow.py smoke ok: {outdir}")


def main():
    for config_name in UNET_CONFIGS:
        _instantiate_backbone(config_name)

    temp_parent = REPO_ROOT / "outputs" / "_tmp_selfcheck"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_parent) as temp_dir:
        temp_root = Path(temp_dir)
        tokenizer_config_path, tokenizer_ckpt_path = _write_temp_tokenizer_artifacts(temp_root)
        flow_config_path, flow_ckpt_path = _write_temp_flow_artifacts(
            temp_root,
            tokenizer_config_path,
            tokenizer_ckpt_path,
        )
        _run_sample_smoke(temp_root, flow_config_path, flow_ckpt_path)
    shutil.rmtree(temp_parent, ignore_errors=True)


if __name__ == "__main__":
    main()

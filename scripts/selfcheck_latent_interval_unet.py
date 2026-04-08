import shutil
import subprocess
import sys
import tempfile
from types import MethodType
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
from latent_meanflow.models.backbones.latent_interval_unet import LatentIntervalUNet, _timestep_embedding
import sample_latent_flow
import train_latent_meanflow


UNET_CONFIGS = [
    "latent_meanflow_semantic_256_unet_tiny.yaml",
    "latent_meanflow_semantic_256_unet.yaml",
    "latent_meanflow_semantic_256_unet_large.yaml",
    "latent_alphaflow_semantic_256_unet.yaml",
]
ABLATION_CONFIGS = [
    "configs/ablations/latent_meanflow_semantic_256_unet_tscale1.yaml",
    "configs/ablations/latent_meanflow_semantic_256_unet_tscale100.yaml",
    "configs/ablations/latent_meanflow_semantic_256_unet_tscale1000.yaml",
    "configs/ablations/latent_meanflow_semantic_256_unet_rt_tscale100.yaml",
]


def _instantiate_backbone(config_path):
    cfg = OmegaConf.load(REPO_ROOT / config_path)
    model = LatentIntervalUNet(in_channels=4, **cfg.model.params.backbone_config.params)
    batch_size = 2
    z_t = torch.randn(batch_size, 4, 64, 64)
    t = torch.rand(batch_size)
    r = torch.rand(batch_size) * 0.5
    out = model(z_t, t=t, r=r)
    assert out.shape == z_t.shape, f"{config_path}: expected {tuple(z_t.shape)}, got {tuple(out.shape)}"
    params = sum(parameter.numel() for parameter in model.parameters())
    print(f"{config_path}: output shape ok {tuple(out.shape)}, params={params}")


def _legacy_resolve_time_embedding(model, t, r=None, delta_t=None):
    if t is None:
        raise ValueError("t must be provided")
    if t.ndim == 0:
        t = t[None]
    if t.ndim != 1:
        raise ValueError(f"Expected t with shape [B], got {tuple(t.shape)}")

    embedding = model.t_embed(_timestep_embedding(t.float(), model.model_channels))
    if model.time_conditioning == "t_delta":
        if delta_t is None:
            if r is None:
                raise ValueError("delta_t or r must be provided when time_conditioning='t_delta'")
            delta_t = t - r
        embedding = embedding + model.delta_embed(_timestep_embedding(delta_t.float(), model.model_channels))
    elif model.time_conditioning == "r_t":
        if r is None:
            raise ValueError("r must be provided when time_conditioning='r_t'")
        embedding = embedding + model.r_embed(_timestep_embedding(r.float(), model.model_channels))
    return embedding


def _assert_scale_one_matches_legacy(time_conditioning):
    params = {
        "model_channels": 32,
        "time_embed_dim": 128,
        "num_res_blocks": 1,
        "channel_mult": (1, 2, 4),
        "attention_resolutions": (4,),
        "dropout": 0.0,
        "time_conditioning": time_conditioning,
        "num_heads": 4,
        "num_head_channels": 32,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "t_time_scale": 1.0,
        "delta_time_scale": 1.0,
        "r_time_scale": 1.0,
    }
    model = LatentIntervalUNet(in_channels=4, **params)
    legacy_model = LatentIntervalUNet(in_channels=4, **params)
    legacy_model.load_state_dict(model.state_dict())

    def _patched_resolve(self, t, r=None, delta_t=None):
        return _legacy_resolve_time_embedding(self, t=t, r=r, delta_t=delta_t)

    legacy_model._resolve_time_embedding = MethodType(_patched_resolve, legacy_model)

    batch_size = 2
    z_t = torch.randn(batch_size, 4, 64, 64)
    t = torch.rand(batch_size)
    r = torch.rand(batch_size) * 0.5
    delta_t = t - r
    new_out = model(z_t, t=t, r=r, delta_t=delta_t)
    legacy_out = legacy_model(z_t, t=t, r=r, delta_t=delta_t)

    torch.testing.assert_close(new_out, legacy_out, atol=1.0e-6, rtol=1.0e-5)
    print(f"scale=1 legacy equivalence ok for time_conditioning={time_conditioning}")


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
    config_name = "latent_meanflow_semantic_256_unet_tscale100"
    flow_config = OmegaConf.load(REPO_ROOT / "configs" / "ablations" / f"{config_name}.yaml")
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


def _assert_default_routes():
    expected_alphaflow_config = REPO_ROOT / "configs" / "latent_alphaflow_semantic_256_unet.yaml"
    if train_latent_meanflow.DEFAULT_CONFIGS["alphaflow"] != expected_alphaflow_config:
        raise AssertionError(
            "train_latent_meanflow.py default alphaflow route drifted: "
            f"expected {expected_alphaflow_config}, got {train_latent_meanflow.DEFAULT_CONFIGS['alphaflow']}"
        )
    if sample_latent_flow.DEFAULT_CONFIG != expected_alphaflow_config:
        raise AssertionError(
            "sample_latent_flow.py default config drifted: "
            f"expected {expected_alphaflow_config}, got {sample_latent_flow.DEFAULT_CONFIG}"
        )
    print("default alphaflow training/sampling routes point to the U-Net config")


def main():
    _assert_default_routes()
    _assert_scale_one_matches_legacy(time_conditioning="t_delta")
    _assert_scale_one_matches_legacy(time_conditioning="r_t")

    for config_name in UNET_CONFIGS:
        _instantiate_backbone(Path("configs") / config_name)
    for config_name in ABLATION_CONFIGS:
        _instantiate_backbone(Path(config_name))

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

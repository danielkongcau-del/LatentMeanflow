# LatentMeanflow

LatentMeanflow now contains two parallel project-layer routes:

- a legacy binary-mask / 4-channel latent diffusion baseline
- a semantic tokenizer + latent flow research path for paired `RGB image + multiclass semantic mask`

The vendored runtime is still CompVis `latent-diffusion`, but the active project-layer research direction is:

`semantic tokenizer -> latent FM -> latent MeanFlow / AlphaFlow`

See [docs/latentmeanflow_migration_plan.md](docs/latentmeanflow_migration_plan.md) for the migration roadmap and [docs/semantic_label_spec.md](docs/semantic_label_spec.md) for the semantic mask contract.

## Repository Layout

- `configs/`: project-owned training configs.
- `configs/label_specs/`: reusable semantic label specifications shared by tokenizer and latent-flow configs.
- `scripts/`: project-owned entry points for training, sampling, and utility setup.
- `latent_meanflow/`: project-specific Python code such as dataset loaders, tokenizers, objectives, samplers, and trainers.
- `docs/`: project documentation, including migration planning and label contracts.
- `third_party/latent-diffusion/`: vendored upstream fork with local compatibility patches.
- `third_party/flow_matching/`: vendored upstream reference code. It is the future reference base for latent FM / MeanFlow / AlphaFlow, but it is not on the current training path.
- `docs/third_party.md`: provenance and patch summary for vendored code.

## Terminology

- `image-level class_label`: sample-level domain or category metadata. It is optional and applies to the whole image, not to individual pixels.
- `pixel-level semantic class`: the discrete label assigned to each mask pixel. This is the future primary paired-generation signal.
- `class_label != semantic_mask`: image-level metadata and pixel-level semantics are different contracts and should not be conflated.

## Data Layout

### Current legacy path

The checked-in legacy configs currently point to a paired dataset rooted at `data/remote`:

```text
data/
  remote/
    train/
      images/
      masks/
    val/
      images/
      masks/
    test/
      images/
      masks/
```

Each file in `images/` must have a same-stem partner in `masks/`.

The legacy loader treats each mask as a grayscale image, thresholds it to binary foreground/background, and packs `RGB + mask` into a single 4-channel tensor. That behavior is intentionally preserved for the old baseline only.

### Planned semantic path

The semantic route keeps `image_rgb` and `semantic_mask` as separate project-layer fields, uses an explicit label specification, and treats image-level `class_label` only as optional metadata. Multiclass semantic masks are not reduced to binary foreground/background.

The current shared label spec is:

```text
configs/label_specs/remote_semantic.yaml
```

All semantic tokenizer and latent-flow configs should reference that spec instead of duplicating `gray_to_class_id` mappings inline.

## Environment

The maintained environment target for this repository is:

- Python `3.10`
- PyTorch `2.11.0+cu128`
- torchvision `0.26.0+cu128`
- a single Conda environment such as `lmf`

For remote Linux GPU servers, use the stricter step-by-step guide in [docs/linux_server_environment.md](docs/linux_server_environment.md). That guide includes the Linux-specific install order, a locked dependency file, and the headless OpenCV choice used to avoid server-side dependency conflicts.

After activating your environment, install PyTorch from the official CUDA 12.8 wheel index:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/lmf.txt
python -m pip install -e third_party/latent-diffusion/taming-transformers
python -m pip install -e third_party/latent-diffusion
python -m pip install -e third_party/flow_matching
python -m pip install git+https://github.com/openai/CLIP.git
```

If LPIPS asks for VGG weights, place them with:

```bash
python scripts/place_vgg16_weights.py --src /path/to/vgg16-397923af.pth
```

## Training

### Legacy Binary-Mask / 4-Channel LDM Path

Train the 4-channel autoencoder:

```bash
python scripts/train_autoencoder.py --gpus 0
```

Train the latent diffusion model after the autoencoder checkpoint exists:

```bash
python scripts/train_ldm.py --gpus 0 --ae-ckpt logs/autoencoder/checkpoints/last.ckpt
```

Sample paired image and mask outputs:

```bash
python scripts/sample_mask_image_pairs.py --ckpt logs/ldm/checkpoints/last.ckpt --outdir outputs/remote
```

These commands target the legacy binary / 4-channel baseline only. They do not implement the planned multiclass semantic latent flow path yet.

### Semantic Tokenizer Path

Train the project-layer semantic tokenizer / autoencoder:

```bash
python scripts/train_semantic_autoencoder.py --gpus 0
```

This path reads `image` and multiclass `mask` separately, reconstructs RGB with an RGB head, and reconstructs semantic masks with `K`-channel logits rather than a single grayscale mask channel.

### Latent FM Path

Train the latent flow-matching prior on semantic tokenizer latents:

```bash
python scripts/train_latent_meanflow.py --objective fm --gpus 0
```

### Latent MeanFlow / AlphaFlow Path

Train the latent MeanFlow prior:

```bash
python scripts/train_latent_meanflow.py --objective meanflow --gpus 0
```

Run a tiny/debug MeanFlow pilot with the dedicated tiny config:

```bash
python scripts/train_latent_meanflow.py --objective meanflow --config configs/latent_meanflow_semantic_256_tiny.yaml --gpus 0
```

Run the first real MeanFlow baseline with the baseline config:

```bash
python scripts/train_latent_meanflow.py --objective meanflow --config configs/latent_meanflow_semantic_256.yaml --gpus 0
```

Train the recommended AlphaFlow curriculum:

```bash
python scripts/train_latent_meanflow.py --objective alphaflow --gpus 0
```

For quick validation or debugging, use the explicit smoke config:

```bash
python scripts/train_latent_meanflow.py --config configs/latent_alphaflow_semantic_256_smoke.yaml --gpus 0
```

Sample from a latent-flow checkpoint with `NFE=1` or `NFE=2`:

```bash
python scripts/sample_latent_flow.py --config configs/latent_alphaflow_semantic_256.yaml --ckpt /path/to/last.ckpt --nfe 1
python scripts/sample_latent_flow.py --config configs/latent_alphaflow_semantic_256.yaml --ckpt /path/to/last.ckpt --nfe 2
```

For MeanFlow, keep tiny/debug and baseline checkpoints separate. The training script now tags runs with the config stem, so use the helper below to resolve the baseline checkpoint before sampling the final `NFE` curve:

```bash
python scripts/find_checkpoint.py --config configs/latent_meanflow_semantic_256.yaml
python scripts/sample_latent_flow.py --config configs/latent_meanflow_semantic_256.yaml --ckpt <baseline-meanflow-ckpt> --nfe 2
```

`sample_latent_flow.py` now performs a basic path-level safety check and will reject a clearly mismatched config/checkpoint pair such as `configs/latent_meanflow_semantic_256.yaml` together with a checkpoint from a `*_latent_meanflow_semantic_256_tiny/` run.

The sampler writes:

- `image/`
- `mask_raw/`
- `mask_color/`
- `overlay/`

## Objective Semantics: alpha=1 vs r=t

- `alpha=1` means trajectory flow matching, noted here as `L_TFM`.
- `r=t` means the border-case flow-matching supervision, noted here as `L_FM'`.
- These are different axes:
  - `L_TFM` changes the AlphaFlow interpolation target by setting `alpha=1`
  - `L_FM'` changes the interval by collapsing it to `r=t`
- They must not be configured with the same ratio knob.

In project configs:

- `trajectory_fm_ratio` controls `alpha=1` trajectory-flow supervision
- `border_fm_ratio` controls `r=t` border-case supervision
- `flow_matching_ratio` is kept only as a deprecated compatibility alias for `trajectory_fm_ratio`

Config intent:

- `configs/latent_alphaflow_semantic_256_smoke.yaml`: smoke
- `configs/latent_fm_semantic_256.yaml`: project baseline
- `configs/latent_meanflow_semantic_256.yaml`: paper-like MeanFlow config
- `configs/latent_alphaflow_semantic_256.yaml`: project AlphaFlow baseline

## AlphaFlow Weighting Semantics

- `alpha_adaptive_exact` is only defined for samples with `alpha > 0`.
- When the alpha curriculum reaches `alpha = 0`, those samples are treated as MeanFlow samples and use MeanFlow `paper_like` adaptive weighting instead of `alpha_adaptive_exact`.
- This avoids the real bug where `alpha=0` samples would otherwise receive zero adaptive weight and silently stop contributing to training.

## Notes

- Autoencoder and legacy LDM runs use fixed paths such as `logs/autoencoder/` and `logs/ldm/`. Project-layer latent-flow runs are timestamped and now include the config stem in the run directory name, for example `logs/..._latent_meanflow_semantic_256/` or `logs/..._latent_meanflow_semantic_256_tiny/`.
- The root-level configs, scripts, docs, and dataset code are the project layer; avoid editing `third_party/` unless you are intentionally changing the vendored fork.
- `third_party/flow_matching/` is kept isolated so future flow-matching work does not pollute the current latent-diffusion baseline.
- The FM rectified path follows `z_t = (1 - t) x + t eps` and `v_t = eps - x`.
- The MeanFlow implementation uses JVP with tangent `(v, 0, 1)` and `t_delta` conditioning by default.
- The paper-like MeanFlow config uses `logit_normal(-0.4, 1.0)`, `border_fm_ratio=0.75`, and adaptive power `1.0`.
- The project AlphaFlow baseline uses the alpha curriculum and `border_fm_ratio=0.25`; it does not reinterpret that ratio as random `alpha=1` overrides.
- In the current AlphaFlow implementation, `alpha=0` samples fall back to MeanFlow `paper_like` weighting while `alpha>0` samples use `alpha_adaptive_exact`.
- The AlphaFlow implementation includes the `alpha^{-1}` prefactor and configurable curriculum, but `u_theta^-` is currently approximated with a detached online target branch rather than a separate EMA teacher. That is a project-layer engineering approximation, not a full paper-equivalent teacher setup.
- The `logit_normal` interval sampler is still an engineering approximation of the paper-aligned interval sampling policy: the code samples two scalar times independently from the chosen marginal distribution and sorts them into `(r, t)`.

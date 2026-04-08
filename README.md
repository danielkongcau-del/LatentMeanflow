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
Project-layer training wrappers disable post-fit `Trainer.test()` by default because the current semantic tokenizer and latent-flow trainers do not define `test_step()`.

If RGB reconstructions look washed out or overly smooth, try the LPIPS-enabled 24 GB config:

```bash
python scripts/train_semantic_autoencoder.py --config configs/autoencoder_semantic_pair_24gb_lpips_256.yaml --gpus 0
```

Current tokenizer geometry:

- `configs/autoencoder_semantic_pair_256.yaml`
- `configs/autoencoder_semantic_pair_24gb_lpips_256.yaml`

both produce a `64x64x4` latent from a `256x256` input.

Stronger tokenizer candidate geometry:

- `configs/autoencoder_semantic_pair_f8_256.yaml`
- `configs/autoencoder_semantic_pair_f8_24gb_lpips_256.yaml`

These keep the paired split-head decode contract but deepen the autoencoder so
that a `256x256` input maps to a `32x32x4` latent. This route is
meant as a tokenizer candidate for downstream few-step latent flow work. It is
not an SD-VAE-equivalent claim.

Train the stronger f=8 tokenizer candidate:

```bash
python scripts/train_semantic_autoencoder.py --config configs/autoencoder_semantic_pair_f8_256.yaml --gpus 0
```

Train the stronger f=8 tokenizer candidate with LPIPS:

```bash
python scripts/train_semantic_autoencoder.py --config configs/autoencoder_semantic_pair_f8_24gb_lpips_256.yaml --gpus 0
```

Evaluate one tokenizer checkpoint and write JSON + markdown summaries:

```bash
python scripts/eval_semantic_tokenizer.py --config configs/autoencoder_semantic_pair_256.yaml --ckpt /path/to/tokenizer.ckpt --outdir outputs/tokenizer_eval/base
```

Evaluate the stronger f=8 tokenizer against the current tokenizer on the same
split and batches:

```bash
python scripts/eval_semantic_tokenizer.py \
  --config configs/autoencoder_semantic_pair_f8_256.yaml \
  --ckpt /path/to/f8_tokenizer.ckpt \
  --reference-config configs/autoencoder_semantic_pair_256.yaml \
  --reference-ckpt /path/to/base_tokenizer.ckpt \
  --outdir outputs/tokenizer_eval/f8_vs_base
```

Reserve downstream latent-prior experiments for the stronger tokenizer branch:

```bash
python scripts/train_latent_meanflow.py --objective meanflow --config configs/latent_meanflow_semantic_f8_unet.yaml --tokenizer-config configs/autoencoder_semantic_pair_f8_256.yaml --tokenizer-ckpt /path/to/f8_tokenizer.ckpt --gpus 0
python scripts/train_latent_meanflow.py --objective alphaflow --config configs/latent_alphaflow_semantic_f8_unet.yaml --tokenizer-config configs/autoencoder_semantic_pair_f8_256.yaml --tokenizer-ckpt /path/to/f8_tokenizer.ckpt --gpus 0
```

### Image-Only Tokenizer Path

The project-layer image-only tokenizer is a parallel route for future
`p(image | mask)` work.

Unlike the joint semantic tokenizer:

- the encoder only sees RGB image
- the semantic mask does not enter tokenizer latent `z`
- the mask is intended to stay outside the tokenizer and be injected later as
  an external condition to a mask-conditioned latent prior

This route still reuses the current paired dataset loader, but only reads the
RGB image tensor from each sample. It is a project-layer image tokenizer, not
an SD-VAE-equivalent claim. The downstream mask-conditioned latent prior route
now exists as a separate project-layer path and keeps the semantic mask outside
tokenizer latent `z`.

Base image-only geometry:

- `configs/autoencoder_image_256.yaml`
- `configs/autoencoder_image_24gb_lpips_256.yaml`

both produce a `64x64x4` latent from a `256x256` input.

Stronger image-only `f=8` geometry:

- `configs/autoencoder_image_f8_256.yaml`
- `configs/autoencoder_image_f8_24gb_lpips_256.yaml`

both produce a `32x32x4` latent from a `256x256` input.

Train the base image-only tokenizer:

```bash
python scripts/train_image_autoencoder.py --config configs/autoencoder_image_256.yaml --gpus 0
```

Train the stronger image-only `f=8` tokenizer:

```bash
python scripts/train_image_autoencoder.py --config configs/autoencoder_image_f8_256.yaml --gpus 0
```

Train the LPIPS-enabled image-only variants:

```bash
python scripts/train_image_autoencoder.py --config configs/autoencoder_image_24gb_lpips_256.yaml --gpus 0
python scripts/train_image_autoencoder.py --config configs/autoencoder_image_f8_24gb_lpips_256.yaml --gpus 0
```

Evaluate one image-only tokenizer checkpoint and write JSON + markdown
summaries:

```bash
python scripts/eval_image_tokenizer.py --config configs/autoencoder_image_256.yaml --ckpt /path/to/image_tokenizer.ckpt --outdir outputs/image_tokenizer_eval/base
```

Compare the stronger image-only `f=8` tokenizer against the base image-only
tokenizer on the same split and batches:

```bash
python scripts/eval_image_tokenizer.py \
  --config configs/autoencoder_image_f8_256.yaml \
  --ckpt /path/to/image_tokenizer_f8.ckpt \
  --reference-config configs/autoencoder_image_256.yaml \
  --reference-ckpt /path/to/image_tokenizer_base.ckpt \
  --outdir outputs/image_tokenizer_eval/f8_vs_base
```

### Mask-Conditioned Image Route

The project-layer conditional renderer is a separate route:

- it models `p(image | semantic_mask)`
- it does not model `p(image, semantic_mask)`
- it uses the image-only tokenizer, not the joint semantic tokenizer
- the semantic mask stays outside latent `z`
- the semantic mask is injected as an external spatial condition into the U-Net

This route is intended as the first sanity check for whether the latent prior
can render plausible RGB images from a given semantic layout before adding any
separate semantic-mask prior.

Condition-path variants:

- `input_concat`
  - current clean baseline
  - one latent-scale one-hot mask concat at the U-Net input only
- `pyramid_concat`
  - project-layer conditioning upgrade
  - builds a semantic-mask pyramid and injects per-scale features across the U-Net
- `pyramid_concat + boundary`
  - same multi-scale route plus a simple boundary-aware auxiliary channel

The semantic mask class count is now derived from the shared label spec rather
than hard-coded as a checked-in `7` channel constant in the main configs.

Checked-in configs:

- `configs/latent_fm_mask2image_unet.yaml`
- `configs/latent_meanflow_mask2image_unet.yaml`
- `configs/latent_alphaflow_mask2image_unet.yaml`
- `configs/latent_alphaflow_mask2image_unet_tiny.yaml`
- `configs/latent_alphaflow_mask2image_f8_unet.yaml`

Renderer conditioning ablations:

- `configs/ablations/latent_alphaflow_mask2image_unet_input_concat.yaml`
- `configs/ablations/latent_alphaflow_mask2image_unet_pyramid.yaml`
- `configs/ablations/latent_alphaflow_mask2image_unet_pyramid_boundary.yaml`

Use the dedicated wrapper so bare `--objective fm/meanflow/alphaflow` resolves
inside the mask-conditioned route rather than the existing paired joint route:

```bash
python scripts/train_mask_conditioned_image.py --objective alphaflow --config configs/latent_alphaflow_mask2image_unet_tiny.yaml --tokenizer-ckpt /path/to/image_tokenizer.ckpt --gpus 0
python scripts/train_mask_conditioned_image.py --objective alphaflow --config configs/latent_alphaflow_mask2image_unet.yaml --tokenizer-ckpt /path/to/image_tokenizer.ckpt --gpus 0
```

Sample the fixed `NFE=8/4/2/1` few-step sweep:

```bash
python scripts/find_checkpoint.py --run-dir logs/<your_run> --selection best --monitor val/base_error_mean
python scripts/sample_mask_conditioned_image.py --config configs/latent_alphaflow_mask2image_unet.yaml --ckpt <best-ckpt> --outdir outputs/mask_conditioned_image/example --split validation --seed 23 --nfe-values 8 4 2 1
```

Evaluate the generated images against ground truth sanity metrics:

```bash
python scripts/eval_mask_conditioned_image.py --config configs/latent_alphaflow_mask2image_unet.yaml --ckpt <best-ckpt> --outdir outputs/mask_conditioned_eval/example --split validation --seed 23 --nfe-values 8 4 2 1
```

For the full run order, sampling protocol, and success criteria, use
[docs/mask_conditioned_image_plan.md](docs/mask_conditioned_image_plan.md).
For the condition-path comparison protocol, use
[docs/mask_conditioned_renderer_benchmark.md](docs/mask_conditioned_renderer_benchmark.md).

### Latent FM Path

Train the latent flow-matching prior on semantic tokenizer latents:

```bash
python scripts/train_latent_meanflow.py --objective fm --gpus 0
```

### Latent MeanFlow / AlphaFlow Path

Train the default latent MeanFlow prior. The default `meanflow` route now
points to the U-Net baseline config because the seed-matched ConvNet vs U-Net
benchmark showed the U-Net backbone was clearly better at low-NFE sampling
without regressing at `NFE=8`.

```bash
python scripts/train_latent_meanflow.py --objective meanflow --gpus 0
```

Run a tiny/debug MeanFlow pilot with the dedicated tiny config:

```bash
python scripts/train_latent_meanflow.py --objective meanflow --config configs/latent_meanflow_semantic_256_tiny.yaml --gpus 0
```

Run the default U-Net MeanFlow baseline explicitly:

```bash
python scripts/train_latent_meanflow.py --objective meanflow --config configs/latent_meanflow_semantic_256_unet.yaml --gpus 0
```

Keep the old ConvNet MeanFlow baseline available as a legacy backbone baseline:

```bash
python scripts/train_latent_meanflow.py --objective meanflow --config configs/latent_meanflow_semantic_256.yaml --gpus 0
```

Parallel backbone experiment path: U-Net MeanFlow. The benchmark-backed U-Net
baseline is now the default MeanFlow route. The old ConvNet configs are kept as
legacy baselines for rollback and comparison.

Run the U-Net tiny/debug pilot:

```bash
python scripts/train_latent_meanflow.py --objective meanflow --config configs/latent_meanflow_semantic_256_unet_tiny.yaml --gpus 0
```

Run the U-Net MeanFlow baseline:

```bash
python scripts/train_latent_meanflow.py --objective meanflow --config configs/latent_meanflow_semantic_256_unet.yaml --gpus 0
```

Run the U-Net MeanFlow large capacity bump:

```bash
python scripts/train_latent_meanflow.py --objective meanflow --config configs/latent_meanflow_semantic_256_unet_large.yaml --gpus 0
```

The current few-step paired-generation main line is U-Net AlphaFlow. Keep the
tokenizer and paired task fixed, and treat the following configs as a single
project-layer recipe family:

- `configs/latent_alphaflow_semantic_256_unet_tiny.yaml`: tiny/debug pilot
- `configs/latent_alphaflow_semantic_256_unet.yaml`: default project baseline
- `configs/latent_alphaflow_semantic_256_unet_large.yaml`: large-capacity U-Net bump
- `configs/latent_alphaflow_semantic_256_unet_paper_attempt.yaml`: paper-aligned schedule attempt, not a paper-equivalent claim
- `configs/latent_alphaflow_semantic_256.yaml`: legacy ConvNet rollback baseline

Run the U-Net AlphaFlow tiny/debug pilot:

```bash
python scripts/train_latent_meanflow.py --objective alphaflow --config configs/latent_alphaflow_semantic_256_unet_tiny.yaml --gpus 0
```

Run the default U-Net AlphaFlow project baseline:

```bash
python scripts/train_latent_meanflow.py --objective alphaflow --config configs/latent_alphaflow_semantic_256_unet.yaml --gpus 0
```

Run the U-Net AlphaFlow large capacity bump:

```bash
python scripts/train_latent_meanflow.py --objective alphaflow --config configs/latent_alphaflow_semantic_256_unet_large.yaml --gpus 0
```

Run the U-Net AlphaFlow paper-aligned schedule attempt:

```bash
python scripts/train_latent_meanflow.py --objective alphaflow --config configs/latent_alphaflow_semantic_256_unet_paper_attempt.yaml --gpus 0
```

Keep the old ConvNet AlphaFlow baseline available as a legacy rollback path:

```bash
python scripts/train_latent_meanflow.py --objective alphaflow --config configs/latent_alphaflow_semantic_256.yaml --gpus 0
```

For the multi-GPU few-step training recipe, effective batch math, and fixed
`NFE=8/4/2/1` evaluation protocol, use
[docs/unet_alphaflow_training_plan.md](docs/unet_alphaflow_training_plan.md).

Engineering ablation path: U-Net time-embedding input scale. These configs do
not change the MeanFlow objective math. They only rescale the raw `[0,1]`
scalar inputs before the U-Net sinusoidal embedding, so they should be treated
as engineering ablations rather than paper-equivalent claims.
The existing U-Net configs keep the backward-compatible default behavior
because omitted scales still resolve to `1.0`.

Run the raw-scale control (`t` and `delta_t` scale = `1`):

```bash
python scripts/train_latent_meanflow.py --objective meanflow --config configs/ablations/latent_meanflow_semantic_256_unet_tscale1.yaml --gpus 0
```

Run the medium-scale ablation (`t` and `delta_t` scale = `100`):

```bash
python scripts/train_latent_meanflow.py --objective meanflow --config configs/ablations/latent_meanflow_semantic_256_unet_tscale100.yaml --gpus 0
```

Run the large-scale ablation (`t` and `delta_t` scale = `1000`):

```bash
python scripts/train_latent_meanflow.py --objective meanflow --config configs/ablations/latent_meanflow_semantic_256_unet_tscale1000.yaml --gpus 0
```

Optional conditioning-form comparison: explicit `(r, t)` with both scales set
to `100`:

```bash
python scripts/train_latent_meanflow.py --objective meanflow --config configs/ablations/latent_meanflow_semantic_256_unet_rt_tscale100.yaml --gpus 0
```

Resume modes:

- `fresh run`: pass `--config` normally, or rely on the default chosen by `--objective`
- `safe resume`: pass only `--resume <ckpt-or-logdir>` and let the vendored trainer reload the saved `configs/*.yaml`
- `dangerous override resume`: only if you explicitly want to override the saved config or tokenizer settings

Recommended safe resume:

```bash
python scripts/train_latent_meanflow.py --resume logs/<your_run>/checkpoints/last.ckpt
```

`safe resume` does not allow `--set` dotlist overrides.
It also does not allow wrapper flags that become CLI dotlist overrides, such as `--batch-size`,
`--enable-image-logger`, or `--image-log-frequency`.

Consistency-check resume with an explicit matching config:

```bash
python scripts/train_latent_meanflow.py --resume logs/<your_run>/checkpoints/last.ckpt --config configs/latent_meanflow_semantic_256.yaml
```

In that mode, the wrapper checks that the explicit config matches the resume run, but it still does not inject a new `--base`.

Dangerous override resume:

```bash
python scripts/train_latent_meanflow.py \
  --resume logs/<your_run>/checkpoints/last.ckpt \
  --config configs/latent_meanflow_semantic_256.yaml \
  --allow-config-override \
  --force-tokenizer-config \
  --force-tokenizer-ckpt
```

Dangerous dotlist override resume:

```bash
python scripts/train_latent_meanflow.py \
  --resume logs/<your_run>/checkpoints/last.ckpt \
  --allow-dotlist-override \
  --set model.params.some_field=some_value
```

Dangerous wrapper-dotlist override resume:

```bash
python scripts/train_latent_meanflow.py \
  --resume logs/<your_run>/checkpoints/last.ckpt \
  --allow-dotlist-override \
  --batch-size 8 \
  --enable-image-logger \
  --image-log-frequency 50
```

By default, resume no longer injects:

- a new `--base <config>`
- `model.params.tokenizer_config_path=...`
- `model.params.tokenizer_ckpt_path=...`
- any `--set` dotlist overrides
- `data.params.batch_size=...`
- `lightning.callbacks.image_logger.params.disabled=False`
- `lightning.callbacks.image_logger.params.batch_frequency=...`

`--allow-dotlist-override` is a dangerous escape hatch. It can bypass the wrapper's safety model, including protections around tokenizer-related settings and wrapper-managed runtime dotlist flags, so it is not the recommended path.

This prevents a default `alphaflow` config, a new tokenizer path, or any arbitrary dotlist override, including wrapper-managed batch-size and image-logger flags, from being silently merged into an existing `meanflow` resume run.

Train the recommended AlphaFlow curriculum. Bare `--objective alphaflow` now
resolves to the U-Net AlphaFlow project baseline:

```bash
python scripts/train_latent_meanflow.py --objective alphaflow --gpus 0
```

For quick validation or debugging, use the U-Net AlphaFlow tiny/debug config:

```bash
python scripts/train_latent_meanflow.py --objective alphaflow --config configs/latent_alphaflow_semantic_256_unet_tiny.yaml --gpus 0
```

Sample from a latent-flow checkpoint with an explicit U-Net AlphaFlow config.
For quick looks use `NFE=4` or `NFE=2`; for formal few-step comparisons use the
fixed `NFE=8/4/2/1` sweep in
[docs/unet_alphaflow_training_plan.md](docs/unet_alphaflow_training_plan.md)
or [docs/unet_backbone_benchmark.md](docs/unet_backbone_benchmark.md).
Bare `sample_latent_flow.py` now defaults to the U-Net AlphaFlow config, so
keep the config stem explicit whenever you want the legacy ConvNet rollback
path:

```bash
python scripts/sample_latent_flow.py --config configs/latent_alphaflow_semantic_256_unet.yaml --ckpt /path/to/last.ckpt --nfe 4
python scripts/sample_latent_flow.py --config configs/latent_alphaflow_semantic_256_unet.yaml --ckpt /path/to/last.ckpt --nfe 2
```

For MeanFlow, keep tiny/debug and baseline checkpoints separate. The training script now tags runs with the config stem, so use the helper below to resolve the baseline checkpoint before sampling the final `NFE` curve:

```bash
python scripts/find_checkpoint.py --config configs/latent_meanflow_semantic_256_unet.yaml
python scripts/sample_latent_flow.py --config configs/latent_meanflow_semantic_256_unet.yaml --ckpt <baseline-meanflow-ckpt> --nfe 2
```

For the legacy ConvNet fallback route, keep the matching ConvNet config stem so
the checkpoint safety check resolves the correct run family:

```bash
python scripts/find_checkpoint.py --config configs/latent_meanflow_semantic_256.yaml
python scripts/sample_latent_flow.py --config configs/latent_meanflow_semantic_256.yaml --ckpt <legacy-convnet-meanflow-ckpt> --nfe 2
python scripts/find_checkpoint.py --config configs/latent_alphaflow_semantic_256.yaml
python scripts/sample_latent_flow.py --config configs/latent_alphaflow_semantic_256.yaml --ckpt <legacy-convnet-alphaflow-ckpt> --nfe 2
python scripts/find_checkpoint.py --config configs/latent_alphaflow_semantic_256_unet.yaml
python scripts/sample_latent_flow.py --config configs/latent_alphaflow_semantic_256_unet.yaml --ckpt <baseline-alphaflow-unet-ckpt> --nfe 2
python scripts/find_checkpoint.py --config configs/latent_alphaflow_semantic_256_unet_paper_attempt.yaml
python scripts/sample_latent_flow.py --config configs/latent_alphaflow_semantic_256_unet_paper_attempt.yaml --ckpt <paper-attempt-alphaflow-unet-ckpt> --nfe 2
```

`sample_latent_flow.py` now performs a basic path-level safety check and will reject a clearly mismatched config/checkpoint pair such as `configs/latent_meanflow_semantic_256.yaml` together with a checkpoint from a `*_latent_meanflow_semantic_256_tiny/` run.

The sampler writes:

- `image/`
- `mask_raw/`
- `mask_color/`
- `overlay/`

For apples-to-apples ConvNet vs U-Net benchmarking, do not rely on the sampler
defaults. Use [docs/unet_backbone_benchmark.md](docs/unet_backbone_benchmark.md)
together with:

```bash
python scripts/find_checkpoint.py --run-dir logs/<your_run> --selection best --monitor val/base_error_mean
python scripts/eval_backbone_nfe_sweep.py --config configs/latent_meanflow_semantic_256.yaml --ckpt <best-ckpt> --outdir outputs/benchmarks/example --seed 23 --nfe-values 8 4 2 1
```

That benchmark route fixes the checkpoint, seed, and `NFE=8/4/2/1` sweep and
always writes `image/`, `mask_raw/`, `mask_color/`, and `overlay/` for every
NFE.

For research-style paired RGB + semantic-mask evaluation, use
[docs/semantic_pair_eval_protocol.md](docs/semantic_pair_eval_protocol.md)
with:

```bash
python scripts/find_checkpoint.py --run-dir logs/<your_run> --selection best --monitor val/base_error_mean
python scripts/eval_semantic_pair_generation.py --config configs/latent_alphaflow_semantic_256_unet.yaml --ckpt <best-ckpt> --outdir outputs/semantic_pair_eval/example --seed 23 --nfe-values 8 4 2 1 --teacher-hf-model <hf-teacher-model-id-or-local-path>
```

That route reports RGB `FID/KID`, teacher-aligned mask `mIoU` /
`per-class IoU`, `Boundary F1`, and paired-consistency agreement between the
generated mask and the teacher segmentation of the generated RGB image.

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
- `configs/autoencoder_semantic_pair_256.yaml`: current semantic tokenizer baseline with `64x64x4` latents
- `configs/autoencoder_semantic_pair_24gb_lpips_256.yaml`: current semantic tokenizer baseline with LPIPS and `64x64x4` latents
- `configs/autoencoder_semantic_pair_f8_256.yaml`: stronger semantic tokenizer candidate with `32x32x4` latents
- `configs/autoencoder_semantic_pair_f8_24gb_lpips_256.yaml`: stronger `f=8` tokenizer candidate with LPIPS
- `configs/autoencoder_image_256.yaml`: project-layer image-only tokenizer baseline with `64x64x4` latents for future `p(image | mask)` work
- `configs/autoencoder_image_24gb_lpips_256.yaml`: image-only tokenizer baseline with LPIPS and `64x64x4` latents
- `configs/autoencoder_image_f8_256.yaml`: stronger image-only tokenizer candidate with `32x32x4` latents
- `configs/autoencoder_image_f8_24gb_lpips_256.yaml`: stronger image-only `f=8` tokenizer candidate with LPIPS
- `configs/latent_meanflow_semantic_256.yaml`: legacy ConvNet MeanFlow baseline
- `configs/latent_meanflow_semantic_256_unet_tiny.yaml`: tiny/debug U-Net MeanFlow parallel path
- `configs/latent_meanflow_semantic_256_unet.yaml`: default MeanFlow backbone path after the benchmark-backed promotion
- `configs/latent_meanflow_semantic_256_unet_large.yaml`: U-Net MeanFlow large parallel path
- `configs/latent_meanflow_semantic_f8_unet.yaml`: MeanFlow U-Net route reserved for the stronger `f=8` tokenizer branch
- `configs/ablations/latent_meanflow_semantic_256_unet_tscale{1,100,1000}.yaml`: engineering-only U-Net time-scale ablations
- `configs/ablations/latent_meanflow_semantic_256_unet_rt_tscale100.yaml`: engineering-only `(r, t)` conditioning ablation
- `configs/latent_alphaflow_semantic_256.yaml`: legacy ConvNet AlphaFlow rollback baseline
- `configs/latent_alphaflow_semantic_256_unet_tiny.yaml`: U-Net AlphaFlow tiny/debug pilot
- `configs/latent_alphaflow_semantic_256_unet.yaml`: default AlphaFlow U-Net project baseline for few-step paired generation
- `configs/latent_alphaflow_semantic_256_unet_large.yaml`: U-Net AlphaFlow large-capacity path with the same effective batch target as the base route
- `configs/latent_alphaflow_semantic_256_unet_paper_attempt.yaml`: U-Net AlphaFlow paper-aligned schedule attempt, not a paper-equivalent claim
- `configs/latent_alphaflow_semantic_f8_unet.yaml`: AlphaFlow U-Net route reserved for the stronger `f=8` tokenizer branch
- `configs/latent_fm_mask2image_unet.yaml`: flow-matching baseline for `p(image | semantic_mask)` with image-only tokenizer latents
- `configs/latent_meanflow_mask2image_unet.yaml`: MeanFlow baseline for `p(image | semantic_mask)` with image-only tokenizer latents
- `configs/latent_alphaflow_mask2image_unet.yaml`: recommended AlphaFlow baseline for `p(image | semantic_mask)`
- `configs/latent_alphaflow_mask2image_unet_tiny.yaml`: tiny/debug AlphaFlow mask-conditioned sanity route
- `configs/latent_alphaflow_mask2image_f8_unet.yaml`: AlphaFlow mask-conditioned route for the stronger image-only `f=8` tokenizer branch

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
- The default MeanFlow route now points to the U-Net baseline because the pinned benchmark protocol showed it beat the ConvNet baseline at `NFE=4/2` without regressing at `NFE=8`.
- The legacy ConvNet MeanFlow baseline is still kept for rollback and comparison.
- The default AlphaFlow route now also points to the U-Net project baseline; the legacy ConvNet AlphaFlow config remains checked in as a rollback baseline.
- The promoted U-Net AlphaFlow route keeps the tokenizer and paired task fixed while increasing the effective batch through gradient accumulation. On the checked-in 2-GPU recipes the effective batch target is `64`.
- The U-Net time-scale configs under `configs/ablations/` are engineering-only ablations of the scalar embedding input scale; they do not change the objective and they are not paper-equivalent claims.
- The project AlphaFlow baseline uses the alpha curriculum and `border_fm_ratio=0.25`; it does not reinterpret that ratio as random `alpha=1` overrides.
- The paper-aligned AlphaFlow attempt only moves the curriculum schedule and clamping closer to the paper's semantics. It does not add a separate EMA teacher, so it is still not paper-equivalent.
- In the current AlphaFlow implementation, `alpha=0` samples fall back to MeanFlow `paper_like` weighting while `alpha>0` samples use `alpha_adaptive_exact`.
- The AlphaFlow implementation includes the `alpha^{-1}` prefactor and configurable curriculum, but `u_theta^-` is currently approximated with a detached online target branch rather than a separate EMA teacher. That is a project-layer engineering approximation, not a full paper-equivalent teacher setup.
- The `logit_normal` interval sampler is still an engineering approximation of the paper-aligned interval sampling policy: the code samples two scalar times independently from the chosen marginal distribution and sorts them into `(r, t)`.
- The mask-conditioned image route is a separate project-layer task definition: it models `p(image | semantic_mask)` with an image-only tokenizer and external spatial mask conditioning. It does not reinterpret the paired joint route as a conditional route.

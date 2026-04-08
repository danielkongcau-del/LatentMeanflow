# Mask-Conditioned Image Plan

This runbook defines the project-layer `p(image | semantic_mask)` route.

It is deliberately different from the existing paired joint-generation route:

- tokenizer latent represents `image` only
- `semantic_mask` stays outside latent `z`
- the prior learns `p(z_image | semantic_mask)`
- the clean baseline uses `condition_mode: input_concat`
- stronger ablations use multi-scale `pyramid_concat` conditioning
- high-resolution-aware ablations use `condition_source: fullres_mask`

This route keeps the checked-in latent FM / MeanFlow / AlphaFlow objective math
unchanged. It only changes the task definition, tokenizer family, and
conditioning path.

## Scope

- Keep the image-only tokenizer fixed within a given run.
- Keep the semantic pair dataset contract fixed.
- Keep the flow objective math fixed.
- Keep `semantic_mask` separate from image-level `class_label`.
- Do not interpret `L1` / `LPIPS` to ground truth as the primary task metric.

This is a project-layer route, not an SD-VAE-equivalent claim and not a
paper-equivalent conditional renderer claim.

## What Is Being Modeled

- Current route: `p(image | semantic_mask)`
- Not modeled here: `p(image, semantic_mask)`
- Not modeled here: unconditional semantic-mask prior

The mask is an external spatial condition only. It does not enter the
image-only tokenizer encoder and does not become part of latent `z`.

## Config Roles

- `configs/latent_fm_mask2image_unet.yaml`
  - Flow-matching baseline for the conditional renderer.
- `configs/latent_meanflow_mask2image_unet.yaml`
  - MeanFlow conditional baseline.
- `configs/latent_alphaflow_mask2image_unet.yaml`
  - Recommended first formal long run today.
  - Uses the clean `input_concat` baseline.
- `configs/latent_alphaflow_mask2image_unet_tiny.yaml`
  - Tiny/debug overfit route for 32 or 64 samples.
- `configs/latent_alphaflow_mask2image_f8_unet.yaml`
  - Same route as the base AlphaFlow config, but with the stronger `32x32x4`
    image-only tokenizer geometry.
- `configs/ablations/latent_alphaflow_mask2image_unet_input_concat.yaml`
  - Explicit benchmark copy of the clean baseline.
- `configs/ablations/latent_alphaflow_mask2image_unet_pyramid.yaml`
  - Multi-scale semantic-mask pyramid injected across U-Net stages.
- `configs/ablations/latent_alphaflow_mask2image_unet_pyramid_boundary.yaml`
  - Multi-scale pyramid plus a simple boundary-aware auxiliary channel.
- `configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid.yaml`
  - Builds the semantic condition pyramid from the original mask resolution.
- `configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid_boundary.yaml`
  - Full-resolution pyramid plus binary boundary channel.
- `configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid_boundary_encoder.yaml`
  - Full-resolution pyramid plus stronger per-class boundary features and a
    lightweight semantic condition encoder.

The semantic mask channel count is derived from
`configs/label_specs/remote_semantic.yaml` in the main checked-in configs. The
baseline no longer relies on a hand-written `spatial_condition_channels: 7`
constant.

## Best Checkpoint Rule

For MeanFlow / AlphaFlow, select the best checkpoint by:

- `val/base_error_mean`

Do not choose by:

- `val/meanflow_loss`
- `val/alphaflow_loss`
- `last.ckpt` only

Resolve the checkpoint explicitly:

```bash
python scripts/find_checkpoint.py --run-dir logs/<your_run> --selection best --monitor val/base_error_mean
```

## First Run Recommendation

1. Tiny pilot:
   - `configs/latent_alphaflow_mask2image_unet_tiny.yaml`
2. First formal long run:
   - `configs/latent_alphaflow_mask2image_unet.yaml`
3. Stronger tokenizer follow-up:
   - `configs/latent_alphaflow_mask2image_f8_unet.yaml`
4. Conditioning benchmark:
   - `configs/ablations/latent_alphaflow_mask2image_unet_input_concat.yaml`
   - `configs/ablations/latent_alphaflow_mask2image_unet_pyramid.yaml`
   - `configs/ablations/latent_alphaflow_mask2image_unet_pyramid_boundary.yaml`
   - `configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid.yaml`
   - `configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid_boundary.yaml`
   - `configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid_boundary_encoder.yaml`

## Training Commands

These examples assume:

- `lmf` environment is active
- the image-only tokenizer checkpoint already exists
- two GPUs are available

### Tiny Pilot / Overfit Sanity Check

Use this first. The success criterion is simple: the generated image must
clearly follow the input semantic layout rather than ignoring it.

```bash
python scripts/train_mask_conditioned_image.py \
  --objective alphaflow \
  --config configs/latent_alphaflow_mask2image_unet_tiny.yaml \
  --tokenizer-ckpt /path/to/image_tokenizer.ckpt \
  --gpus 0,1
```

### Base AlphaFlow Long Run

```bash
python scripts/train_mask_conditioned_image.py \
  --objective alphaflow \
  --config configs/latent_alphaflow_mask2image_unet.yaml \
  --tokenizer-ckpt /path/to/image_tokenizer.ckpt \
  --gpus 0,1
```

### Conditioning Upgrade Benchmark

```bash
python scripts/train_mask_conditioned_image.py \
  --objective alphaflow \
  --config configs/ablations/latent_alphaflow_mask2image_unet_input_concat.yaml \
  --tokenizer-ckpt /path/to/image_tokenizer.ckpt \
  --gpus 0,1

python scripts/train_mask_conditioned_image.py \
  --objective alphaflow \
  --config configs/ablations/latent_alphaflow_mask2image_unet_pyramid.yaml \
  --tokenizer-ckpt /path/to/image_tokenizer.ckpt \
  --gpus 0,1

python scripts/train_mask_conditioned_image.py \
  --objective alphaflow \
  --config configs/ablations/latent_alphaflow_mask2image_unet_pyramid_boundary.yaml \
  --tokenizer-ckpt /path/to/image_tokenizer.ckpt \
  --gpus 0,1

python scripts/train_mask_conditioned_image.py \
  --objective alphaflow \
  --config configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid.yaml \
  --tokenizer-ckpt /path/to/image_tokenizer.ckpt \
  --gpus 0,1

python scripts/train_mask_conditioned_image.py \
  --objective alphaflow \
  --config configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid_boundary.yaml \
  --tokenizer-ckpt /path/to/image_tokenizer.ckpt \
  --gpus 0,1

python scripts/train_mask_conditioned_image.py \
  --objective alphaflow \
  --config configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid_boundary_encoder.yaml \
  --tokenizer-ckpt /path/to/image_tokenizer.ckpt \
  --gpus 0,1
```

### Stronger `f=8` Follow-Up

```bash
python scripts/train_mask_conditioned_image.py \
  --objective alphaflow \
  --config configs/latent_alphaflow_mask2image_f8_unet.yaml \
  --tokenizer-ckpt /path/to/image_tokenizer_f8.ckpt \
  --gpus 0,1
```

## Sampling Protocol

Always run the same `NFE=8/4/2/1` sweep on the same checkpoint and seed.

The checked-in sampler writes:

- `input_mask_raw/`
- `input_mask_color/`
- `generated_image/`
- `ground_truth_image/` when available
- `overlay/`
- `panel/`

### Validation-Set Mask Sweep

```bash
python scripts/sample_mask_conditioned_image.py \
  --config configs/latent_alphaflow_mask2image_unet.yaml \
  --ckpt <best-ckpt> \
  --outdir outputs/mask_conditioned_image/base \
  --split validation \
  --seed 23 \
  --n-samples 32 \
  --batch-size 4 \
  --nfe-values 8 4 2 1
```

### Standalone Mask Directory Sweep

```bash
python scripts/sample_mask_conditioned_image.py \
  --config configs/latent_alphaflow_mask2image_unet.yaml \
  --ckpt <best-ckpt> \
  --outdir outputs/mask_conditioned_image/custom_masks \
  --mask-dir /path/to/masks \
  --image-dir /path/to/images \
  --label-spec configs/label_specs/remote_semantic.yaml \
  --seed 23 \
  --n-samples 32 \
  --batch-size 4 \
  --nfe-values 8 4 2 1
```

## Evaluation Protocol

Use:

```bash
python scripts/eval_mask_conditioned_image.py \
  --config configs/latent_alphaflow_mask2image_unet.yaml \
  --ckpt <best-ckpt> \
  --outdir outputs/mask_conditioned_eval/base \
  --split validation \
  --seed 23 \
  --n-samples 32 \
  --batch-size 4 \
  --nfe-values 8 4 2 1
```

The evaluation summary writes:

- `summary.json`
- `summary.csv`

Current first-pass metrics:

- `L1` to ground truth
- `LPIPS` to ground truth

These are sanity-only metrics because `image | semantic_mask` is a one-to-many
task. They do not replace the actual first-look readout:

- does the generated image respect the input mask layout?
- does the texture look plausible?
- does `NFE=4/2` stay stable, or collapse into random texture?

PSNR / SSIM are not the primary metrics for this route.

## Minimal Success Criteria

### Tiny Overfit

On 32 or 64 samples:

- the model must clearly follow the mask layout
- object regions and background regions must not be random swaps

### Validation Sweep

On normal validation masks:

- `NFE=4/2` should be clearly better than random texture or layout mismatch
- if `NFE=8` still cannot respect the mask, first suspect:
  - tokenizer quality
  - conditioning path
  - training budget

Do not blame one-step difficulty first if even `NFE=8` cannot render the mask
layout coherently.

For the fixed condition-path benchmark protocol, use
[docs/mask_conditioned_renderer_benchmark.md](docs/mask_conditioned_renderer_benchmark.md).

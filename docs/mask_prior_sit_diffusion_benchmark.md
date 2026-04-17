# SiT Mask Prior Benchmark

This document defines the next apples-to-apples benchmark for the project-layer
upstream route:

- task: `p(semantic_mask)`
- task type: unconditional semantic layout generation
- representation: direct mask-space modeling under fixed project-layer scripts
- no mask tokenizer
- no image branch
- no change to the frozen `p(image | semantic_mask)` renderer

The purpose of this step is narrow and explicit:

- test whether a stronger SiT-style transformer backbone plus either a
  conventional diffusion control or a direct discrete baseline is a better fit
  for unconditional semantic-mask generation than the current U-Net +
  AlphaFlow baseline
- keep the evaluation protocol fixed so the comparison stays
  apples-to-apples

This step does **not** claim that flow is dead. It only asks whether the
current direct-mask U-Net flow baseline is being limited by backbone capacity
and objective choice. If a SiT-style baseline wins under the same protocol, the
next comparison should be the same SiT-style backbone under flow or AlphaFlow.

## Compared Routes

Legacy baseline:

- backbone: project-layer `LatentIntervalUNet`
- objective: AlphaFlow
- config: `configs/latent_alphaflow_mask_prior_unet.yaml`
- train entrypoint: `scripts/train_mask_prior.py`
- sample entrypoint: `scripts/sample_mask_prior.py`

Continuous SiT control baseline:

- backbone: project-layer `LatentIntervalSiT`
- objective: standard Gaussian diffusion training
- sampler: DDIM-style deterministic few-step sampler
- config: `configs/latent_diffusion_mask_prior_sit.yaml`
- train entrypoint: `scripts/train_mask_prior_diffusion.py`
- sample entrypoint: `scripts/sample_mask_prior_diffusion.py`

Current benchmark candidate:

- backbone: project-layer `LatentIntervalSiT`
- objective: absorbing-mask discrete diffusion over `mask_index`
- sampler: seeded stochastic few-step progressive reveal
- configs:
  - `configs/discrete_mask_prior_sit_tiny.yaml`
  - `configs/discrete_mask_prior_sit.yaml`
- train entrypoint: `scripts/train_mask_prior_diffusion.py`
- sample entrypoint: `scripts/sample_mask_prior_diffusion.py`

All three routes still model `p(semantic_mask)` only.

## Fixed Comparison Matrix

Keep the following fixed across the benchmark:

- same dataset root: `outputs/segmentation_teacher_data/remote_semantic_indexed`
- same label spec: `configs/label_specs/remote_semantic.yaml`
- same task: unconditional semantic-mask generation
- same downstream interface: generated outputs must end as discrete
  `mask_index` maps
- same frozen downstream renderer
- same frozen in-domain teacher
- same mask-only metrics
- same compose-to-image metrics
- same `NFE=8/4/2/1` sweep
- same seed

Do **not** change the renderer config, renderer checkpoint rule, or teacher
workflow while running this comparison.

## Why The Discrete Phase A Baseline Exists

The continuous SiT diffusion route stays checked in as a control, but the
current Phase A upgrade changes the state object itself for three reasons:

1. The failure mode in the continuous route is not just low capacity. It is
   also a mismatch between the true target and the modeled state: final masks
   are discrete semantic variables, not continuous one-hot fields.
2. This Phase A patch changes only one major variable at the project layer:
   state semantics. It still avoids a tokenizer, an image branch, and any
   structure auxiliary losses.
3. If this direct discrete route improves mask-only and compose metrics under
   the same frozen protocol, the next comparison against SiT + AlphaFlow
   becomes much clearer.

## Tiny Pilot

Use the tiny config for shape checks and deliberate overfit.

Continuous one-hot control:

```bash
python scripts/train_mask_prior_diffusion.py \
  --config configs/latent_diffusion_mask_prior_sit_tiny.yaml \
  --scale-lr true \
  --gpus 0
```

Direct discrete Phase A:

```bash
python scripts/train_mask_prior_diffusion.py \
  --config configs/discrete_mask_prior_sit_tiny.yaml \
  --scale-lr true \
  --gpus 0
```

Success criteria for the tiny pilot:

- clearly memorize the dominant large-region layout
- main class boundaries remain visible
- thin or narrow structures do not collapse completely

## Base Training

```bash
python scripts/train_mask_prior_diffusion.py \
  --config configs/discrete_mask_prior_sit.yaml \
  --scale-lr true \
  --gpus 0
```

Sample a fixed few-step sweep:

```bash
python scripts/sample_mask_prior_diffusion.py \
  --config configs/discrete_mask_prior_sit.yaml \
  --ckpt <best-mask-prior-ckpt> \
  --outdir outputs/mask_prior_samples/discrete_sit \
  --n-samples 32 \
  --batch-size 8 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

The sampling and evaluation scripts in this benchmark require explicit
checkpoint paths. They do not fall back to `last.ckpt`.

Phase A notes:

- the sampler is seeded stochastic, not pure deterministic MAP fill-in
- `sample_latents(..., noise=...)` remains meaningful because the existing
  project-layer scripts reuse it for reproducible few-step sweeps
- this patch does not add boundary / area / adjacency / topology auxiliary
  losses yet

## Shared Evaluation Protocol

The benchmark intentionally reuses the checked-in evaluation scripts without
changing their metric definitions.

Mask-only evaluation:

```bash
python scripts/eval_mask_prior.py \
  --config configs/discrete_mask_prior_sit.yaml \
  --ckpt <best-mask-prior-ckpt> \
  --outdir outputs/mask_prior_eval/discrete_sit \
  --n-samples 32 \
  --batch-size 8 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

Compose-to-image evaluation:

```bash
python scripts/eval_mask_prior_composed_renderer.py \
  --mask-config configs/discrete_mask_prior_sit.yaml \
  --mask-ckpt <best-mask-prior-ckpt> \
  --renderer-config configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid_boundary_encoder.yaml \
  --renderer-ckpt <best-renderer-ckpt> \
  --renderer-tokenizer-config configs/autoencoder_image_lpips_adv_256.yaml \
  --renderer-tokenizer-ckpt <best-image-tokenizer-ckpt> \
  --teacher-run-dir logs/segmentation_teacher/<winner_run> \
  --label-spec configs/label_specs/remote_semantic.yaml \
  --outdir outputs/mask_prior_compose_eval/discrete_sit \
  --n-samples 32 \
  --mask-nfe-values 8 4 2 1 \
  --renderer-nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

## Minimum Readout Standard

For the new baseline to count as a useful upgrade, the expectation is:

- `NFE=8/4` mask samples are visibly more coherent than the U-Net + AlphaFlow
  baseline
- mask-only distribution statistics do not collapse
- compose-to-image teacher metrics are clearly better than the legacy U-Net
  baseline under the same frozen renderer and teacher

If this Phase A discrete baseline still fails, then the next variables to
consider are:

- structure-aware auxiliary losses
- a semantic-mask tokenizer
- stronger flow variants such as OT-CFM
- a same-backbone SiT + AlphaFlow comparison

Do not introduce those extra variables in this benchmark step.

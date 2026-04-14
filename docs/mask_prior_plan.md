# Mask Prior Plan

## Route Definition

This document defines a new project-layer baseline for unconditional semantic-mask generation:

- route: `p(semantic_mask)`
- task type: unconditional multi-class semantic layout generation
- representation: direct `mask_onehot` field during training, `argmax` back to `mask_index` at sampling

This route is explicitly separate from the current renderer route:

- `p(mask)` models semantic layout by itself
- `p(image | mask)` renders an image conditioned on a semantic mask

The current project decomposition is therefore:

`p(mask) + p(image | mask)`

This document does **not** redefine the current renderer mathematics. It adds a new upstream route in parallel.

## Why This First Baseline Is Conservative

The first `p(mask)` baseline intentionally does **not** use a mask tokenizer.

Reasons:

1. Remote-sensing semantic masks are layout-heavy and topology-heavy, so the first question is whether a few-step flow prior can capture large-scale layout directly.
2. A tokenizer would add another compression bottleneck and another potential failure mode before the unconditional prior itself has been validated.
3. Direct mask-space training keeps failure analysis simple:
   - global layout wrong
   - boundaries unstable
   - thin structures missing
   - collapse or fragmentation under low NFE

The first `p(mask)` baseline also intentionally does **not** model `p(image, mask)` jointly.

Reasons:

1. Joint modeling would re-couple mask generation with image appearance, making it harder to locate the actual failure source.
2. The project already has a validated `p(image | mask)` route, so the more natural next step is to solve the missing upstream factor by itself.
3. The project goal is now operationally decomposed, not fused.

## Baseline Design

The first main baseline is:

- trainer: `latent_meanflow.trainers.mask_prior_trainer.MaskPriorTrainer`
- backbone: project-layer `LatentIntervalUNet`
- objective: AlphaFlow
- sampler: interval sampler with fixed `NFE=8/4/2/1` evaluation sweep
- decoding: sampled `K`-channel field -> `argmax` -> discrete class-id mask

Interpretation:

- the model learns a direct flow prior over `K x H x W` semantic-mask fields
- there is no image branch
- there is no mask tokenizer
- there is no latent downsampling beyond the native mask resolution configured for training

## Deliverables

Project-layer files for this route:

- `latent_meanflow/trainers/mask_prior_trainer.py`
- `latent_meanflow/data/semantic_mask.py`
- `scripts/train_mask_prior.py`
- `scripts/sample_mask_prior.py`
- `scripts/eval_mask_prior.py`
- `configs/latent_alphaflow_mask_prior_unet_tiny.yaml`
- `configs/latent_alphaflow_mask_prior_unet.yaml`

## Success Criteria

Tiny overfit target:

- remember large contiguous class regions
- recover main boundaries
- let narrow roads / thin channels at least partially emerge

Base target:

- `NFE=8/4` should not collapse into random local fragments or pure noise-like blocks
- if `NFE=8` still lacks global layout coherence, the next upgrade should be backbone capacity, not a premature one-step blame cycle

## Recommended Workflow

1. Overfit with the tiny config.
2. Verify that `mask_raw/`, `mask_color/`, and `panel/` all look structurally coherent.
3. Train the base AlphaFlow route.
4. Run the fixed `NFE=8/4/2/1` sweep.
5. Evaluate generated masks against the real-mask bank before composing with the renderer.

## Commands

Tiny pilot:

```bash
python scripts/train_mask_prior.py \
  --config configs/latent_alphaflow_mask_prior_unet_tiny.yaml \
  --gpus 0
```

Base AlphaFlow training:

```bash
python scripts/train_mask_prior.py \
  --config configs/latent_alphaflow_mask_prior_unet.yaml \
  --gpus 0
```

Sampling sweep:

```bash
python scripts/sample_mask_prior.py \
  --config configs/latent_alphaflow_mask_prior_unet.yaml \
  --ckpt <best_or_last_ckpt> \
  --outdir outputs/mask_prior_samples/alphaflow_unet \
  --n-samples 32 \
  --batch-size 8 \
  --nfe-values 8 4 2 1 \
  --overwrite
```

Evaluation:

```bash
python scripts/eval_mask_prior.py \
  --config configs/latent_alphaflow_mask_prior_unet.yaml \
  --ckpt <best_or_last_ckpt> \
  --outdir outputs/mask_prior_eval/alphaflow_unet \
  --n-samples 32 \
  --batch-size 8 \
  --nfe-values 8 4 2 1 \
  --overwrite
```

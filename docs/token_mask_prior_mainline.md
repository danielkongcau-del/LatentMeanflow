# Token-Code Mask Generator Mainline

This document describes the current upstream project-layer `p(mask)` mainline:

`token-code mask generator -> frozen tokenizer decode -> semantic_mask -> frozen image renderer`

The goal of this route is to model semantic layout through the code indices of
the frozen balanced VQ semantic-mask tokenizer, not through raw pixel-space
class ids.

## Frozen Tokenizer Contract

The upstream tokenizer is pinned to:

- config: `configs/semantic_mask_vq_tokenizer_main_balanced_256.yaml`
- checkpoint: explicit runtime path, required by train/sample/eval workflows

The token-mask generator treats that tokenizer as frozen infrastructure:

- tokenizer weights stay frozen
- tokenizer runs in `eval()` mode
- tokenizer encode path produces training targets as code indices
- tokenizer decode path converts sampled code grids back into `semantic_mask`
- tokenizer code-grid shape comes from the tokenizer itself

This route does **not** fine-tune the tokenizer during prior training.

## Checked-In Files

Configs:

- `configs/token_mask_prior_vq_sit_tiny.yaml`
- `configs/token_mask_prior_vq_sit.yaml`
- `configs/token_mask_prior_vq_sit_tiny_control.yaml`
- `configs/token_mask_prior_vq_sit_control.yaml`
- `configs/diagnostics/token_mask_prior_vq_sit_memorize_1.yaml`
- `configs/diagnostics/token_mask_prior_vq_sit_memorize_4.yaml`

Python entrypoints:

- `scripts/train_token_mask_prior.py`
- `scripts/sample_token_mask_prior.py`
- `scripts/eval_token_mask_prior.py`

Trainer:

- `latent_meanflow/trainers/token_mask_prior_trainer.py`

Tests:

- `tests/test_token_mask_prior_smoke.py`

## Route Semantics

The current promoted mainline is the refinement upgrade:

- unconditional `p(code_indices)` only
- project-layer discrete diffusion objective and sampler reused from the older direct-discrete route
- absorbing `MASK` token at `codebook_size`
- final sampled grids must contain only valid tokenizer code ids before decode
- objective-side corruption uses `exact_count`
- mainline sampler uses `proposal_visible_refine`
- early proposals can stay visible for later refinement instead of being
  permanently locked by the first reveal decision

Mainline configs:

- `configs/token_mask_prior_vq_sit.yaml`
- `configs/token_mask_prior_vq_sit_tiny.yaml`

Control configs:

- `configs/token_mask_prior_vq_sit_control.yaml`
- `configs/token_mask_prior_vq_sit_tiny_control.yaml`
- these keep the older `progressive_reveal` semantics for ablation and rollback

This route is separate from both:

- legacy direct pixel-space `p(semantic_mask)` priors
- the downstream `p(image | semantic_mask)` renderer

## Commands

Train the main token-code baseline:

```bash
python scripts/train_token_mask_prior.py \
  --config configs/token_mask_prior_vq_sit.yaml \
  --tokenizer-ckpt /path/to/semantic_mask_vq_tokenizer_balanced.ckpt \
  --scale-lr true \
  --gpus 0
```

Train the old progressive-reveal control:

```bash
python scripts/train_token_mask_prior.py \
  --config configs/token_mask_prior_vq_sit_control.yaml \
  --tokenizer-ckpt /path/to/semantic_mask_vq_tokenizer_balanced.ckpt \
  --scale-lr true \
  --gpus 0
```

Run the tiny memorize diagnostics:

```bash
python scripts/train_token_mask_prior.py \
  --config configs/diagnostics/token_mask_prior_vq_sit_memorize_1.yaml \
  --tokenizer-ckpt /path/to/semantic_mask_vq_tokenizer_balanced.ckpt \
  --scale-lr false \
  --gpus 0

python scripts/train_token_mask_prior.py \
  --config configs/diagnostics/token_mask_prior_vq_sit_memorize_4.yaml \
  --tokenizer-ckpt /path/to/semantic_mask_vq_tokenizer_balanced.ckpt \
  --scale-lr false \
  --gpus 0
```

Sample code grids and decode them into semantic masks:

```bash
python scripts/sample_token_mask_prior.py \
  --config configs/token_mask_prior_vq_sit.yaml \
  --ckpt /path/to/token_mask_prior.ckpt \
  --tokenizer-config configs/semantic_mask_vq_tokenizer_main_balanced_256.yaml \
  --tokenizer-ckpt /path/to/semantic_mask_vq_tokenizer_balanced.ckpt \
  --outdir outputs/token_mask_prior_sample/main \
  --n-samples 32 \
  --batch-size 8 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

Evaluate mask-only decoded semantic-mask quality:

```bash
python scripts/eval_token_mask_prior.py \
  --config configs/token_mask_prior_vq_sit.yaml \
  --ckpt /path/to/token_mask_prior.ckpt \
  --tokenizer-config configs/semantic_mask_vq_tokenizer_main_balanced_256.yaml \
  --tokenizer-ckpt /path/to/semantic_mask_vq_tokenizer_balanced.ckpt \
  --outdir outputs/token_mask_prior_eval/main \
  --n-samples 32 \
  --batch-size 8 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

Compose the token-code route through the frozen renderer:

```bash
python scripts/eval_mask_prior_composed_renderer.py \
  --mask-config configs/token_mask_prior_vq_sit.yaml \
  --mask-ckpt /path/to/token_mask_prior.ckpt \
  --mask-tokenizer-config configs/semantic_mask_vq_tokenizer_main_balanced_256.yaml \
  --mask-tokenizer-ckpt /path/to/semantic_mask_vq_tokenizer_balanced.ckpt \
  --renderer-config configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid_boundary_encoder.yaml \
  --renderer-ckpt /path/to/mask_conditioned_renderer.ckpt \
  --renderer-tokenizer-config configs/autoencoder_image_lpips_adv_256.yaml \
  --renderer-tokenizer-ckpt /path/to/image_tokenizer.ckpt \
  --teacher-run-dir logs/segmentation_teacher/<winner_run> \
  --label-spec configs/label_specs/remote_semantic.yaml \
  --outdir outputs/token_mask_prior_compose_eval/main \
  --n-samples 32 \
  --mask-nfe-values 8 4 2 1 \
  --renderer-nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

## Outputs

Sampling exports:

- `code_raw/`
- `mask_raw/`
- `mask_color/`
- `boundary/`
- `panel/`
- `summary.json`
- `summary.csv`
- `summary.md`

Evaluation adds:

- decoded semantic-mask distribution summaries
- token-usage diagnostics such as active-code count and code perplexity
- route metadata such as `refinement_mode`, `corruption_mode`, and reveal /
  lock-noise settings
- compose-to-image metrics through the frozen renderer and frozen teacher

## Still Not Implemented Yet

The current mainline intentionally does **not** claim:

- autoregressive token prior
- class-conditional token prior
- joint image+mask generation
- tokenizer fine-tuning inside prior training
- renderer architecture changes

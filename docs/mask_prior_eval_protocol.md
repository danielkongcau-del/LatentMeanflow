# Mask Prior Evaluation Protocol

This document defines the fixed research-style evaluation protocol for the
project-layer unconditional semantic-mask route:

- upstream route: `p(semantic_mask)` with the token-code mask generator as the current mainline
- downstream frozen evaluator: `p(image | semantic_mask)`
- frozen judge: in-domain segmentation teacher

The project is now operationally decomposed as:

`p(mask) + p(image | semantic_mask)`

This protocol does not change training math. It only fixes how `p(mask)` is
evaluated.

## Why `p(mask)` Needs Two Layers Of Evaluation

For unconditional semantic-mask generation, there is no paired target mask for
each sample. That means pixelwise `L1`, paired `IoU`, or similar one-to-one
metrics are not valid as the primary readout.

The protocol therefore splits evaluation into two questions:

1. `mask-only distribution quality`
   - Does the generated mask bank match the real split on area, topology,
     adjacency, boundary complexity, holes, thin structures, and small-region behavior?

2. `compose-to-image quality`
   - If the generated masks are fed into the already validated frozen renderer,
     do they still produce structurally valid images under the frozen in-domain
     teacher?

Looking only at generated mask pictures is not enough. A mask can look locally
plausible while still having the wrong global area balance, too many fragments,
collapsed small regions, or layout statistics that break once composed through
the renderer.

## Fixed Protocol Rules

- Best checkpoint selection must use `val/base_error_mean`.
- Do not choose checkpoints by latest timestamp or train visuals.
- `scripts/train_token_mask_prior.py`, `scripts/sample_token_mask_prior.py`, and `scripts/eval_token_mask_prior.py` require an explicit frozen tokenizer checkpoint for the balanced VQ tokenizer.
- `scripts/sample_token_mask_prior.py` and `scripts/eval_token_mask_prior.py` require explicit token-mask-prior checkpoints and do not fall back to `last.ckpt`.
- `scripts/sample_mask_prior.py` and `scripts/eval_mask_prior.py` require explicit checkpoint paths and do not fall back to `last.ckpt`.
- Do not report only `NFE=1`.
- The default `p(mask)` sweep is `NFE=8/4/2/1`.
- The default frozen renderer sweep is also `NFE=8/4/2/1`.
- Use the same seed and the same sample count in every comparison.
- The renderer and teacher are frozen during compose evaluation.

## Mask-Only Metrics Answer What

Mask-only metrics answer:

- whether class area proportions are preserved
- whether class-to-class contact patterns stay realistic
- whether the generated masks keep the right amount of fragmentation
- whether each class still keeps a dominant connected core instead of shattering
- whether enclosed holes or islands appear with the right frequency
- whether boundaries stay present instead of being oversmoothed
- whether small connected regions still exist instead of disappearing
- whether thin roads, waterways, or narrow strips remain continuous instead of breaking into fragments

The mask-only protocol reports, for both the real split and generated masks:

- class area histogram
- per-class area ratio statistics
- 4-neighbor class adjacency matrix statistics
- connected component count statistics
- connected component size statistics
- largest connected-component share per class
- hole count / enclosed-region area statistics
- boundary length statistics
- small-region frequency statistics
- optional thin-structure continuity statistics for user-specified class ids

These additions are especially useful for remote sensing:

- adjacency matters because land-use realism depends on which classes touch,
  not only how much area they occupy
- largest-component share catches parcel/road/water masks that fragment into
  implausible islands
- hole statistics catch enclosed voids or islands inside large regions
- thin-structure continuity is a direct sanity check for roads, rivers, and
  other narrow elongated targets

The main script is:

```bash
python scripts/eval_token_mask_prior.py \
  --config configs/token_mask_prior_vq_sit.yaml \
  --ckpt <best-token-mask-prior-ckpt> \
  --tokenizer-config configs/semantic_mask_vq_tokenizer_main_balanced_256.yaml \
  --tokenizer-ckpt <best-balanced-tokenizer-ckpt> \
  --outdir outputs/token_mask_prior_eval/main \
  --n-samples 32 \
  --batch-size 8 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

For the current promoted remask-low-confidence mainline, keep
`configs/token_mask_prior_vq_sit.yaml`. For the proposal-visible rollback
control, swap only the config path to
`configs/token_mask_prior_vq_sit_control.yaml` while keeping the same frozen
balanced tokenizer contract.

To evaluate the older direct pixel-space AlphaFlow control baseline, use the
legacy evaluator:

```bash
python scripts/eval_mask_prior.py \
  --config configs/latent_alphaflow_mask_prior_unet.yaml \
  --ckpt <best-mask-prior-ckpt> \
  --outdir outputs/mask_prior_eval/alphaflow_unet \
  --n-samples 32 \
  --batch-size 8 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

To evaluate the SiT-style continuous control baseline, keep the same legacy
script and swap only the config/checkpoint pair:

```bash
python scripts/eval_mask_prior.py \
  --config configs/latent_diffusion_mask_prior_sit.yaml \
  --ckpt <best-mask-prior-ckpt> \
  --outdir outputs/mask_prior_eval/sit_diffusion \
  --n-samples 32 \
  --batch-size 8 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

To evaluate the direct discrete semantic-variable control baseline, keep the
same legacy script and swap only the config/checkpoint pair again:

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

Outputs for the token-code mainline:

- `summary.json`
- `summary.csv`
- `summary.md`

The token-code summaries also record route metadata such as
`refinement_mode`, `corruption_mode`, `final_full_reveal`, and the reveal /
lock-noise settings so refine runs and progressive-reveal controls remain
separable during reporting. `summary.json/csv/md` now also expose adjacency,
largest-component share, hole, and optional thin-structure gap metrics.

## Compose-To-Image Metrics Answer What

Compose metrics answer:

- whether sampled masks are still valid when used as renderer inputs
- whether the frozen renderer can turn them into structurally coherent remote
  sensing images
- whether the sampled mask distribution preserves the semantics that the frozen
  in-domain teacher can recover from rendered RGB

The compose protocol is:

1. sample masks from `p(mask)`
2. feed them into the frozen `p(image | semantic_mask)` renderer
3. run the frozen in-domain teacher on the rendered RGB images
4. compare teacher masks to the input masks

Primary compose metrics:

- `teacher_miou`
- `teacher_per_class_iou`
- `boundary_f1`
- `layout_pixel_accuracy`
- `small_region_miou`

The compose evaluator also writes structure-gap summaries between the input
masks and the teacher-decoded masks, including adjacency and
largest-component-share gaps. This helps separate renderer-induced drift from
upstream `p(mask)` layout failures.

The main script is:

```bash
python scripts/eval_mask_prior_composed_renderer.py \
  --mask-config configs/token_mask_prior_vq_sit.yaml \
  --mask-ckpt <best-token-mask-prior-ckpt> \
  --mask-tokenizer-config configs/semantic_mask_vq_tokenizer_main_balanced_256.yaml \
  --mask-tokenizer-ckpt <best-balanced-tokenizer-ckpt> \
  --renderer-config configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid_boundary_encoder.yaml \
  --renderer-ckpt <best-renderer-ckpt> \
  --renderer-tokenizer-config configs/autoencoder_image_lpips_adv_256.yaml \
  --renderer-tokenizer-ckpt <best-image-tokenizer-ckpt> \
  --teacher-run-dir logs/segmentation_teacher/<winner_run> \
  --label-spec configs/label_specs/remote_semantic.yaml \
  --outdir outputs/token_mask_prior_compose_eval/main \
  --n-samples 32 \
  --mask-nfe-values 8 4 2 1 \
  --renderer-nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

Again, use `configs/token_mask_prior_vq_sit.yaml` for the promoted refine
mainline and `configs/token_mask_prior_vq_sit_control.yaml` for the old
progressive-reveal control.

The same compose evaluator also applies to the older direct pixel-space
AlphaFlow control baseline by swapping only `--mask-config` and `--mask-ckpt`
and omitting the tokenizer override flags:

```bash
python scripts/eval_mask_prior_composed_renderer.py \
  --mask-config configs/latent_alphaflow_mask_prior_unet.yaml \
  --mask-ckpt <best-mask-prior-ckpt> \
  --renderer-config configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid_boundary_encoder.yaml \
  --renderer-ckpt <best-renderer-ckpt> \
  --renderer-tokenizer-config configs/autoencoder_image_lpips_adv_256.yaml \
  --renderer-tokenizer-ckpt <best-image-tokenizer-ckpt> \
  --teacher-run-dir logs/segmentation_teacher/<winner_run> \
  --label-spec configs/label_specs/remote_semantic.yaml \
  --outdir outputs/mask_prior_compose_eval/alphaflow_unet \
  --n-samples 32 \
  --mask-nfe-values 8 4 2 1 \
  --renderer-nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

The same compose evaluator also applies to the SiT-style continuous control
baseline by swapping only `--mask-config` and `--mask-ckpt` again:

```bash
python scripts/eval_mask_prior_composed_renderer.py \
  --mask-config configs/latent_diffusion_mask_prior_sit.yaml \
  --mask-ckpt <best-mask-prior-ckpt> \
  --renderer-config configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid_boundary_encoder.yaml \
  --renderer-ckpt <best-renderer-ckpt> \
  --renderer-tokenizer-config configs/autoencoder_image_lpips_adv_256.yaml \
  --renderer-tokenizer-ckpt <best-image-tokenizer-ckpt> \
  --teacher-run-dir logs/segmentation_teacher/<winner_run> \
  --label-spec configs/label_specs/remote_semantic.yaml \
  --outdir outputs/mask_prior_compose_eval/sit_diffusion \
  --n-samples 32 \
  --mask-nfe-values 8 4 2 1 \
  --renderer-nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

The direct discrete semantic-variable control uses the same compose evaluator
with the same frozen renderer and teacher:

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

Outputs:

- `summary.json`
- `summary.csv`
- `summary.md`
- generated mask sweep under `mask_prior_generated/`
- composed renderer outputs under `composed_renderer/`

## Fixed NFE Sweep Commands

Resolve the best checkpoints explicitly:

```bash
python scripts/find_checkpoint.py \
  --run-dir logs/<mask_prior_run> \
  --selection best \
  --monitor val/base_error_mean

python scripts/find_image_tokenizer_checkpoint.py \
  --config configs/autoencoder_image_lpips_adv_256.yaml \
  --run-dir logs/<image_tokenizer_run> \
  --selection best

python scripts/find_checkpoint.py \
  --run-dir logs/<renderer_run> \
  --selection best \
  --monitor val/base_error_mean
```

Formal reporting sweep:

```bash
python scripts/eval_mask_prior.py \
  --config configs/latent_alphaflow_mask_prior_unet.yaml \
  --ckpt <best-mask-prior-ckpt> \
  --outdir outputs/mask_prior_eval/alphaflow_unet \
  --n-samples 32 \
  --batch-size 8 \
  --nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite

python scripts/eval_mask_prior_composed_renderer.py \
  --mask-config configs/latent_alphaflow_mask_prior_unet.yaml \
  --mask-ckpt <best-mask-prior-ckpt> \
  --renderer-config configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid_boundary_encoder.yaml \
  --renderer-ckpt <best-renderer-ckpt> \
  --renderer-tokenizer-config configs/autoencoder_image_lpips_adv_256.yaml \
  --renderer-tokenizer-ckpt <best-image-tokenizer-ckpt> \
  --teacher-run-dir logs/segmentation_teacher/<winner_run> \
  --label-spec configs/label_specs/remote_semantic.yaml \
  --outdir outputs/mask_prior_compose_eval/alphaflow_unet \
  --n-samples 32 \
  --mask-nfe-values 8 4 2 1 \
  --renderer-nfe-values 8 4 2 1 \
  --seed 23 \
  --overwrite
```

## Minimum Success Standard

Mask-only layer:

- class area distribution must not collapse or drift badly
- thin boundaries and small connected regions must not disappear entirely

Compose layer:

- frozen renderer outputs should remain structurally interpretable under the
  frozen in-domain teacher
- if compose collapses, suspect the sampled `p(mask)` distribution first rather
  than blaming the already validated renderer route

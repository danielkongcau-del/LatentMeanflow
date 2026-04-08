# Semantic Pair Evaluation Protocol

This document defines the fixed evaluation protocol for paired RGB image +
semantic-mask generation. It is intended for research-style reporting on
few-step latent flow priors and should be used for MeanFlow / AlphaFlow U-Net
comparisons instead of ad-hoc one-step image inspection.

The protocol is project-layer only. It does not change the training objective
math, and it does not modify `third_party/`.

## Protocol Rules

- Best checkpoint selection must use `val/base_error_mean`.
- Do not choose checkpoints by looking only at `val/meanflow_loss`.
- Do not report only `NFE=1`.
- The default reporting sweep is `NFE=8/4/2/1`.
- The same seed, checkpoint rule, teacher source, and label remap must be used
  for every model being compared.

## What The Metrics Mean

- `FID` / `KID`
  - Answer: does the generated RGB image distribution look like the real split?
  - Use these to measure marginal RGB realism and distribution match.

- `mIoU` / `per-class IoU`
  - Answer: does the generated semantic mask agree with the semantics parsed
    from the generated RGB image?
  - For unconditional few-step generation there is no sample-wise ground-truth
    pair, so these are teacher-aligned metrics comparing `M_hat` with
    `S(I_hat)`, not metrics against a matched validation mask.

- `Boundary F1`
  - Answer: do semantic edges in the generated mask line up with semantic edges
    implied by the generated RGB image?
  - This is more sensitive than plain IoU to boundary quality.

- `pair_pixel_accuracy`
  - Answer: at the pixel level, how often does the generated mask agree with
    the teacher segmentation of the generated RGB image?
  - This is a direct pair-consistency score.

## Teacher Requirement

The paired-task protocol requires a fixed segmentation teacher `S(I)`.

Supported routes:

- Load an external teacher with `--teacher-hf-model`
- Reuse precomputed teacher masks with `--teacher-mask-root`

Teacher training inside this repository is not implemented yet. If you want a
custom teacher, train it elsewhere, freeze it, then plug it into this protocol
as a loaded teacher or as precomputed teacher masks.

## Standard Workflow

### 1. Resolve The Best Checkpoint

```bash
python scripts/find_checkpoint.py \
  --run-dir logs/<timestamp>_latent_alphaflow_semantic_256_unet \
  --selection best \
  --monitor val/base_error_mean
```

### 2. Run The Fixed Paired Evaluation Sweep

With a Hugging Face segmentation teacher:

```bash
python scripts/eval_semantic_pair_generation.py \
  --config configs/latent_alphaflow_semantic_256_unet.yaml \
  --ckpt <best-ckpt> \
  --outdir outputs/semantic_pair_eval/alphaflow_unet_base \
  --split validation \
  --seed 23 \
  --n-samples 32 \
  --batch-size 4 \
  --nfe-values 8 4 2 1 \
  --teacher-hf-model <hf-teacher-model-id-or-local-path>
```

With precomputed teacher masks:

```bash
python scripts/eval_semantic_pair_generation.py \
  --config configs/latent_alphaflow_semantic_256_unet.yaml \
  --ckpt <best-ckpt> \
  --outdir outputs/semantic_pair_eval/alphaflow_unet_base \
  --split validation \
  --seed 23 \
  --n-samples 32 \
  --batch-size 4 \
  --nfe-values 8 4 2 1 \
  --teacher-mask-root outputs/precomputed_teacher_masks
```

Optional label remapping for a teacher with different semantic ids:

```bash
python scripts/eval_semantic_pair_generation.py \
  --config configs/latent_alphaflow_semantic_256_unet.yaml \
  --ckpt <best-ckpt> \
  --outdir outputs/semantic_pair_eval/alphaflow_unet_base \
  --nfe-values 8 4 2 1 \
  --teacher-hf-model <hf-teacher-model-id-or-local-path> \
  --teacher-remap-json configs/teacher_remap_example.json
```

### 3. Evaluate An Existing Sweep Without Re-Sampling

```bash
python scripts/eval_semantic_pair_generation.py \
  --config configs/latent_alphaflow_semantic_256_unet.yaml \
  --generated-root outputs/benchmarks/alphaflow_unet_base \
  --outdir outputs/semantic_pair_eval/alphaflow_unet_base_metrics \
  --split validation \
  --seed 23 \
  --n-samples 32 \
  --batch-size 4 \
  --nfe-values 8 4 2 1 \
  --teacher-mask-root outputs/precomputed_teacher_masks
```

## Outputs

For every `NFE`, the evaluation route keeps or produces:

- `image/`
- `mask_raw/`
- `mask_color/`
- `overlay/`

If a live teacher is used, it also writes:

- `teacher_mask_raw/`
- `teacher_mask_color/`
- `teacher_overlay/`

At the root `outdir`, it writes:

- `summary.json`
- `summary.csv`

These are the files to use when writing comparison tables.

## What Counts As A Real Improvement

For paired RGB + semantic-mask generation, a method is only meaningfully better
than the current U-Net baseline if:

- it improves teacher-aligned `mIoU`, `Boundary F1`, and `pair_pixel_accuracy`
  at `NFE=4` and `NFE=2`, and
- it does not materially regress RGB `FID` / `KID` at `NFE=8`, and
- the gains are not limited to a single easy class while small or thin classes
  collapse in `per-class IoU`.

If a candidate only lowers `val/meanflow_loss`, or only looks better at
`NFE=1`, or only improves RGB realism while pair-consistency collapses, it
should not be considered a real paired-task win.

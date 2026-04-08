# Mask-Conditioned Layout Faithfulness Protocol

This document defines the fixed evaluation protocol for the project-layer
`p(image | semantic_mask)` renderer.

It does not change training math. It only fixes how to evaluate whether the
generated image actually obeys the input semantic layout.

## Protocol Rules

- Best checkpoint selection must use `val/base_error_mean`.
- Do not choose checkpoints by looking only at `val/alphaflow_loss`.
- Do not report only `NFE=1`.
- The default reporting sweep is `NFE=8/4/2/1`.
- The same seed, checkpoint rule, teacher source, and sample subset must be
  used for every renderer being compared.

## What The Metrics Mean

- `L1` / `LPIPS`
  - Answer: how close is the generated RGB image to the paired ground-truth RGB?
  - Use these as sanity-only metrics.
  - They are not the primary task metrics because `image | semantic_mask` is
    one-to-many.

- `teacher_miou`
  - Answer: does the generated image obey the input semantic layout?
  - The protocol computes a teacher segmentation `S(I_hat)` on generated RGB
    image `I_hat`, then compares `S(I_hat)` to the input semantic mask `M`.
  - This is the primary metric.

- `teacher_per_class_iou`
  - Answer: which semantic classes are preserved well, and which collapse?
  - Use this to detect whether a gain is only coming from easy large classes.

- `boundary_f1`
  - Answer: do semantic edges in the generated image line up with edges in the
    input mask?
  - This is the main metric for narrow roads, canals, pond rims, and field
    boundaries.

- `layout_pixel_accuracy`
  - Answer: how often does the parsed semantic layout of the generated image
    match the input mask at the pixel level?

- `small_region_miou`
  - Answer: how well does the renderer preserve low-area semantic regions?
  - This is a heuristic subset metric over target regions occupying no more
    than a fixed fraction of valid pixels in a sample.

## Teacher Requirement

This protocol requires a frozen segmentation teacher `S(image)`.

Supported routes:

- load a live teacher with `--teacher-hf-model`
- reuse precomputed teacher masks with `--teacher-mask-root`

Teacher training inside this repository is not implemented yet.

## Standard Workflow

### 1. Resolve The Best Checkpoint

```bash
python scripts/find_checkpoint.py \
  --run-dir logs/<your_run> \
  --selection best \
  --monitor val/base_error_mean
```

### 2. Run The Fixed Layout-Faithfulness Sweep

With a live Hugging Face segmentation teacher:

```bash
python scripts/eval_mask_layout_faithfulness.py \
  --config configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid_boundary.yaml \
  --ckpt <best-ckpt> \
  --outdir outputs/mask_conditioned_layout_eval/fullres_pyramid_boundary \
  --split validation \
  --seed 23 \
  --n-samples 32 \
  --batch-size 4 \
  --nfe-values 8 4 2 1 \
  --teacher-hf-model <hf-teacher-model-id-or-local-path>
```

With precomputed teacher masks:

```bash
python scripts/eval_mask_layout_faithfulness.py \
  --config configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid_boundary.yaml \
  --generated-root outputs/mask_conditioned_renderer_benchmark/fullres_pyramid_boundary \
  --outdir outputs/mask_conditioned_layout_eval/fullres_pyramid_boundary_metrics \
  --split validation \
  --seed 23 \
  --n-samples 32 \
  --batch-size 4 \
  --nfe-values 8 4 2 1 \
  --teacher-mask-root outputs/precomputed_teacher_masks
```

Optional teacher label remap:

```bash
python scripts/eval_mask_layout_faithfulness.py \
  --config configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid_boundary.yaml \
  --ckpt <best-ckpt> \
  --outdir outputs/mask_conditioned_layout_eval/fullres_pyramid_boundary \
  --teacher-hf-model <hf-teacher-model-id-or-local-path> \
  --teacher-remap-json configs/teacher_remap_example.json
```

## Outputs

At the root `outdir`, the protocol writes:

- `summary.json`
- `summary.csv`
- `report.md`

If a live teacher is used, it also writes under each `nfe*/` directory:

- `teacher_mask_raw/`
- `teacher_mask_color/`
- `teacher_overlay/`

The existing generated sample layout remains unchanged:

- `input_mask_raw/`
- `input_mask_color/`
- `generated_image/`
- `ground_truth_image/` when available
- `overlay/`
- `panel/`

## What Counts As A Real Improvement

For `p(image | semantic_mask)`, a renderer is meaningfully better than the
current baseline only if:

- it improves `teacher_miou` and `boundary_f1` at `NFE=4` and `NFE=2`
- it does not regress at `NFE=8`
- the gains are not limited to only easy, large semantic classes
- `small_region_miou` does not collapse on narrow or low-area regions

If a candidate only improves `L1` or `LPIPS`, or only looks better at
`NFE=1`, that is not enough to claim a real renderer upgrade.

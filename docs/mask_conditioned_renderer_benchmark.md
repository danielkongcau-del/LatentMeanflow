# Mask-Conditioned Renderer Benchmark

This runbook compares project-layer condition paths for the checked-in
`p(image | semantic_mask)` renderer.

It does not change:

- tokenizer family within a comparison group
- AlphaFlow math
- seed
- best-checkpoint rule
- `NFE=8/4/2/1` sampling sweep

It only changes the semantic-mask condition path:

- `input_concat`
- `pyramid_concat`
- `pyramid_concat + boundary`
- `fullres_mask + pyramid_concat`
- `fullres_mask + pyramid_concat + boundary`
- `fullres_mask + pyramid_concat + boundary + encoder`

## Fixed Comparison Matrix

Use the same image-only tokenizer checkpoint and the same training budget for:

- `configs/ablations/latent_alphaflow_mask2image_unet_input_concat.yaml`
- `configs/ablations/latent_alphaflow_mask2image_unet_pyramid.yaml`
- `configs/ablations/latent_alphaflow_mask2image_unet_pyramid_boundary.yaml`
- `configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid.yaml`
- `configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid_boundary.yaml`
- `configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid_boundary_encoder.yaml`

Use the base `64x64x4` image-only tokenizer first. The `f=8` tokenizer route is
a follow-up, not the default conditioning benchmark.

## Best Checkpoint Rule

Always resolve the best checkpoint by:

- `val/base_error_mean`

Do not select by:

- `val/alphaflow_loss`
- `last.ckpt`

```bash
python scripts/find_checkpoint.py --run-dir logs/<your_run> --selection best --monitor val/base_error_mean
```

## Training Commands

```bash
python scripts/train_mask_conditioned_image.py \
  --objective alphaflow \
  --config configs/ablations/latent_alphaflow_mask2image_unet_input_concat.yaml \
  --tokenizer-ckpt /path/to/image_tokenizer.ckpt \
  --gpus 0,1
```

```bash
python scripts/train_mask_conditioned_image.py \
  --objective alphaflow \
  --config configs/ablations/latent_alphaflow_mask2image_unet_pyramid.yaml \
  --tokenizer-ckpt /path/to/image_tokenizer.ckpt \
  --gpus 0,1
```

```bash
python scripts/train_mask_conditioned_image.py \
  --objective alphaflow \
  --config configs/ablations/latent_alphaflow_mask2image_unet_pyramid_boundary.yaml \
  --tokenizer-ckpt /path/to/image_tokenizer.ckpt \
  --gpus 0,1
```

```bash
python scripts/train_mask_conditioned_image.py \
  --objective alphaflow \
  --config configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid.yaml \
  --tokenizer-ckpt /path/to/image_tokenizer.ckpt \
  --gpus 0,1
```

```bash
python scripts/train_mask_conditioned_image.py \
  --objective alphaflow \
  --config configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid_boundary.yaml \
  --tokenizer-ckpt /path/to/image_tokenizer.ckpt \
  --gpus 0,1
```

```bash
python scripts/train_mask_conditioned_image.py \
  --objective alphaflow \
  --config configs/ablations/latent_alphaflow_mask2image_unet_fullres_pyramid_boundary_encoder.yaml \
  --tokenizer-ckpt /path/to/image_tokenizer.ckpt \
  --gpus 0,1
```

## Sampling Protocol

Hold checkpoint selection, seed, and few-step sweep fixed across all runs:

```bash
python scripts/sample_mask_conditioned_image.py \
  --config configs/ablations/latent_alphaflow_mask2image_unet_pyramid.yaml \
  --ckpt <best-ckpt> \
  --outdir outputs/mask_conditioned_renderer_benchmark/pyramid \
  --split validation \
  --seed 23 \
  --n-samples 32 \
  --batch-size 4 \
  --nfe-values 8 4 2 1
```

Each run must still save:

- `input_mask_raw/`
- `input_mask_color/`
- `generated_image/`
- `ground_truth_image/` when available
- `overlay/`
- `panel/`

## Evaluation Protocol

```bash
python scripts/eval_mask_layout_faithfulness.py \
  --config configs/ablations/latent_alphaflow_mask2image_unet_pyramid.yaml \
  --ckpt <best-ckpt> \
  --outdir outputs/mask_conditioned_renderer_benchmark/pyramid_eval \
  --split validation \
  --seed 23 \
  --n-samples 32 \
  --batch-size 4 \
  --nfe-values 8 4 2 1 \
  --teacher-hf-model <hf-teacher-model-id-or-local-path>
```

The primary decision rule is teacher-aligned layout faithfulness:

- `teacher_miou`
- `boundary_f1`
- `layout_pixel_accuracy`
- `small_region_miou`

`L1` and `LPIPS` to ground truth remain sanity-only metrics because
`image | semantic_mask` is one-to-many.

## Pass Criteria

Consider the upgrade successful only if the new condition mode:

- is clearly better than `input_concat` on mask boundaries and layout fidelity
  at `NFE=4/2`
- does not regress at `NFE=8`

If `NFE=8` also fails to improve, first suspect:

- tokenizer quality
- training budget
- overall renderer capacity

Do not attribute the failure to one-step difficulty first.

# Segmentation Teacher Bakeoff

This runbook defines the main in-domain segmentation teacher route for the
project-layer `p(image | semantic_mask)` renderer evaluation.

The goal is not to add a new teacher framework. The goal is to use the existing
vendored `third_party/segmentation` codebase as a controlled bakeoff harness on
the real remote-sensing image-mask dataset, pick the best frozen teacher on
held-out data, export teacher masks, and then evaluate renderer layout
faithfulness with `--teacher-mask-root`.

## Why The Main Teacher Must Be In-Domain

The renderer is judged by:

- `teacher_miou`
- `teacher_per_class_iou`
- `boundary_f1`
- `layout_pixel_accuracy`
- `small_region_miou`

If the teacher itself is out-of-domain, then the main renderer metric becomes
noisy exactly where this project cares the most:

- narrow roads
- canals and thin water channels
- pond rims
- field boundaries
- small semantic regions

That is why the main route is:

- train teachers only on real remote-sensing image-mask data
- compare them only on held-out `val` or `test`
- freeze the winner
- export precomputed teacher masks

Online teachers are not the final judge for this route.

## Why SegFormer Is Not In The Main Bakeoff

SegFormer is explicitly excluded from the first-round teacher bakeoff.

Reasons:

1. The repository already contains three cleaner first-round candidates for the
   current goal: a strong robust baseline, a stronger domain-specific candidate,
   and a simple fallback.
2. The old vendored segmentation launcher previously defaulted to `SegFormer`,
   which is not the teacher route we want to standardize around.
3. The first-round teacher goal is not architecture novelty. It is a stable,
   in-domain evaluation teacher for the current renderer protocol.

First-round candidates are restricted to:

- `deeplabv3-resnet`
- `csnet`
- `unet`

Recommended execution order:

1. `deeplabv3-resnet`
2. `csnet`
3. `unet`

Interpretation:

- `deeplabv3-resnet`: robust teacher baseline
- `csnet`: stronger domain-specific candidate from the vendored remote-sensing repo
- `unet`: simple fallback and sanity baseline

## Data Preparation

The real dataset uses grayscale semantic masks defined by:

- `configs/label_specs/remote_semantic.yaml`

The vendored segmentation trainer expects contiguous class ids `0..K-1`. The
project-layer preparation step remaps the raw grayscale mask values into an
indexed dataset view without modifying the source dataset.

Prepare the teacher dataset view:

```bash
python scripts/prepare_segmentation_teacher_data.py \
  --src-root data/remote \
  --dst-root outputs/segmentation_teacher_data/remote_semantic_indexed \
  --label-spec configs/label_specs/remote_semantic.yaml \
  --splits train val test
```

This writes:

```text
outputs/segmentation_teacher_data/remote_semantic_indexed/
  dataset_manifest.json
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

Teacher training must use:

- only real RGB images from the real dataset
- only real semantic masks from the real dataset
- no generated images

## Unified First-Round Recipe

Keep the first-round candidate comparison fixed:

- input size: `512 x 512`
- batch size: `4`
- epoch budget: `120`
- train split: `train`
- model selection split: `val`
- final held-out comparison split: `val` first, `test` after narrowing

Checkpoint selection rule:

- use the vendored trainer's explicit `*_best.pt`
- that checkpoint is selected by held-out validation `mIoU`
- do not select by latest timestamp
- do not select by train visuals

## Training Commands

All three candidates should use the same prepared dataset root, image size,
batch size, and epoch budget.

### 1. `deeplabv3-resnet`

```bash
python third_party/segmentation/train.py \
  --net-name deeplabv3-resnet \
  --out-channels 7 \
  --height 512 \
  --width 512 \
  --batch-size 4 \
  --epoch 120 \
  --train-set outputs/segmentation_teacher_data/remote_semantic_indexed/train \
  --val-set outputs/segmentation_teacher_data/remote_semantic_indexed/val \
  --test-set outputs/segmentation_teacher_data/remote_semantic_indexed/test \
  --save-dir logs/segmentation_teacher/deeplabv3_resnet_h512_w512
```

### 2. `csnet`

```bash
python third_party/segmentation/train.py \
  --net-name csnet \
  --out-channels 7 \
  --height 512 \
  --width 512 \
  --batch-size 4 \
  --epoch 120 \
  --train-set outputs/segmentation_teacher_data/remote_semantic_indexed/train \
  --val-set outputs/segmentation_teacher_data/remote_semantic_indexed/val \
  --test-set outputs/segmentation_teacher_data/remote_semantic_indexed/test \
  --save-dir logs/segmentation_teacher/csnet_h512_w512
```

### 3. `unet`

```bash
python third_party/segmentation/train.py \
  --net-name unet \
  --out-channels 7 \
  --height 512 \
  --width 512 \
  --batch-size 4 \
  --epoch 120 \
  --train-set outputs/segmentation_teacher_data/remote_semantic_indexed/train \
  --val-set outputs/segmentation_teacher_data/remote_semantic_indexed/val \
  --test-set outputs/segmentation_teacher_data/remote_semantic_indexed/test \
  --save-dir logs/segmentation_teacher/unet_h512_w512
```

## Candidate Comparison

Run the held-out bakeoff on the same split for every candidate:

```bash
python scripts/eval_segmentation_teacher_candidates.py \
  --dataset-root outputs/segmentation_teacher_data/remote_semantic_indexed \
  --split val \
  --label-spec configs/label_specs/remote_semantic.yaml \
  --candidate-run deeplabv3_resnet=logs/segmentation_teacher/deeplabv3_resnet_h512_w512 \
  --candidate-run csnet=logs/segmentation_teacher/csnet_h512_w512 \
  --candidate-run unet=logs/segmentation_teacher/unet_h512_w512 \
  --outdir outputs/segmentation_teacher_bakeoff/val
```

Outputs:

- `summary.json`
- `summary.csv`
- `summary.md`

Reported metrics:

- `mIoU`
- `per-class IoU`
- `pixel accuracy`
- `boundary_f1`
- `small_class_miou`
- `worst_class_iou`

Selection rule:

- do not select only by total `mIoU`
- check `boundary_f1`
- check `small_class_miou`
- inspect the lowest-IoU classes
- if roads / channels / boundaries correspond to known class ids, rerun with
  `--focus-class-ids <id ...>` and inspect `focus_class_mean_iou`

Recommended decision flow:

1. Pick the best validation candidate by the bakeoff summary, not by train
   visuals.
2. Re-run the narrowed candidates on `test`.
3. Freeze the winner.

## Export Precomputed Teacher Masks

After choosing the winning teacher, export masks on the renderer outputs in the
directory format expected by the current layout-faithfulness protocol.

Example:

```bash
python scripts/export_teacher_masks.py \
  --run-dir logs/segmentation_teacher/deeplabv3_resnet_h512_w512 \
  --generated-root outputs/mask_conditioned_renderer_benchmark/fullres_pyramid_boundary \
  --split validation \
  --outdir outputs/precomputed_teacher_masks/deeplabv3_resnet_validation
```

This writes:

```text
outputs/precomputed_teacher_masks/deeplabv3_resnet_validation/
  summary.json
  summary.csv
  summary.md
  nfe8/
    teacher_mask_raw/
    teacher_mask_color/
    teacher_overlay/
  nfe4/
    ...
```

This exported root is the object passed to `--teacher-mask-root`.

## Renderer Evaluation With The Frozen Teacher

Main route:

```bash
python scripts/eval_mask_layout_faithfulness.py \
  --config configs/latent_alphaflow_mask2image_unet.yaml \
  --generated-root outputs/mask_conditioned_renderer_benchmark/fullres_pyramid_boundary \
  --outdir outputs/mask_conditioned_layout_eval/fullres_pyramid_boundary \
  --split validation \
  --seed 23 \
  --n-samples 32 \
  --batch-size 4 \
  --nfe-values 8 4 2 1 \
  --teacher-mask-root outputs/precomputed_teacher_masks/deeplabv3_resnet_validation
```

Sanity-check-only route:

```bash
python scripts/eval_mask_layout_faithfulness.py \
  --config configs/latent_alphaflow_mask2image_unet.yaml \
  --generated-root outputs/mask_conditioned_renderer_benchmark/fullres_pyramid_boundary \
  --outdir outputs/mask_conditioned_layout_eval/fullres_pyramid_boundary_hf_sanity \
  --split validation \
  --seed 23 \
  --nfe-values 8 4 2 1 \
  --teacher-hf-model <hf-teacher-model-id-or-local-path>
```

Interpretation:

- live HF teacher: sanity check only
- in-domain trained teacher + exported masks: main evaluation route

## Practical Notes

- The first-round bakeoff is intentionally narrow. Do not widen the candidate
  set until the first three candidates are compared cleanly.
- Keep the data split, input size, batch size, and epoch budget fixed across
  candidates.
- Do not train the teacher on generated renderer images.
- Do not use train visual quality as the winner criterion.
- Do not treat an online teacher as the final judge for the renderer.

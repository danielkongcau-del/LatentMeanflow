# First Semantic Pair Experiment Plan

This runbook is the minimum experiment order for the current project-layer stack:

`semantic dataset -> semantic tokenizer -> latent FM -> latent MeanFlow`

It is intentionally biased toward getting the first stable paired `RGB image + semantic mask` results today. AlphaFlow is not the focus of this first round.

Treat this document as the historical MeanFlow-first runbook. For the current
few-step U-Net AlphaFlow main line, use
[docs/unet_alphaflow_training_plan.md](unet_alphaflow_training_plan.md).

## Assumptions

- Commands below use the fixed `lmf` environment through `D:\Anaconda\envs\lmf\python.exe`.
- The dataset root is `data/remote`.
- The shared label spec is [configs/label_specs/remote_semantic.yaml](../configs/label_specs/remote_semantic.yaml).
- `third_party/` is frozen.

## Config Tiers

- `smoke`: single-purpose correctness check, usually synthetic or tiny.
- `tiny/debug`: short real-data run to verify training stability before spending GPU time.
- `project baseline`: the config to use for the first real run on your dataset.

Current configs in this runbook:

- tokenizer tiny/debug: [configs/semantic_tokenizer_tiny_256.yaml](../configs/semantic_tokenizer_tiny_256.yaml)
- tokenizer baseline: [configs/autoencoder_semantic_pair_256.yaml](../configs/autoencoder_semantic_pair_256.yaml)
- latent FM tiny/debug: [configs/latent_fm_semantic_256_tiny.yaml](../configs/latent_fm_semantic_256_tiny.yaml)
- latent FM baseline: [configs/latent_fm_semantic_256.yaml](../configs/latent_fm_semantic_256.yaml)
- latent MeanFlow tiny/debug: [configs/latent_meanflow_semantic_256_tiny.yaml](../configs/latent_meanflow_semantic_256_tiny.yaml)
- latent MeanFlow legacy ConvNet baseline: [configs/latent_meanflow_semantic_256.yaml](../configs/latent_meanflow_semantic_256.yaml)
- latent MeanFlow U-Net tiny/debug parallel path: [configs/latent_meanflow_semantic_256_unet_tiny.yaml](../configs/latent_meanflow_semantic_256_unet_tiny.yaml)
- latent MeanFlow default U-Net baseline path: [configs/latent_meanflow_semantic_256_unet.yaml](../configs/latent_meanflow_semantic_256_unet.yaml)
- latent MeanFlow U-Net large parallel path: [configs/latent_meanflow_semantic_256_unet_large.yaml](../configs/latent_meanflow_semantic_256_unet_large.yaml)
- latent MeanFlow U-Net time-scale ablations: [configs/ablations/latent_meanflow_semantic_256_unet_tscale1.yaml](../configs/ablations/latent_meanflow_semantic_256_unet_tscale1.yaml), [configs/ablations/latent_meanflow_semantic_256_unet_tscale100.yaml](../configs/ablations/latent_meanflow_semantic_256_unet_tscale100.yaml), [configs/ablations/latent_meanflow_semantic_256_unet_tscale1000.yaml](../configs/ablations/latent_meanflow_semantic_256_unet_tscale1000.yaml)
- latent MeanFlow U-Net `(r, t)` engineering ablation: [configs/ablations/latent_meanflow_semantic_256_unet_rt_tscale100.yaml](../configs/ablations/latent_meanflow_semantic_256_unet_rt_tscale100.yaml)
- latent AlphaFlow default U-Net route: [configs/latent_alphaflow_semantic_256_unet.yaml](../configs/latent_alphaflow_semantic_256_unet.yaml)

## Artifact Policy

For generative stages, always archive all four paired outputs:

- `image/`
- `mask_raw/`
- `mask_color/`
- `overlay/`

The current tokenizer logger already writes reconstruction grids under `semantic_images/`, but it does not yet export per-sample `mask_raw/` and `overlay/`. For this first round, treat latent FM / MeanFlow sample exports as the required archival paired outputs.

## Stage 1. Semantic Mask Inspect

Purpose: verify that the real dataset gray values still match the current semantic label spec before any training.

Command:

```powershell
D:\Anaconda\envs\lmf\python.exe scripts\inspect_semantic_masks.py `
  --roots data/remote `
  --splits train val test `
  --mask-dir masks `
  --output outputs/inspect/remote_semantic_scan.yaml
```

Expected artifacts:

- `outputs/inspect/remote_semantic_scan.yaml`
- terminal summary of every gray value and pixel ratio

Minimum success criteria:

- every observed gray value is expected by [configs/label_specs/remote_semantic.yaml](../configs/label_specs/remote_semantic.yaml)
- no unexpected sparse gray values appear from interpolation or annotation corruption
- train / val / test all scan successfully

Check first if it fails:

- `data/remote/<split>/masks` path spelling
- mask file extension coverage
- unknown gray values caused by bad export or antialiased masks

## Stage 2. Semantic Tokenizer Tiny Overfit

Purpose: prove the semantic tokenizer can overfit a couple of batches before the real run.

Command:

```powershell
D:\Anaconda\envs\lmf\python.exe scripts\train_semantic_autoencoder.py `
  --config configs/semantic_tokenizer_tiny_256.yaml `
  --gpus 0
```

Expected artifacts:

- a timestamped run under `logs/*_semantic_tokenizer_tiny_256/`
- `checkpoints/last.ckpt`
- `semantic_images/train/inputs_image.png`
- `semantic_images/train/reconstructions_image.png`
- `semantic_images/train/inputs_mask_index.png`
- `semantic_images/train/reconstructions_mask_index.png`

Minimum success criteria:

- `train/total_loss` drops clearly within the run
- `train/mask_ce` drops and reconstructed masks stop looking random
- RGB reconstructions keep coarse structure instead of collapsing to a constant color

Check first if it fails:

- `num_classes` mismatch between dataset and model
- bad `gray_to_class_id` spec or `ignore_index`
- CUDA OOM from another process
- mask channel mismatch caused by using the wrong config or checkpoint

## Stage 3. Semantic Tokenizer Baseline Training

Purpose: produce the tokenizer checkpoint that all latent-prior stages depend on.

Command:

```powershell
D:\Anaconda\envs\lmf\python.exe scripts\train_semantic_autoencoder.py `
  --config configs/autoencoder_semantic_pair_256.yaml `
  --gpus 0 `
  --max-epochs 100 `
  --image-log-frequency 50
```

Expected artifacts:

- `logs/autoencoder/checkpoints/last.ckpt`
- `logs/autoencoder/semantic_images/train/*.png`
- `logs/autoencoder/semantic_images/val/*.png`

Minimum success criteria:

- `val/total_loss` trends down instead of diverging
- `val/rgb_l1` and `val/mask_ce` are both stable
- validation reconstructions keep object boundaries and class regions aligned

Check first if it fails:

- Stage 2 was not stable enough
- label spec drift between inspect output and training config
- data loader reading wrong directories or wrong split
- learning rate too high after auto-scaling

Artifacts needed by later stages:

- tokenizer checkpoint: `logs/autoencoder/checkpoints/last.ckpt`
- tokenizer config: [configs/autoencoder_semantic_pair_256.yaml](../configs/autoencoder_semantic_pair_256.yaml)

## Stage 4. Latent FM Tiny Pilot

Purpose: verify `encode -> latent prior -> sample -> decode` on real semantic latents with the simplest prior objective.

Train command:

```powershell
D:\Anaconda\envs\lmf\python.exe scripts\train_latent_fm.py `
  --config configs/latent_fm_semantic_256_tiny.yaml `
  --tokenizer-config configs/autoencoder_semantic_pair_256.yaml `
  --tokenizer-ckpt logs/autoencoder/checkpoints/last.ckpt `
  --gpus 0
```

Sample command:

```powershell
$fmCkpt = (Get-ChildItem logs -Recurse -Filter last.ckpt | Where-Object { $_.FullName -like '*latent_fm*' } | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
D:\Anaconda\envs\lmf\python.exe scripts\sample_latent_fm.py `
  --config configs/latent_fm_semantic_256_tiny.yaml `
  --ckpt $fmCkpt `
  --outdir outputs/fm_tiny_samples `
  --n-samples 16 `
  --steps 16
```

Expected artifacts:

- a timestamped run under `logs/*_latent_fm/`
- `outputs/fm_tiny_samples/image/`
- `outputs/fm_tiny_samples/mask_raw/`
- `outputs/fm_tiny_samples/mask_color/`
- `outputs/fm_tiny_samples/overlay/`

Minimum success criteria:

- training loss is finite from the first epoch
- decoded samples are not all identical
- `mask_raw/` contains valid class ids instead of a single constant map
- overlays look paired rather than obviously shuffled

Check first if it fails:

- tokenizer checkpoint quality from Stage 3
- wrong tokenizer checkpoint passed to the FM run
- latent prior run accidentally using a stale checkpoint
- class histogram collapse to one dominant class

## Stage 5. Latent MeanFlow Tiny Pilot

Purpose: check that the interval-conditioned prior is stable before the longer baseline run.

Train command:

```powershell
D:\Anaconda\envs\lmf\python.exe scripts\train_latent_meanflow.py `
  --objective meanflow `
  --config configs/latent_meanflow_semantic_256_tiny.yaml `
  --tokenizer-config configs/autoencoder_semantic_pair_256.yaml `
  --tokenizer-ckpt logs/autoencoder/checkpoints/last.ckpt `
  --gpus 0
```

Quick sample command:

```powershell
$meanflowTinyCkpt = D:\Anaconda\envs\lmf\python.exe scripts\find_checkpoint.py --config configs/latent_meanflow_semantic_256_tiny.yaml
D:\Anaconda\envs\lmf\python.exe scripts\sample_latent_flow.py `
  --config configs/latent_meanflow_semantic_256_tiny.yaml `
  --ckpt $meanflowTinyCkpt `
  --outdir outputs/meanflow_tiny_samples_nfe2 `
  --n-samples 16 `
  --nfe 2
```

Expected artifacts:

- a timestamped tiny run under `logs/*_latent_meanflow_semantic_256_tiny/`
- `outputs/meanflow_tiny_samples_nfe2/image/`
- `outputs/meanflow_tiny_samples_nfe2/mask_raw/`
- `outputs/meanflow_tiny_samples_nfe2/mask_color/`
- `outputs/meanflow_tiny_samples_nfe2/overlay/`

Minimum success criteria:

- `train/meanflow_loss` or `train/loss` is finite and not exploding
- decoded `mask_raw/` contains multiple valid classes
- overlays are spatially plausible enough to justify a longer baseline run

Check first if it fails:

- FM stage was unstable, which usually means the tokenizer latent is still weak
- using the paper-like baseline config too early instead of the tiny/debug config
- wrong checkpoint selected for the tiny config

## Stage 5.5. Latent MeanFlow Baseline Training

Purpose: produce the default U-Net MeanFlow checkpoint that Stage 6 will use for the real few-step sampling curve.

Train command:

```powershell
D:\Anaconda\envs\lmf\python.exe scripts\train_latent_meanflow.py `
  --objective meanflow `
  --config configs/latent_meanflow_semantic_256_unet.yaml `
  --tokenizer-config configs/autoencoder_semantic_pair_256.yaml `
  --tokenizer-ckpt logs/autoencoder/checkpoints/last.ckpt `
  --gpus 0
```

Expected artifacts:

- a timestamped baseline run under `logs/*_latent_meanflow_semantic_256_unet/`
- `checkpoints/last.ckpt` inside that baseline run
- `semantic_images/train/*.png`
- `semantic_images/val/*.png`

Minimum success criteria:

- validation loss stays finite through the first real baseline run
- `samples_*` grids look at least as stable as the tiny pilot
- this run produces the baseline checkpoint used by Stage 6

Check first if it fails:

- Stage 5 tiny pilot was not actually stable
- the tokenizer checkpoint from Stage 3 is weak
- the run path contains `latent_meanflow_semantic_256_tiny` or the legacy ConvNet stem instead of the default U-Net config stem

## Stage 6. MeanFlow Sampling Check at NFE = 8 / 4 / 2 / 1

Purpose: establish the first few-step sampling quality curve before spending time on AlphaFlow.

Commands:

```powershell
$meanflowBaselineRun = "logs/<timestamp>_latent_meanflow_semantic_256_unet"
$meanflowBaselineCkpt = D:\Anaconda\envs\lmf\python.exe scripts\find_checkpoint.py --run-dir $meanflowBaselineRun --selection best --monitor val/base_error_mean

D:\Anaconda\envs\lmf\python.exe scripts\sample_latent_flow.py --config configs/latent_meanflow_semantic_256_unet.yaml --ckpt $meanflowBaselineCkpt --outdir outputs/meanflow_nfe8 --n-samples 32 --nfe 8
D:\Anaconda\envs\lmf\python.exe scripts\sample_latent_flow.py --config configs/latent_meanflow_semantic_256_unet.yaml --ckpt $meanflowBaselineCkpt --outdir outputs/meanflow_nfe4 --n-samples 32 --nfe 4
D:\Anaconda\envs\lmf\python.exe scripts\sample_latent_flow.py --config configs/latent_meanflow_semantic_256_unet.yaml --ckpt $meanflowBaselineCkpt --outdir outputs/meanflow_nfe2 --n-samples 32 --nfe 2
D:\Anaconda\envs\lmf\python.exe scripts\sample_latent_flow.py --config configs/latent_meanflow_semantic_256_unet.yaml --ckpt $meanflowBaselineCkpt --outdir outputs/meanflow_nfe1 --n-samples 32 --nfe 1
```

Expected artifacts:

- `outputs/meanflow_nfe8/{image,mask_raw,mask_color,overlay}/`
- `outputs/meanflow_nfe4/{image,mask_raw,mask_color,overlay}/`
- `outputs/meanflow_nfe2/{image,mask_raw,mask_color,overlay}/`
- `outputs/meanflow_nfe1/{image,mask_raw,mask_color,overlay}/`

Minimum success criteria:

- all four NFE settings run end-to-end without NaNs or shape errors
- `NFE=8` and `NFE=4` are visibly better than `NFE=1`
- `NFE=2` is not catastrophically worse than `NFE=4`
- all four runs use the default U-Net MeanFlow checkpoint, not the tiny/debug checkpoint or the legacy ConvNet checkpoint

Check first if it fails:

- baseline checkpoint was not selected explicitly from a pinned run directory
- sampler midpoint assumptions for low NFE
- decoding artifacts that actually come from a weak tokenizer instead of the prior

Legacy ConvNet fallback command:

```powershell
D:\Anaconda\envs\lmf\python.exe scripts\train_latent_meanflow.py `
  --objective meanflow `
  --config configs/latent_meanflow_semantic_256.yaml `
  --tokenizer-config configs/autoencoder_semantic_pair_256.yaml `
  --tokenizer-ckpt logs/autoencoder/checkpoints/last.ckpt `
  --gpus 0
```

## Parallel Backbone Experiment Path: U-Net

Purpose: compare a project-layer U-Net latent field predictor against the
existing ConvNet backbone without changing the tokenizer, data contract, or
objective recipe. This is a side route for backbone ablation, not the default
MeanFlow or AlphaFlow path.

Recommended tiny/debug U-Net MeanFlow pilot:

```powershell
D:\Anaconda\envs\lmf\python.exe scripts\train_latent_meanflow.py `
  --objective meanflow `
  --config configs/latent_meanflow_semantic_256_unet_tiny.yaml `
  --tokenizer-config configs/autoencoder_semantic_pair_256.yaml `
  --tokenizer-ckpt logs/autoencoder/checkpoints/last.ckpt `
  --gpus 0
```

Recommended U-Net MeanFlow baseline:

```powershell
D:\Anaconda\envs\lmf\python.exe scripts\train_latent_meanflow.py `
  --objective meanflow `
  --config configs/latent_meanflow_semantic_256_unet.yaml `
  --tokenizer-config configs/autoencoder_semantic_pair_256.yaml `
  --tokenizer-ckpt logs/autoencoder/checkpoints/last.ckpt `
  --gpus 0
```

Recommended U-Net MeanFlow large run:

```powershell
D:\Anaconda\envs\lmf\python.exe scripts\train_latent_meanflow.py `
  --objective meanflow `
  --config configs/latent_meanflow_semantic_256_unet_large.yaml `
  --tokenizer-config configs/autoencoder_semantic_pair_256.yaml `
  --tokenizer-ckpt logs/autoencoder/checkpoints/last.ckpt `
  --gpus 0
```

Default U-Net AlphaFlow baseline:

```powershell
D:\Anaconda\envs\lmf\python.exe scripts\train_latent_meanflow.py `
  --objective alphaflow `
  --config configs/latent_alphaflow_semantic_256_unet.yaml `
  --tokenizer-config configs/autoencoder_semantic_pair_256.yaml `
  --tokenizer-ckpt logs/autoencoder/checkpoints/last.ckpt `
  --gpus 0
```

Suggested U-Net checkpoint lookup and quick sampling:

```powershell
$meanflowUnetCkpt = D:\Anaconda\envs\lmf\python.exe scripts\find_checkpoint.py --config configs/latent_meanflow_semantic_256_unet.yaml
D:\Anaconda\envs\lmf\python.exe scripts\sample_latent_flow.py `
  --config configs/latent_meanflow_semantic_256_unet.yaml `
  --ckpt $meanflowUnetCkpt `
  --outdir outputs/meanflow_unet_nfe2 `
  --n-samples 16 `
  --nfe 2
```

Interpretation notes:

- the benchmark-backed U-Net route is now the default MeanFlow path
- keep the ConvNet configs as legacy baselines for rollback and controlled comparison
- compare ConvNet and U-Net checkpoints within the same objective family
- do not mix `*_unet*.yaml` configs with ConvNet checkpoints or vice versa

## Engineering Ablation Path: U-Net Time Scales

Purpose: make the raw scalar scale that feeds the U-Net sinusoidal time
embedding explicit and test whether larger numeric input ranges help the U-Net
backbone. This is an engineering ablation only. It does not change the
MeanFlow objective and it should not be presented as a paper-equivalent claim.
The standard U-Net configs remain backward-compatible because omitted scales
still default to `1.0`.

Raw-scale control:

```powershell
D:\Anaconda\envs\lmf\python.exe scripts\train_latent_meanflow.py `
  --objective meanflow `
  --config configs/ablations/latent_meanflow_semantic_256_unet_tscale1.yaml `
  --tokenizer-config configs/autoencoder_semantic_pair_256.yaml `
  --tokenizer-ckpt logs/autoencoder/checkpoints/last.ckpt `
  --gpus 0
```

Medium-scale ablation:

```powershell
D:\Anaconda\envs\lmf\python.exe scripts\train_latent_meanflow.py `
  --objective meanflow `
  --config configs/ablations/latent_meanflow_semantic_256_unet_tscale100.yaml `
  --tokenizer-config configs/autoencoder_semantic_pair_256.yaml `
  --tokenizer-ckpt logs/autoencoder/checkpoints/last.ckpt `
  --gpus 0
```

Large-scale ablation:

```powershell
D:\Anaconda\envs\lmf\python.exe scripts\train_latent_meanflow.py `
  --objective meanflow `
  --config configs/ablations/latent_meanflow_semantic_256_unet_tscale1000.yaml `
  --tokenizer-config configs/autoencoder_semantic_pair_256.yaml `
  --tokenizer-ckpt logs/autoencoder/checkpoints/last.ckpt `
  --gpus 0
```

Optional `(r, t)` conditioning comparison:

```powershell
D:\Anaconda\envs\lmf\python.exe scripts\train_latent_meanflow.py `
  --objective meanflow `
  --config configs/ablations/latent_meanflow_semantic_256_unet_rt_tscale100.yaml `
  --tokenizer-config configs/autoencoder_semantic_pair_256.yaml `
  --tokenizer-ckpt logs/autoencoder/checkpoints/last.ckpt `
  --gpus 0
```

Interpretation notes:

- this ablation only changes the scalar range seen by the sinusoidal embedding
- `tscale1` is the explicit compatibility control and should match the old implicit behavior
- compare these runs against the U-Net baseline, not against a different tokenizer or objective

## Minimum Metrics to Track

Tokenizer:

- `rgb recon loss`: use `train/val rgb_l1`
- `mask CE`: use `train/val mask_ce`
- `recon mIoU`: recommended offline metric on validation reconstructions before comparing latent priors

Generation:

- `image FID` or `image KID`: use once you have enough samples
- `mask class histogram`: compare generated `mask_raw/` against train/val distribution to catch class collapse
- `visual consistency check`: inspect `overlay/` to see whether semantic regions and RGB appearance stay paired

## Recommended First-Day Order

1. Run Stage 1 and confirm the current label spec is still valid.
2. Run Stage 2 until the tokenizer clearly overfits two batches.
3. Launch Stage 3 and keep only the tokenizer checkpoint if reconstructions are visually aligned.
4. Run Stage 4 to confirm the simplest latent prior can sample valid paired outputs.
5. Run Stage 5 to confirm MeanFlow is stable on the same tokenizer.
6. Run Stage 5.5 to produce a baseline MeanFlow checkpoint.
7. Run Stage 6 and compare the first `NFE=8/4/2/1` outputs side by side using that baseline checkpoint.

For the formal ConvNet vs U-Net backbone benchmark protocol after these first
runs, use [docs/unet_backbone_benchmark.md](unet_backbone_benchmark.md). That
runbook fixes the checkpoint selection rule to `val/base_error_mean` and uses a
shared-noise `NFE=8/4/2/1` sweep instead of ad-hoc one-step previews.

## Summary Table

| Stage | Command | Artifact | Pass criteria | Common failure |
| --- | --- | --- | --- | --- |
| Semantic inspect | `inspect_semantic_masks.py --roots data/remote --splits train val test --mask-dir masks --output outputs/inspect/remote_semantic_scan.yaml` | `outputs/inspect/remote_semantic_scan.yaml` | all gray values expected and splits scan cleanly | unknown gray values, wrong mask dir, missing files |
| Tokenizer tiny overfit | `train_semantic_autoencoder.py --config configs/semantic_tokenizer_tiny_256.yaml --gpus 0` | `logs/*_semantic_tokenizer_tiny_256/checkpoints/last.ckpt` and `semantic_images/` | loss drops fast and recon masks stop looking random | label spec mismatch, OOM, wrong class count |
| Tokenizer baseline | `train_semantic_autoencoder.py --config configs/autoencoder_semantic_pair_256.yaml --gpus 0 --max-epochs 100 --image-log-frequency 50` | `logs/autoencoder/checkpoints/last.ckpt` | stable `val/rgb_l1` and `val/mask_ce`, visually aligned reconstructions | weak overfit result, data path drift, LR too high |
| Latent FM tiny pilot | `train_latent_fm.py --config configs/latent_fm_semantic_256_tiny.yaml --tokenizer-ckpt logs/autoencoder/checkpoints/last.ckpt --gpus 0` plus `sample_latent_fm.py` | `logs/*_latent_fm/` and `outputs/fm_tiny_samples/{image,mask_raw,mask_color,overlay}` | finite loss and non-collapsed paired samples | stale tokenizer ckpt, wrong latent_fm ckpt, class collapse |
| Latent MeanFlow tiny pilot | `train_latent_meanflow.py --objective meanflow --config configs/latent_meanflow_semantic_256_tiny.yaml --tokenizer-ckpt logs/autoencoder/checkpoints/last.ckpt --gpus 0` plus `sample_latent_flow.py --config configs/latent_meanflow_semantic_256_tiny.yaml --ckpt <tiny-ckpt> --nfe 2` | `logs/*_latent_meanflow_semantic_256_tiny/` and `outputs/meanflow_tiny_samples_nfe2/{image,mask_raw,mask_color,overlay}` | finite loss and plausible paired overlays | unstable tokenizer, wrong config tier, wrong tiny ckpt |
| Latent MeanFlow baseline | `train_latent_meanflow.py --objective meanflow --config configs/latent_meanflow_semantic_256_unet.yaml --tokenizer-ckpt logs/autoencoder/checkpoints/last.ckpt --gpus 0` | `logs/*_latent_meanflow_semantic_256_unet/checkpoints/last.ckpt` | default U-Net run is stable and produces the checkpoint for Stage 6 | tiny config run reused by mistake, weak tokenizer, stale checkpoint |
| NFE sampling check | `sample_latent_flow.py` at `NFE=8/4/2/1` with `scripts/find_checkpoint.py --run-dir logs/<timestamp>_latent_meanflow_semantic_256_unet --selection best --monitor val/base_error_mean` | `outputs/meanflow_nfe{8,4,2,1}/{image,mask_raw,mask_color,overlay}` | all NFEs run, quality degrades gracefully, and the ckpt is the default U-Net MeanFlow ckpt | baseline/tiny ckpt mismatch, low-NFE sampler issues, tokenizer bottleneck |

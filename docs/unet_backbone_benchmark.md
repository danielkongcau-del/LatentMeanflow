# U-Net Backbone Benchmark

This runbook defines the apples-to-apples benchmark protocol for comparing the
project ConvNet backbone against the project-layer `LatentIntervalUNet` under
the same semantic tokenizer and latent-flow objective.

It is a benchmark protocol, not a training-math change.

## Benchmark Rules

- Keep the tokenizer fixed.
- Keep the dataset fixed.
- Keep the objective family fixed.
- Keep the sampling protocol fixed.
- Keep the checkpoint selection rule fixed.

For the current MeanFlow benchmark, that means:

- same tokenizer config and tokenizer checkpoint
- same semantic dataset rooted at `data/remote`
- same MeanFlow objective recipe
- same seed
- same `NFE=8/4/2/1` sweep
- same checkpoint monitor: `val/base_error_mean`

Do not compare:

- ConvNet `last.ckpt` against U-Net `best` checkpoint
- `val/meanflow_loss` against `val/base_error_mean`
- `NFE=1` only
- one config family against a different tokenizer checkpoint

## Checkpoint Selection Rule

Use the best checkpoint selected by `val/base_error_mean`.

Do not use `last.ckpt` for the main backbone comparison.
Do not rely on "latest checkpoint under logs" logic.
Pin the run directory explicitly, then resolve the best checkpoint inside that
run.

Example:

```powershell
$convBaseRun = "logs/<timestamp>_latent_meanflow_semantic_256"
$convBaseBest = D:\Anaconda\envs\lmf\python.exe scripts\find_checkpoint.py `
  --run-dir $convBaseRun `
  --selection best `
  --monitor val/base_error_mean
```

## Sweep Protocol

Use the benchmark script rather than the ad-hoc sampler defaults:

- fixed checkpoint
- fixed seed
- fixed `NFE=8/4/2/1`
- shared initial latent noise across all NFEs inside one sweep
- required output folders:
  - `image/`
  - `mask_raw/`
  - `mask_color/`
  - `overlay/`

The benchmark script also writes:

- `summary.json`
- `summary.csv`

## Comparison Matrix

Run this matrix in order:

1. MeanFlow ConvNet base vs MeanFlow U-Net base
2. MeanFlow ConvNet large vs MeanFlow U-Net large
3. If MeanFlow U-Net wins clearly, then add AlphaFlow U-Net as the next branch

## Base Pair

Resolve the checkpoints:

```powershell
$convBaseRun = "logs/<timestamp>_latent_meanflow_semantic_256"
$unetBaseRun = "logs/<timestamp>_latent_meanflow_semantic_256_unet"

$convBaseBest = D:\Anaconda\envs\lmf\python.exe scripts\find_checkpoint.py `
  --run-dir $convBaseRun `
  --selection best `
  --monitor val/base_error_mean

$unetBaseBest = D:\Anaconda\envs\lmf\python.exe scripts\find_checkpoint.py `
  --run-dir $unetBaseRun `
  --selection best `
  --monitor val/base_error_mean
```

Run the sweeps:

```powershell
D:\Anaconda\envs\lmf\python.exe scripts\eval_backbone_nfe_sweep.py `
  --config configs/latent_meanflow_semantic_256.yaml `
  --ckpt $convBaseBest `
  --outdir outputs/benchmarks/meanflow_convnet_base `
  --seed 23 `
  --n-samples 32 `
  --batch-size 4 `
  --nfe-values 8 4 2 1

D:\Anaconda\envs\lmf\python.exe scripts\eval_backbone_nfe_sweep.py `
  --config configs/latent_meanflow_semantic_256_unet.yaml `
  --ckpt $unetBaseBest `
  --outdir outputs/benchmarks/meanflow_unet_base `
  --seed 23 `
  --n-samples 32 `
  --batch-size 4 `
  --nfe-values 8 4 2 1
```

## Large Pair

Resolve the checkpoints:

```powershell
$convLargeRun = "logs/<timestamp>_latent_meanflow_semantic_256_large"
$unetLargeRun = "logs/<timestamp>_latent_meanflow_semantic_256_unet_large"

$convLargeBest = D:\Anaconda\envs\lmf\python.exe scripts\find_checkpoint.py `
  --run-dir $convLargeRun `
  --selection best `
  --monitor val/base_error_mean

$unetLargeBest = D:\Anaconda\envs\lmf\python.exe scripts\find_checkpoint.py `
  --run-dir $unetLargeRun `
  --selection best `
  --monitor val/base_error_mean
```

Run the sweeps:

```powershell
D:\Anaconda\envs\lmf\python.exe scripts\eval_backbone_nfe_sweep.py `
  --config configs/latent_meanflow_semantic_256_large.yaml `
  --ckpt $convLargeBest `
  --outdir outputs/benchmarks/meanflow_convnet_large `
  --seed 23 `
  --n-samples 32 `
  --batch-size 4 `
  --nfe-values 8 4 2 1

D:\Anaconda\envs\lmf\python.exe scripts\eval_backbone_nfe_sweep.py `
  --config configs/latent_meanflow_semantic_256_unet_large.yaml `
  --ckpt $unetLargeBest `
  --outdir outputs/benchmarks/meanflow_unet_large `
  --seed 23 `
  --n-samples 32 `
  --batch-size 4 `
  --nfe-values 8 4 2 1
```

## Optional Next Branch: AlphaFlow U-Net

Only add this branch if the MeanFlow U-Net benchmark already shows a clear gain
over the ConvNet baseline.

```powershell
$alphaUnetRun = "logs/<timestamp>_latent_alphaflow_semantic_256_unet"
$alphaUnetBest = D:\Anaconda\envs\lmf\python.exe scripts\find_checkpoint.py `
  --run-dir $alphaUnetRun `
  --selection best `
  --monitor val/base_error_mean

D:\Anaconda\envs\lmf\python.exe scripts\eval_backbone_nfe_sweep.py `
  --config configs/latent_alphaflow_semantic_256_unet.yaml `
  --ckpt $alphaUnetBest `
  --outdir outputs/benchmarks/alphaflow_unet `
  --seed 23 `
  --n-samples 32 `
  --batch-size 4 `
  --nfe-values 8 4 2 1
```

## Pass Criteria

Minimum pass criteria for the backbone swap:

- If U-Net is visibly better than ConvNet at `NFE=4` and `NFE=2`, the backbone
  replacement is worth continuing.
- If `NFE=8` is also not better, do not blame one-step sampling difficulty
  first.
- If `NFE=8` shows no meaningful gain for U-Net, prioritize investigating:
  - tokenizer latent quality
  - objective choice
  - training budget

Results that are strong enough to justify promoting U-Net toward the default
MeanFlow route should satisfy both:

- better visual quality at `NFE=4/2` in the same seed-matched sweep
- no regression at `NFE=8`

## Notes

- `summary.json` and `summary.csv` record the exact config path, checkpoint
  path, monitor name, seed, and NFE values used for each sweep.
- The benchmark script reuses the same initial latent noise bank across all
  NFEs so the sweep compares solver depth rather than different random starts.
- This runbook does not change the MeanFlow objective formula.

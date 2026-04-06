# AGENTS.md

## Repo Layout

- Editable project layer: `configs/`, `docs/`, `latent_meanflow/`, `scripts/`, `README.md`, `AGENTS.md`
- Frozen vendored layer: `third_party/`
- Local-only runtime outputs: `data/`, `logs/`, `outputs/`

## Editable vs Frozen Areas

- Prefer adding parallel project-layer implementations over rewriting the legacy binary pipeline in place.
- Do not modify `third_party/` unless a task explicitly authorizes vendored-runtime changes.
- If a migration task can be solved in project-owned configs, scripts, datasets, or docs, keep the change there.

## Common Commands

Assume the `lmf` environment is active before running commands.

- Legacy autoencoder training:
  - `python scripts/train_autoencoder.py --gpus 0`
- Legacy latent diffusion training:
  - `python scripts/train_ldm.py --gpus 0 --ae-ckpt logs/autoencoder/checkpoints/last.ckpt`
- Legacy paired sampling:
  - `python scripts/sample_mask_image_pairs.py --ckpt logs/ldm/checkpoints/last.ckpt --outdir outputs/remote`

These commands belong to the legacy binary-mask / 4-channel baseline. They are not the future semantic latent-flow path.

## Style and Change Policy

- Keep diffs minimal and scoped to the task.
- Preserve the existing binary baseline unless the task explicitly replaces it.
- Prefer new parallel files over embedding migration-specific branches into legacy files.
- Prefer project-layer wrappers, configs, and dataset code over vendored edits.
- Keep documentation aligned with the checked-in scripts and configs.
- If a task changes repo-tracked files, commit and push when the work is complete unless the user explicitly says not to sync.

## Required Self-Check Before Finish

- `git diff --name-only` must not include `third_party/` unless the task explicitly allows vendored edits.
- `class_label` and `semantic_mask` must be described as different concepts wherever both appear.
- Legacy behavior and planned semantic behavior must be labeled clearly.
- Every documented command must match an existing checked-in path or script.
- Any future interface, file, or workflow that does not exist yet must be marked `planned` or `not implemented yet`.
- If repo-tracked files changed, leave the worktree clean after commit and push.

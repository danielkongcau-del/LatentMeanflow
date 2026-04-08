#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/sample_meanflow.sh [extra args passed to sample_latent_flow.py]

Environment variables:
  PYTHON_BIN            Python executable inside the lmf environment. Default: python
  CONFIG                Sampling config path. Default: configs/latent_meanflow_semantic_256_unet.yaml
  CKPT                  Optional explicit checkpoint path. Default: auto-resolve with scripts/find_checkpoint.py
  OUTDIR                Output directory. Default: outputs/meanflow_samples
  N_SAMPLES             Number of samples. Default: 32
  BATCH_SIZE            Sampling batch size. Default: 4
  NFE                   Number of function evaluations. Default: 4
  SEED                  Random seed. Default: 23
  CLASS_LABEL           Optional image-level condition. Default: unset
  OVERLAY_ALPHA         Overlay alpha for visualizations. Default: 0.4
  TWO_STEP_TIME         Optional midpoint override for 2-step interval sampling. Default: unset
  LOGS_ROOT             Optional logs root for checkpoint lookup. Default: logs
  NO_ALBUMENTATIONS_UPDATE
                        Defaults to 1 on headless Linux servers.

Examples:
  ./scripts/sample_meanflow.sh
  NFE=8 OUTDIR=outputs/meanflow_nfe8 ./scripts/sample_meanflow.sh
  CONFIG=configs/latent_meanflow_semantic_256_tiny.yaml OUTDIR=outputs/meanflow_tiny_samples NFE=2 ./scripts/sample_meanflow.sh
  CKPT=logs/2026-04-07T12-00-00_latent_meanflow_semantic_256_unet/checkpoints/last.ckpt ./scripts/sample_meanflow.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/latent_meanflow_semantic_256_unet.yaml}"
CKPT="${CKPT:-}"
OUTDIR="${OUTDIR:-outputs/meanflow_samples}"
N_SAMPLES="${N_SAMPLES:-32}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NFE="${NFE:-4}"
SEED="${SEED:-23}"
CLASS_LABEL="${CLASS_LABEL:-}"
OVERLAY_ALPHA="${OVERLAY_ALPHA:-0.4}"
TWO_STEP_TIME="${TWO_STEP_TIME:-}"
LOGS_ROOT="${LOGS_ROOT:-logs}"

export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config file not found: ${CONFIG}" >&2
  exit 1
fi

if [[ -z "${CKPT}" ]]; then
  CKPT="$("${PYTHON_BIN}" scripts/find_checkpoint.py --config "${CONFIG}" --logs-root "${LOGS_ROOT}")"
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint file not found: ${CKPT}" >&2
  exit 1
fi

cmd=(
  "${PYTHON_BIN}"
  "scripts/sample_latent_flow.py"
  "--config" "${CONFIG}"
  "--ckpt" "${CKPT}"
  "--outdir" "${OUTDIR}"
  "--n-samples" "${N_SAMPLES}"
  "--batch-size" "${BATCH_SIZE}"
  "--nfe" "${NFE}"
  "--seed" "${SEED}"
  "--overlay-alpha" "${OVERLAY_ALPHA}"
)

if [[ -n "${CLASS_LABEL}" ]]; then
  cmd+=("--class-label" "${CLASS_LABEL}")
fi
if [[ -n "${TWO_STEP_TIME}" ]]; then
  cmd+=("--two-step-time" "${TWO_STEP_TIME}")
fi

cmd+=("$@")

printf 'Using checkpoint: %s\n' "${CKPT}"
printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"

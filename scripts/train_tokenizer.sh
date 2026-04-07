#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/train_tokenizer.sh [extra args passed to train_semantic_autoencoder.py]

Environment variables:
  PYTHON_BIN            Python executable inside the lmf environment. Default: python
  CONFIG                Training config path. Default: configs/autoencoder_semantic_pair_256.yaml
  GPUS                  GPU argument passed through to the Python launcher. Default: 0
  MAX_EPOCHS            Optional max epochs override. Default: unset
  BATCH_SIZE            Optional batch size override. Default: unset
  RESUME                Optional resume checkpoint or logdir path. Default: unset
  IMAGE_LOG_FREQUENCY   Optional image log frequency override. Default: use config value
  ENABLE_IMAGE_LOGGER   Set to 1 to force-enable the image logger. Default: 0
  NO_ALBUMENTATIONS_UPDATE
                        Defaults to 1 on headless Linux servers.

Examples:
  ./scripts/train_tokenizer.sh
  CONFIG=configs/semantic_tokenizer_tiny_256.yaml MAX_EPOCHS=40 ./scripts/train_tokenizer.sh
  IMAGE_LOG_FREQUENCY=50 ./scripts/train_tokenizer.sh
  RESUME=logs/autoencoder/checkpoints/last.ckpt ./scripts/train_tokenizer.sh

Notes:
  Project-layer tokenizer training disables post-fit Trainer.test() by default
  because the semantic autoencoder does not define test_step.
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
CONFIG="${CONFIG:-configs/autoencoder_semantic_pair_256.yaml}"
GPUS="${GPUS:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
RESUME="${RESUME:-}"
IMAGE_LOG_FREQUENCY="${IMAGE_LOG_FREQUENCY:-}"
ENABLE_IMAGE_LOGGER="${ENABLE_IMAGE_LOGGER:-0}"

export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config file not found: ${CONFIG}" >&2
  exit 1
fi

cmd=(
  "${PYTHON_BIN}"
  "scripts/train_semantic_autoencoder.py"
  "--config" "${CONFIG}"
)

if [[ -n "${GPUS}" ]]; then
  cmd+=("--gpus" "${GPUS}")
fi
if [[ -n "${MAX_EPOCHS}" ]]; then
  cmd+=("--max-epochs" "${MAX_EPOCHS}")
fi
if [[ -n "${BATCH_SIZE}" ]]; then
  cmd+=("--batch-size" "${BATCH_SIZE}")
fi
if [[ -n "${RESUME}" ]]; then
  cmd+=("--resume" "${RESUME}")
fi
if [[ "${ENABLE_IMAGE_LOGGER}" == "1" ]]; then
  cmd+=("--enable-image-logger")
fi
if [[ -n "${IMAGE_LOG_FREQUENCY}" ]]; then
  cmd+=("--image-log-frequency" "${IMAGE_LOG_FREQUENCY}")
fi

cmd+=("$@")

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"

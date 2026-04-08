#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/train_meanflow.sh [extra args passed to train_latent_meanflow.py]

Environment variables:
  PYTHON_BIN            Python executable inside the lmf environment. Default: python
  OBJECTIVE             One of fm, meanflow, alphaflow. Default: meanflow
  CONFIG                Training config path. Default: configs/latent_meanflow_semantic_256_unet.yaml
  TOKENIZER_CONFIG      Tokenizer config path. Default: configs/autoencoder_semantic_pair_256.yaml
  TOKENIZER_CKPT        Tokenizer checkpoint path. Default: logs/autoencoder/checkpoints/last.ckpt
  GPUS                  GPU argument passed through to the Python launcher. Default: 0
  MAX_EPOCHS            Optional max epochs override. Default: unset
  BATCH_SIZE            Optional batch size override. Default: unset
  RESUME                Optional resume checkpoint or logdir path. Default: unset
  IMAGE_LOG_FREQUENCY   Optional image log frequency override. Default: unset
  ENABLE_IMAGE_LOGGER   Set to 1 to force-enable the image logger. Default: 0
  NO_ALBUMENTATIONS_UPDATE
                        Defaults to 1 on headless Linux servers.

Examples:
  ./scripts/train_meanflow.sh
  CONFIG=configs/latent_meanflow_semantic_256_tiny.yaml MAX_EPOCHS=3 ./scripts/train_meanflow.sh
  OBJECTIVE=alphaflow CONFIG=configs/latent_alphaflow_semantic_256.yaml ./scripts/train_meanflow.sh
  RESUME=logs/2026-04-07T12-00-00_latent_meanflow_semantic_256_unet/checkpoints/last.ckpt ./scripts/train_meanflow.sh

Notes:
  Project-layer latent-flow training disables post-fit Trainer.test() by default
  because the current trainers do not define test_step.
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
OBJECTIVE="${OBJECTIVE:-meanflow}"
CONFIG="${CONFIG:-configs/latent_meanflow_semantic_256_unet.yaml}"
TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-configs/autoencoder_semantic_pair_256.yaml}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-logs/autoencoder/checkpoints/last.ckpt}"
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
if [[ ! -f "${TOKENIZER_CONFIG}" ]]; then
  echo "Tokenizer config file not found: ${TOKENIZER_CONFIG}" >&2
  exit 1
fi
if [[ -z "${RESUME}" && ! -f "${TOKENIZER_CKPT}" ]]; then
  echo "Tokenizer checkpoint not found for fresh run: ${TOKENIZER_CKPT}" >&2
  echo "Train the semantic tokenizer first or set TOKENIZER_CKPT=/path/to/ckpt" >&2
  exit 1
fi

cmd=(
  "${PYTHON_BIN}"
  "scripts/train_latent_meanflow.py"
  "--objective" "${OBJECTIVE}"
  "--config" "${CONFIG}"
  "--tokenizer-config" "${TOKENIZER_CONFIG}"
  "--tokenizer-ckpt" "${TOKENIZER_CKPT}"
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

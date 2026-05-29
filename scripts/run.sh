#!/usr/bin/env bash
# Usage: bash scripts/run.sh [dataset]
#   Override defaults with environment variables, e.g.
#   MODEL_NAME=Qwen/Qwen2.5-3B-Instruct CUDA_DEVICES=0,1,2,3 bash scripts/run.sh gsm8k
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ----- Environment setup -----
unset CC CXX CUDAHOSTCXX HOST || true
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=CC,CXX
export VLLM_DISABLE_COMPILE_SAMPLER=1

BASE_DIR="${BASE_DIR:-${HOME}/corp_outputs}"
export HF_HOME="${BASE_DIR}/cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_TOKEN="${HF_TOKEN:-}"

export TRITON_CACHE_DIR="${BASE_DIR}/cache/triton"
export TORCH_COMPILE_CACHE_DIR="${BASE_DIR}/cache/torch_compile"
export VLLM_TORCH_COMPILE_CACHE_DIR="${BASE_DIR}/cache/vllm_torch"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_COMPILE_CACHE_DIR" "$VLLM_TORCH_COMPILE_CACHE_DIR"
rm -rf "$TRITON_CACHE_DIR" "$TORCH_COMPILE_CACHE_DIR" "$VLLM_TORCH_COMPILE_CACHE_DIR" || true

ray stop -f >/dev/null 2>&1 || true

# ----- Hyperparameters (override via env) -----
DATASET="${1:-${DATASET:-gsm8k}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${BASE_DIR}/outputs}"

POPULATION_SIZE="${POPULATION_SIZE:-500}"
SIGMA_VALUES="${SIGMA_VALUES:-0.0005,0.001,0.002}"
SOUP_TRAIN_SAMPLES="${SOUP_TRAIN_SAMPLES:-500}"
PROBE_SAMPLES="${PROBE_SAMPLES:-100}"

ALPHA_GRID="${ALPHA_GRID:-0.5,1,2,4,8,16}"
Q_GRID="${Q_GRID:-0.5,0.7,0.9}"
BETA_GRID="${BETA_GRID:-0.5,1,2,5,10,20,50}"

RECENTER_ATTEMPTS="${RECENTER_ATTEMPTS:-3}"
RECENTER_POPULATION_SIZE="${RECENTER_POPULATION_SIZE:-100}"
RECENTER_ISO_LAMBDA="${RECENTER_ISO_LAMBDA:-0.5}"
RECENTER_SIGMA_UP="${RECENTER_SIGMA_UP:-1.25}"
RECENTER_SIGMA_DOWN="${RECENTER_SIGMA_DOWN:-0.5}"
RECENTER_PATIENCE="${RECENTER_PATIENCE:-2}"
MAX_ACCEPTED_RECENTERS="${MAX_ACCEPTED_RECENTERS:-2}"
RECENTER_RANK="${RECENTER_RANK:-8}"
STAGE2_REGRESSION_LAMBDA="${STAGE2_REGRESSION_LAMBDA:-2.0}"

CONSOLIDATION_METHOD="${CONSOLIDATION_METHOD:-corp}"
CONSOLIDATION_BUDGET="${CONSOLIDATION_BUDGET:-fast}"

NUM_ENGINES="${NUM_ENGINES:-4}"
TP="${TP:-1}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
PRECISION="${PRECISION:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
RESUME_DIR="${RESUME_DIR:-}"

# ----- Build command -----
CMD=(
  python3 "${REPO_ROOT}/corp.py"
  --dataset "${DATASET}"
  --model_name "${MODEL_NAME}"
  --experiment_dir "${EXPERIMENT_DIR}"
  --population_size "${POPULATION_SIZE}"
  --sigma_values "${SIGMA_VALUES}"
  --soup_train_samples "${SOUP_TRAIN_SAMPLES}"
  --probe_samples "${PROBE_SAMPLES}"
  --alpha_grid "${ALPHA_GRID}"
  --q_grid "${Q_GRID}"
  --beta_grid "${BETA_GRID}"
  --consolidation_method "${CONSOLIDATION_METHOD}"
  --consolidation_budget "${CONSOLIDATION_BUDGET}"
  --stage2_regression_lambda "${STAGE2_REGRESSION_LAMBDA}"
  --recenter_attempts "${RECENTER_ATTEMPTS}"
  --recenter_population_size "${RECENTER_POPULATION_SIZE}"
  --recenter_iso_lambda "${RECENTER_ISO_LAMBDA}"
  --recenter_sigma_up "${RECENTER_SIGMA_UP}"
  --recenter_sigma_down "${RECENTER_SIGMA_DOWN}"
  --recenter_patience "${RECENTER_PATIENCE}"
  --max_accepted_recenters "${MAX_ACCEPTED_RECENTERS}"
  --recenter_rank "${RECENTER_RANK}"
  --num_engines "${NUM_ENGINES}"
  --tp "${TP}"
  --cuda_devices "${CUDA_DEVICES}"
  --precision "${PRECISION}"
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
)

if [[ -n "${RESUME_DIR}" ]]; then
  CMD+=(--resume_dir "${RESUME_DIR}")
fi
if [[ -n "${TRAIN_DATA_PATH:-}" ]]; then
  CMD+=(--train_data_path "${TRAIN_DATA_PATH}")
fi
if [[ -n "${TEST_DATA_PATH:-}" ]]; then
  CMD+=(--test_data_path "${TEST_DATA_PATH}")
fi

cd "${REPO_ROOT}"
printf 'Running:'
for part in "${CMD[@]}"; do printf ' %q' "${part}"; done
printf '\n'
"${CMD[@]}"

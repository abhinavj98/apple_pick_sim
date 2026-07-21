#!/usr/bin/env bash
# Parallel multi-seed integrity gate for dataset-driven Young's-modulus CMA-ES.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SEEDS_CSV="${SEEDS-0,1,2}"
IFS=',' read -r -a RAW_SEED_LIST <<< "${SEEDS_CSV}"
if [[ ! "${SEEDS_CSV}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "invalid SEEDS='${SEEDS_CSV}' (expected nonempty comma-separated nonnegative decimal integers)" >&2
  exit 2
fi
SEED_LIST=()
declare -A SEEN_SEEDS=()
for RAW_SEED in "${RAW_SEED_LIST[@]}"; do
  SEED="${RAW_SEED#"${RAW_SEED%%[!0]*}"}"
  SEED="${SEED:-0}"
  if [[ -n "${SEEN_SEEDS[${SEED}]+x}" ]]; then
    echo "invalid SEEDS='${SEEDS_CSV}': duplicate seed ${SEED}" >&2
    exit 2
  fi
  SEEN_SEEDS["${SEED}"]=1
  SEED_LIST+=("${SEED}")
done
SEEDS_CSV="$(IFS=,; echo "${SEED_LIST[*]}")"
NUM_STRUCTURES="${NUM_STRUCTURES:-5}"
NUM_DIRECTIONS="${NUM_DIRECTIONS:-5}"
TOPOLOGY_SEED="${TOPOLOGY_SEED:-42}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_SLEEP_S="${RETRY_SLEEP_S:-5}"
DATASET_PREFIX="${DATASET_PREFIX:-tmp/youngs_modulus_cmaes_gate}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_ROOT="${REPORT_ROOT:-tmp/youngs_modulus_cmaes_gate_reports/${TS}}"

COLLECT_TRAJECTORY_ARGS=()
if [[ -n "${HOLD_DURATION_S+x}" ]]; then
  COLLECT_TRAJECTORY_ARGS+=(--hold-duration-s "${HOLD_DURATION_S}")
fi
if [[ -n "${MOVEMENT_PER_STEP_M+x}" ]]; then
  COLLECT_TRAJECTORY_ARGS+=(--movement-per-step-m "${MOVEMENT_PER_STEP_M}")
fi
if [[ -n "${TOTAL_MOVEMENT_M+x}" ]]; then
  COLLECT_TRAJECTORY_ARGS+=(--total-movement-m "${TOTAL_MOVEMENT_M}")
fi
if [[ -n "${MOVE_SPEED_MPS+x}" ]]; then
  COLLECT_TRAJECTORY_ARGS+=(--move-speed-mps "${MOVE_SPEED_MPS}")
fi

SETTLE_ARGS=()
if [[ -n "${SETTLE_SUBSTEPS+x}" ]]; then
  SETTLE_ARGS+=(--settle-substeps "${SETTLE_SUBSTEPS}")
fi
if [[ -n "${SETTLE_QUIET_EVERY+x}" ]]; then
  SETTLE_ARGS+=(--settle-quiet-every "${SETTLE_QUIET_EVERY}")
fi
if [[ -n "${SETTLE_GRAVITY_RAMP+x}" ]]; then
  case "${SETTLE_GRAVITY_RAMP}" in
    1|true|yes|on)
      SETTLE_ARGS+=(--settle-gravity-ramp)
      ;;
    0|false|no|off)
      SETTLE_ARGS+=(--no-settle-gravity-ramp)
      ;;
    *)
      echo "invalid SETTLE_GRAVITY_RAMP='${SETTLE_GRAVITY_RAMP}' (expected 1|true|yes|on or 0|false|no|off)" >&2
      exit 2
      ;;
  esac
fi

# Other CMA search knobs (mean/sigma/popsize/max gens/bounds) live in
# apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py::CMA_SEARCH_PARAMS.
# Gate passes --cma-seed so SEEDS varies both collect and optimizer RNG.
CMAES_ARGS=()
if [[ -n "${RANGES:-}" ]]; then
  CMAES_ARGS+=(--ranges "${RANGES}")
fi

mkdir -p "${REPORT_ROOT}/logs"
rm -f -- "${REPORT_ROOT}/summary.json"

retry_step() {
  local label="$1"
  shift
  local attempt=1
  local ec=0

  while (( attempt <= MAX_RETRIES )); do
    echo "=== ${label} (attempt ${attempt}/${MAX_RETRIES}) ==="
    if "$@"; then
      return 0
    else
      ec=$?
    fi
    if (( attempt >= MAX_RETRIES )); then
      return "${ec}"
    fi
    sleep "${RETRY_SLEEP_S}"
    attempt=$((attempt + 1))
  done
  return "${ec}"
}

run_seed() {
  local SEED="$1"
  local SEED_ROOT="${DATASET_PREFIX}_seed${SEED}"
  local DATASET="${SEED_ROOT}/dataset"
  local CMAES_OUTPUT="${SEED_ROOT}/cmaes"
  local CMAES_JSON="${CMAES_OUTPUT}/cmaes_report.json"
  local SEED_SUMMARY="${REPORT_ROOT}/seed${SEED}_summary.json"
  local seed_ec=0

  mkdir -p "${SEED_ROOT}"
  rm -f -- "${CMAES_JSON}" "${SEED_SUMMARY}"

  run_collect() {
    uv run python apple_pick_gym/batched_examples/example_batched_collect_sysid_data.py \
      --num-structures "${NUM_STRUCTURES}" \
      --num-directions "${NUM_DIRECTIONS}" \
      --output "${DATASET}" \
      --seed "${SEED}" \
      --topology-seed "${TOPOLOGY_SEED}" \
      --overwrite \
      --viewer null \
      "${COLLECT_TRAJECTORY_ARGS[@]}" \
      "${SETTLE_ARGS[@]}"
  }

  run_exclude() {
    uv run python -m apple_pick_gym.batched_envs.exclude_unstable_episodes \
      --dataset "${DATASET}" \
      --inplace
  }

  run_cmaes() {
    uv run python apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py \
      --dataset "${DATASET}" \
      --output "${CMAES_OUTPUT}" \
      --cma-seed "${SEED}" \
      --viewer null \
      --overwrite \
      "${CMAES_ARGS[@]}" \
      "${SETTLE_ARGS[@]}"
  }

  run_seed_report() {
    uv run python -m apple_pick_gym.batched_envs.youngs_modulus_cmaes_gate_report \
      --seed "${SEED}" \
      --cmaes-json "${CMAES_JSON}" \
      --out-summary "${SEED_SUMMARY}" \
      --expected-structures "${NUM_STRUCTURES}"
  }

  if [[ "${SKIP_COLLECT:-0}" != "1" ]]; then
    if ! retry_step "collect seed=${SEED}" run_collect; then
      seed_ec=1
    fi
  fi
  if (( seed_ec == 0 )); then
    if ! retry_step "exclude seed=${SEED}" run_exclude; then
      seed_ec=1
    fi
  fi
  if (( seed_ec == 0 )); then
    if ! retry_step "cmaes seed=${SEED}" run_cmaes; then
      seed_ec=1
    fi
  fi

  run_seed_report || seed_ec=1
  return "${seed_ec}"
}

PASS=1
PIDS=()

echo "======== Young's-modulus CMA-ES gate seeds=${SEEDS_CSV} (parallel) report=${REPORT_ROOT} ========"
for SEED in "${SEED_LIST[@]}"; do
  (
    set +e
    echo "======== seed=${SEED} starting ========"
    run_seed "${SEED}"
    ec=$?
    echo "${ec}" > "${REPORT_ROOT}/logs/seed${SEED}.exit"
    exit "${ec}"
  ) >"${REPORT_ROOT}/logs/seed${SEED}.log" 2>&1 &
  seed_pid=$!
  PIDS+=("${seed_pid}")
  echo "launched seed=${SEED} pid=${seed_pid} log=${REPORT_ROOT}/logs/seed${SEED}.log"
done

for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  seed="${SEED_LIST[$i]}"
  if wait "${pid}"; then
    echo "seed=${seed} finished OK"
  else
    echo "seed=${seed} FAILED (see ${REPORT_ROOT}/logs/seed${seed}.log)" >&2
    PASS=0
  fi
done

if ! uv run python -m apple_pick_gym.batched_envs.youngs_modulus_cmaes_gate_report \
  --finalize \
  --report-dir "${REPORT_ROOT}" \
  --seeds "${SEEDS_CSV}" \
  --out-summary "${REPORT_ROOT}/summary.json"; then
  PASS=0
fi

if [[ "${PASS}" != "1" ]]; then
  echo "GATE FAILED: Young's-modulus CMA-ES integrity (see ${REPORT_ROOT})" >&2
  exit 1
fi
echo "GATE PASSED: Young's-modulus CMA-ES integrity report=${REPORT_ROOT}"

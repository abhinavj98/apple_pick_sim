#!/usr/bin/env bash
# Named Sinkhorn GT-rank gates for sys-ID scoring hardening.
#
# Usage (from repo root):
#   bash scripts/gate_sysid_gt_sinkhorn.sh --gate gate_median_hold
#   bash scripts/gate_sysid_gt_sinkhorn.sh --gate gate_hold_id
#   bash scripts/gate_sysid_gt_sinkhorn.sh --gate gate_pooled_dirs
#
# Env overrides: SEEDS, NUM_STRUCTURES, NUM_DIRECTIONS, TOTAL_MOVEMENT_M, ...
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GATE="gate_pooled_dirs"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gate)
      GATE="$2"
      shift 2
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

case "${GATE}" in
  gate_median_hold|gate_hold_id|gate_pooled_dirs) ;;
  *)
    echo "invalid --gate ${GATE} (expected gate_median_hold|gate_hold_id|gate_pooled_dirs)" >&2
    exit 2
    ;;
esac

SEEDS_CSV="${SEEDS:-0,1,2}"
IFS=',' read -r -a SEED_LIST <<< "${SEEDS_CSV}"
NUM_STRUCTURES="${NUM_STRUCTURES:-5}"
NUM_DIRECTIONS="${NUM_DIRECTIONS:-5}"
TOTAL_MOVEMENT_M="${TOTAL_MOVEMENT_M:-0.08}"
MOVEMENT_PER_STEP_M="${MOVEMENT_PER_STEP_M:-0.01}"
TOPOLOGY_SEED="${TOPOLOGY_SEED:-42}"
SETTLE_SUBSTEPS="${SETTLE_SUBSTEPS:-10000}"
SETTLE_QUIET_EVERY="${SETTLE_QUIET_EVERY:-1000}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_SLEEP_S="${RETRY_SLEEP_S:-5}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_ROOT="${REPORT_ROOT:-tmp/sysid_gate_reports/${GATE}_${TS}}"
mkdir -p "${REPORT_ROOT}"

SCORE_EXTRA=()
case "${GATE}" in
  gate_median_hold)
    SCORE_EXTRA=(--use-median --no-hold-id-onehot --no-pool-directions)
    ;;
  gate_hold_id)
    SCORE_EXTRA=(--use-median --hold-id-onehot --no-pool-directions)
    ;;
  gate_pooled_dirs)
    SCORE_EXTRA=(--use-median --hold-id-onehot --pool-directions)
    ;;
esac

SETTLE_ARGS=(
  --settle-substeps "${SETTLE_SUBSTEPS}"
  --settle-quiet-every "${SETTLE_QUIET_EVERY}"
)

retry_step() {
  local label="$1"
  shift
  local attempt=1
  local ec=0
  while (( attempt <= MAX_RETRIES )); do
    echo "=== ${label} (attempt ${attempt}/${MAX_RETRIES}) ==="
    if "$@"; then
      return 0
    fi
    ec=$?
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
  local DATASET="${DATASET_PREFIX:-tmp/sysid_gate_${GATE}_seed${SEED}}/dataset"
  local PLOT_OUTPUT="${DATASET_PREFIX:-tmp/sysid_gate_${GATE}_seed${SEED}}/plots"
  local SCORE_JSON="${REPORT_ROOT}/seed${SEED}_scores.json"
  local SEED_SUMMARY="${REPORT_ROOT}/seed${SEED}_summary.json"

  mkdir -p "$(dirname "${DATASET}")" "$(dirname "${PLOT_OUTPUT}")"

  run_collect() {
    uv run python apple_pick_gym/batched_examples/example_batched_collect_sysid_data.py \
      --num-structures "${NUM_STRUCTURES}" \
      --num-directions "${NUM_DIRECTIONS}" \
      --output "${DATASET}" \
      --seed "${SEED}" \
      --topology-seed "${TOPOLOGY_SEED}" \
      --movement-per-step-m "${MOVEMENT_PER_STEP_M}" \
      --total-movement-m "${TOTAL_MOVEMENT_M}" \
      --move-speed-mps 0.01 \
      --hold-duration-s 2.0 \
      --overwrite \
      --viewer null \
      "${SETTLE_ARGS[@]}"
  }

  run_exclude() {
    uv run python -m apple_pick_gym.batched_envs.exclude_unstable_episodes \
      --dataset "${DATASET}" \
      --inplace
  }

  run_gt_rank() {
    uv run python apple_pick_gym/batched_examples/example_batched_sysid_mmd_grid.py \
      --viewer null \
      --dataset "${DATASET}" \
      --seed "${SEED}" \
      --score-mse \
      --score-wasserstein \
      --score-json-output "${SCORE_JSON}" \
      --plot-output "${PLOT_OUTPUT}" \
      --grid-values-are-gt-multipliers \
      --primary-bend-stiffness-values 0.1,1,10 \
      --spur-bend-stiffness-values 0.1,1,10 \
      --stem-bend-stiffness-values 0.1,1,10 \
      --secondary-bend-stiffness-values 1 \
      "${SCORE_EXTRA[@]}" \
      "${SETTLE_ARGS[@]}"
  }

  if [[ "${SKIP_COLLECT:-0}" != "1" ]]; then
    retry_step "collect seed=${SEED}" run_collect
  fi
  retry_step "exclude seed=${SEED}" run_exclude
  retry_step "score seed=${SEED}" run_gt_rank

  uv run python -m apple_pick_gym.batched_envs.sysid_gate_report \
    --gate "${GATE}" \
    --seed "${SEED}" \
    --score-json "${SCORE_JSON}" \
    --dataset "${DATASET}" \
    --plot-output "${PLOT_OUTPUT}" \
    --report-dir "${REPORT_ROOT}" \
    --out-summary "${SEED_SUMMARY}"
}

PASS=1
PIDS=()
SEED_ECS=()
mkdir -p "${REPORT_ROOT}/logs"

echo "======== gate=${GATE} seeds=${SEEDS_CSV} (parallel) report=${REPORT_ROOT} ========"
for SEED in "${SEED_LIST[@]}"; do
  (
    set +e
    echo "======== gate=${GATE} seed=${SEED} starting ========"
    run_seed "${SEED}"
    ec=$?
    echo "${ec}" > "${REPORT_ROOT}/logs/seed${SEED}.exit"
    exit "${ec}"
  ) >"${REPORT_ROOT}/logs/seed${SEED}.log" 2>&1 &
  PIDS+=($!)
  echo "launched seed=${SEED} pid=${PIDS[-1]} log=${REPORT_ROOT}/logs/seed${SEED}.log"
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

uv run python -m apple_pick_gym.batched_envs.sysid_gate_report \
  --finalize-gate \
  --gate "${GATE}" \
  --report-dir "${REPORT_ROOT}" \
  --seeds "${SEEDS_CSV}" || PASS=0

if [[ "${PASS}" != "1" ]]; then
  echo "GATE FAILED: ${GATE} (see ${REPORT_ROOT})" >&2
  exit 1
fi
echo "GATE PASSED: ${GATE} report=${REPORT_ROOT}"

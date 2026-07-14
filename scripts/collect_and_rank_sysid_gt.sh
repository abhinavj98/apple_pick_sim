#!/usr/bin/env bash
# Collect a batched sys-ID dataset, then replay a GT-multiplier stiffness grid
# to check that the true params rank #1 for most structures.
#
# Run from repository root:
#   bash scripts/collect_and_rank_sysid_gt.sh
#
# Override via env, e.g.:
#   DATASET=tmp/my_dataset NUM_STRUCTURES=10 bash scripts/collect_and_rank_sysid_gt.sh
#   SKIP_COLLECT=1 bash scripts/collect_and_rank_sysid_gt.sh   # replay only
#   LOG=tmp/my_run.log bash scripts/collect_and_rank_sysid_gt.sh
#   MAX_RETRIES=5 RETRY_SLEEP_S=10 bash scripts/collect_and_rank_sysid_gt.sh
#   SEED=1 bash scripts/collect_and_rank_sysid_gt.sh   # vary collect+replay RNG
#   SEED=2 TOPOLOGY_SEED=7 bash scripts/collect_and_rank_sysid_gt.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# RNG seeds. SEED varies material sampling + replay RNG; TOPOLOGY_SEED fixes segment
# topology (keep constant across seeds to compare the same structures).
SEED="${SEED:-0}"
TOPOLOGY_SEED="${TOPOLOGY_SEED:-42}"

# Shared settle params (must match between collect and replay).
SETTLE_SUBSTEPS="${SETTLE_SUBSTEPS:-10000}"
SETTLE_QUIET_EVERY="${SETTLE_QUIET_EVERY:-1000}"

# Per-seed defaults so parallel seed runs never clobber each other's dataset,
# grid plots, or logs. Override any of these explicitly to pin a path.
DATASET="${DATASET:-tmp/batched_sysid_dataset_settled_seed${SEED}}"
PLOT_OUTPUT="${PLOT_OUTPUT:-tmp/mmd_grid_seed${SEED}}"
NUM_STRUCTURES="${NUM_STRUCTURES:-5}"
NUM_DIRECTIONS="${NUM_DIRECTIONS:-10}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
LOG="${LOG:-tmp/collect_and_rank_sysid_gt_seed${SEED}.log}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_SLEEP_S="${RETRY_SLEEP_S:-5}"

SETTLE_ARGS=(
  --settle-substeps "${SETTLE_SUBSTEPS}"
  --settle-quiet-every "${SETTLE_QUIET_EVERY}"
)

mkdir -p "$(dirname "${LOG}")"

# Retry a labeled step independently. Does not retry the other step.
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
    echo "${label} failed with exit ${ec} (attempt ${attempt}/${MAX_RETRIES})"
    if (( attempt >= MAX_RETRIES )); then
      return "${ec}"
    fi
    echo "retrying ${label} in ${RETRY_SLEEP_S}s..."
    sleep "${RETRY_SLEEP_S}"
    attempt=$((attempt + 1))
  done
  return "${ec}"
}

run_collect() {
  uv run python apple_pick_gym/batched_examples/example_batched_collect_sysid_data.py \
    --num-structures "${NUM_STRUCTURES}" \
    --num-directions "${NUM_DIRECTIONS}" \
    --output "${DATASET}" \
    --seed "${SEED}" \
    --topology-seed "${TOPOLOGY_SEED}" \
    --movement-per-step-m 0.01 \
    --total-movement-m 0.06 \
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
    --replay-only \
    --score-mse \
    --score-wasserstein \
    --plot-output "${PLOT_OUTPUT}" \
    --grid-values-are-gt-multipliers \
    --primary-bend-stiffness-values 0.1,1,10 \
    --spur-bend-stiffness-values 0.1,1,10 \
    --stem-bend-stiffness-values 0.1,1,10 \
    --secondary-bend-stiffness-values 1 \
    "${SETTLE_ARGS[@]}"
}

run() {
  echo "log:      ${LOG}"
  echo "dataset:  ${DATASET}"
  echo "plots:    ${PLOT_OUTPUT}"
  echo "seed:     ${SEED} (topology_seed=${TOPOLOGY_SEED})"
  echo "settle:   substeps=${SETTLE_SUBSTEPS} quiet_every=${SETTLE_QUIET_EVERY}"
  echo "retries:  max=${MAX_RETRIES} sleep_s=${RETRY_SLEEP_S}"

  if [[ "${SKIP_COLLECT}" != "1" ]]; then
    retry_step "collect" run_collect
  fi

  retry_step "exclude unstable episodes" run_exclude

  retry_step "GT-rank replay (MSE + Wasserstein)" run_gt_rank

  echo "done. plots under ${PLOT_OUTPUT}; log at ${LOG}"
}

run 2>&1 | tee "${LOG}"

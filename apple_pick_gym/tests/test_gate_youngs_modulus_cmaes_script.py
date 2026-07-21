"""Static contract tests for the parallel Young's-modulus CMA-ES integrity gate."""

import os
from pathlib import Path
import subprocess

import pytest


SCRIPT_PATH = Path("scripts/gate_youngs_modulus_cmaes.sh")


def _script() -> str:
    return SCRIPT_PATH.read_text(encoding="utf-8")


def test_script_uses_strict_mode_repo_root_and_expected_defaults() -> None:
    script = _script()

    assert "set -euo pipefail" in script
    assert 'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"' in script
    assert 'cd "${ROOT}"' in script
    assert 'SEEDS_CSV="${SEEDS-0,1,2}"' in script
    assert 'NUM_STRUCTURES="${NUM_STRUCTURES:-5}"' in script
    assert 'NUM_DIRECTIONS="${NUM_DIRECTIONS:-5}"' in script
    assert 'TOPOLOGY_SEED="${TOPOLOGY_SEED:-42}"' in script
    assert 'MAX_RETRIES="${MAX_RETRIES:-3}"' in script
    assert 'RETRY_SLEEP_S="${RETRY_SLEEP_S:-5}"' in script
    assert 'REPORT_ROOT="${REPORT_ROOT:-tmp/youngs_modulus_cmaes_gate_reports/${TS}}"' in script
    assert 'DATASET_PREFIX="${DATASET_PREFIX:-tmp/youngs_modulus_cmaes_gate}"' in script
    assert "MAX_GENERATIONS=" not in script
    assert "INITIAL_SIGMA_LOG10=" not in script


def test_script_runs_all_seed_stages_with_seed_specific_artifacts() -> None:
    script = _script()

    assert "example_batched_collect_sysid_data.py" in script
    assert '--num-structures "${NUM_STRUCTURES}"' in script
    assert '--num-directions "${NUM_DIRECTIONS}"' in script
    assert '--output "${DATASET}"' in script
    assert '--seed "${SEED}"' in script
    assert '--topology-seed "${TOPOLOGY_SEED}"' in script
    assert "--overwrite" in script
    assert "--viewer null" in script

    assert "-m apple_pick_gym.batched_envs.exclude_unstable_episodes" in script
    assert '--dataset "${DATASET}"' in script
    assert "--inplace" in script

    assert "example_youngs_modulus_cmaes.py" in script
    assert '--output "${CMAES_OUTPUT}"' in script
    # Gate varies optimizer seed with SEEDS; other search knobs stay in CMA_SEARCH_PARAMS.
    assert '--cma-seed "${SEED}"' in script
    assert "--max-generations" not in script
    assert "--initial-sigma-log10" not in script
    assert "--population-size" not in script
    assert 'local CMAES_JSON="${CMAES_OUTPUT}/cmaes_report.json"' in script

    assert "-m apple_pick_gym.batched_envs.youngs_modulus_cmaes_gate_report" in script
    assert '--cmaes-json "${CMAES_JSON}"' in script
    assert '--expected-structures "${NUM_STRUCTURES}"' in script
    assert '--out-summary "${SEED_SUMMARY}"' in script


def test_script_forwards_optional_ranges_not_grid_ranking_or_search_controls() -> None:
    script = _script()

    assert "POPULATION_SIZE" not in script
    assert 'if [[ -n "${RANGES:-}" ]]; then' in script
    assert 'CMAES_ARGS+=(--ranges "${RANGES}")' in script

    assert "LOG10_E_PRIMARY" not in script
    assert "PASS_THRESHOLD" not in script
    assert "--include-gt-candidate" not in script
    assert "--log10-e-primary" not in script
    assert "example_youngs_modulus_sys_id.py" not in script
    assert "youngs_modulus_gate_report" not in script


def test_script_retries_each_gpu_stage_but_not_reporting() -> None:
    script = _script()

    assert 'retry_step "collect seed=${SEED}" run_collect' in script
    assert 'retry_step "exclude seed=${SEED}" run_exclude' in script
    assert 'retry_step "cmaes seed=${SEED}" run_cmaes' in script
    assert 'run_seed_report || seed_ec=1' in script
    assert 'retry_step "report' not in script
    assert 'sleep "${RETRY_SLEEP_S}"' in script


def test_script_parallelizes_waits_and_aggregates_failures() -> None:
    script = _script()

    assert 'IFS=\',\' read -r -a RAW_SEED_LIST <<< "${SEEDS_CSV}"' in script
    assert 'SEEDS_CSV="$(IFS=,; echo "${SEED_LIST[*]}")"' in script
    assert ') >"${REPORT_ROOT}/logs/seed${SEED}.log" 2>&1 &' in script
    assert 'seed_pid=$!' in script
    assert 'PIDS+=("${seed_pid}")' in script
    assert 'pid=${seed_pid}' in script
    assert "PIDS[-1]" not in script
    assert 'PIDS+=("$!")' not in script
    assert 'echo "${ec}" > "${REPORT_ROOT}/logs/seed${SEED}.exit"' in script
    assert 'for i in "${!PIDS[@]}"; do' in script
    assert 'if wait "${pid}"; then' in script
    assert "PASS=0" in script
    assert 'exit 1' in script


def test_script_always_attempts_final_report_and_preserves_partial_results() -> None:
    script = _script()

    wait_pos = script.index('for i in "${!PIDS[@]}"; do')
    finalize_pos = script.index("--finalize")
    failure_pos = script.index('if [[ "${PASS}" != "1" ]]')
    assert wait_pos < finalize_pos < failure_pos
    assert '--report-dir "${REPORT_ROOT}"' in script
    assert '--seeds "${SEEDS_CSV}"' in script
    assert '--out-summary "${REPORT_ROOT}/summary.json"' in script
    assert 'mkdir -p "${REPORT_ROOT}/logs"' in script
    assert '"${REPORT_ROOT}/logs/seed${SEED}.log"' in script
    assert '"${REPORT_ROOT}/logs/seed${SEED}.exit"' in script


def test_script_removes_stale_final_summary_before_seed_launch() -> None:
    script = _script()

    mkdir_pos = script.index('mkdir -p "${REPORT_ROOT}/logs"')
    invalidate_pos = script.index(
        'rm -f -- "${REPORT_ROOT}/summary.json"', mkdir_pos
    )
    launch_pos = script.index('for SEED in "${SEED_LIST[@]}"; do')

    assert mkdir_pos < invalidate_pos < launch_pos
    assert script.count('rm -f -- "${REPORT_ROOT}/summary.json"') == 1


def test_trajectory_and_settle_overrides_are_forwarded_conditionally() -> None:
    script = _script()

    trajectory_vars = (
        ("HOLD_DURATION_S", "--hold-duration-s"),
        ("MOVEMENT_PER_STEP_M", "--movement-per-step-m"),
        ("TOTAL_MOVEMENT_M", "--total-movement-m"),
        ("MOVE_SPEED_MPS", "--move-speed-mps"),
    )
    for variable, flag in trajectory_vars:
        assert f'if [[ -n "${{{variable}+x}}" ]]; then' in script
        assert f'COLLECT_TRAJECTORY_ARGS+=({flag} "${{{variable}}}")' in script
        assert f'{variable}="${{{variable}:-' not in script

    settle_vars = (
        ("SETTLE_SUBSTEPS", "--settle-substeps"),
        ("SETTLE_QUIET_EVERY", "--settle-quiet-every"),
    )
    for variable, flag in settle_vars:
        assert f'if [[ -n "${{{variable}+x}}" ]]; then' in script
        assert f'SETTLE_ARGS+=({flag} "${{{variable}}}")' in script
        assert f'{variable}="${{{variable}:-' not in script

    assert '"${COLLECT_TRAJECTORY_ARGS[@]}"' in script
    assert script.count('"${SETTLE_ARGS[@]}"') == 2


def test_settle_gravity_ramp_accepts_boolean_values_before_seed_launch() -> None:
    script = _script()

    conditional_pos = script.index('if [[ -n "${SETTLE_GRAVITY_RAMP+x}" ]]; then')
    case_pos = script.index('case "${SETTLE_GRAVITY_RAMP}" in', conditional_pos)
    truthy_pos = script.index("1|true|yes|on)", case_pos)
    falsy_pos = script.index("0|false|no|off)", case_pos)
    error_pos = script.index(
        "invalid SETTLE_GRAVITY_RAMP=", case_pos
    )
    launch_pos = script.index('for SEED in "${SEED_LIST[@]}"; do')

    assert conditional_pos < case_pos < truthy_pos < falsy_pos < error_pos < launch_pos
    assert 'SETTLE_ARGS+=(--settle-gravity-ramp)' in script
    assert 'SETTLE_ARGS+=(--no-settle-gravity-ramp)' in script
    assert 'exit 2' in script[error_pos:launch_pos]
    assert script.count('"${SETTLE_ARGS[@]}"') == 2


def test_stale_cmaes_report_is_removed_before_any_seed_stage_executes() -> None:
    script = _script()

    run_seed_pos = script.index("run_seed() {")
    report_path_pos = script.index(
        'local CMAES_JSON="${CMAES_OUTPUT}/cmaes_report.json"', run_seed_pos
    )
    invalidate_pos = script.index(
        'rm -f -- "${CMAES_JSON}" "${SEED_SUMMARY}"', report_path_pos
    )
    collect_execution_pos = script.index(
        'retry_step "collect seed=${SEED}" run_collect', invalidate_pos
    )
    exclude_execution_pos = script.index(
        'retry_step "exclude seed=${SEED}" run_exclude', invalidate_pos
    )
    cmaes_execution_pos = script.index(
        'retry_step "cmaes seed=${SEED}" run_cmaes', invalidate_pos
    )

    assert (
        report_path_pos
        < invalidate_pos
        < collect_execution_pos
        < exclude_execution_pos
        < cmaes_execution_pos
    )
    assert script.count('rm -f -- "${CMAES_JSON}" "${SEED_SUMMARY}"') == 1
    assert 'rm -rf "${CMAES_OUTPUT}"' not in script


@pytest.mark.parametrize(
    "seeds",
    ["", ",", "1,", ",1", "1,,2", "-1", "+1", "1.0", "1 2", "1;true"],
)
def test_script_rejects_empty_or_unsafe_seed_entries_before_launch(
    tmp_path: Path, seeds: str
) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        cwd=Path.cwd(),
        env={
            **os.environ,
            "SEEDS": seeds,
            "REPORT_ROOT": str(tmp_path / "reports"),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "invalid SEEDS=" in result.stderr
    assert "launched seed=" not in result.stdout
    assert not (tmp_path / "reports").exists()


@pytest.mark.parametrize("seeds", ["1,1", "1,01", "000,0"])
def test_script_rejects_duplicate_normalized_seeds_before_launch(
    tmp_path: Path, seeds: str
) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        cwd=Path.cwd(),
        env={
            **os.environ,
            "SEEDS": seeds,
            "REPORT_ROOT": str(tmp_path / "reports"),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "duplicate seed" in result.stderr
    assert "launched seed=" not in result.stdout
    assert not (tmp_path / "reports").exists()

from pathlib import Path


def test_collect_and_rank_script_passes_seed_knobs() -> None:
    script = Path("scripts/collect_and_rank_sysid_gt.sh").read_text(encoding="utf-8")

    assert 'SEED="${SEED:-0}"' in script
    assert 'TOPOLOGY_SEED="${TOPOLOGY_SEED:-42}"' in script
    assert '--seed "${SEED}"' in script
    assert '--topology-seed "${TOPOLOGY_SEED}"' in script
    assert script.count('--seed "${SEED}"') == 2

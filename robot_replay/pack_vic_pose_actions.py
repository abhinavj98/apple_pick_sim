#!/usr/bin/env python3
"""Rewrite a **legacy 6D-twist** ``batched_sysid_v1`` dataset's ``action`` column into a
19D ``vic_pose`` action: ``[pos(3), quat_wxyz(4), Kp(6), Kd(6)]``.

Datasets exported by ``real_to_batched_sysid`` already carry 19D ``vic_pose_v1``
actions; this packer refuses those (pass ``--force`` to re-pack with different gains).

Position/orientation come from each frame's ``target_pose_4x4`` (falling back
to ``tcp_pose_4x4``, then ``tcp_pos`` + ``tcp_quat`` xyzw when absent);
``Kp``/``Kd`` are constant across the episode,
supplied by the caller (temporary until real parquets ship per-frame gains).

Example::

    uv run python robot_replay/pack_vic_pose_actions.py \\
      --dataset-in /tmp/real_batched_s02_d00 \\
      --dataset-out /tmp/real_batched_s02_d00_vic_pose \\
      --kp 800 800 800 40 40 40 \\
      --kd 80 80 80 4 4 4
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_POS_DIM = 3
_QUAT_DIM = 4
_GAIN_DIM = 6
_ACTION_DIM = _POS_DIM + _QUAT_DIM + 2 * _GAIN_DIM  # 19


def _rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """3x3 rotation -> unit quaternion (w, x, y, z); orthonormalized via SVD."""
    u, _, vt = np.linalg.svd(rot.astype(np.float64))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    trace = np.trace(r)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    else:
        i = int(np.argmax([r[0, 0], r[1, 1], r[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    return (q / n) if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def _pose_action_from_flat16(flat16, kp: tuple[float, ...], kd: tuple[float, ...]) -> list[float]:
    m = np.asarray(flat16, dtype=np.float64).reshape(4, 4)
    pos = m[:3, 3].tolist()
    quat_wxyz = _rotmat_to_quat_wxyz(m[:3, :3]).tolist()
    return [*pos, *quat_wxyz, *kp, *kd]


def _pose_action_from_pos_quat(
    pos,
    quat_xyzw,
    kp: tuple[float, ...],
    kd: tuple[float, ...],
) -> list[float]:
    """Pack pose from ``tcp_pos`` + ``tcp_quat`` (xyzw) into 19D vic_pose action."""
    p = np.asarray(pos, dtype=np.float64).reshape(3).tolist()
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    quat_wxyz = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
    return [*p, *quat_wxyz, *kp, *kd]


def _pose_packed_reason(src_dir: Path) -> str | None:
    """Return why ``src_dir`` already carries 19D ``vic_pose`` actions, else ``None``."""
    manifest_path = src_dir / "manifest.json"
    if manifest_path.exists():
        collection = json.loads(manifest_path.read_text()).get("collection", {})
        if collection.get("action_layout") == "vic_pose_v1":
            return 'manifest collection.action_layout == "vic_pose_v1"'
        if int(collection.get("action_dim") or 0) == _ACTION_DIM:
            return f"manifest collection.action_dim == {_ACTION_DIM}"

    for path in sorted((src_dir / "episodes").glob("*.parquet")):
        table = pq.read_table(path, columns=["action"])
        if table.num_rows == 0:
            continue
        if len(table.column("action")[0].as_py()) == _ACTION_DIM:
            return f"{path.name} action column is already {_ACTION_DIM} wide"
        break
    return None


def pack_vic_pose_actions(
    src_dir: Path,
    dst_dir: Path,
    *,
    kp: tuple[float, ...],
    kd: tuple[float, ...],
    overwrite: bool = False,
    force: bool = False,
) -> dict:
    """Copy ``src_dir`` to ``dst_dir`` with 19D ``vic_pose`` actions; return ``{episodes, frames}``.

    Refuses datasets whose ``action`` is already pose-packed (the exporter writes 19D
    ``vic_pose_v1`` actions directly) unless ``force`` is set: such datasets should be
    replayed as-is rather than rewritten from pose columns.
    """
    if len(kp) != _GAIN_DIM or len(kd) != _GAIN_DIM:
        raise ValueError(f"kp/kd must each have {_GAIN_DIM} entries")

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    if not force:
        reason = _pose_packed_reason(src_dir)
        if reason is not None:
            raise ValueError(
                f"{src_dir} is already pose-packed ({reason}). This packer is for legacy "
                "6D-twist converted datasets only; replay the dataset directly with "
                "--controller-mode vic_pose, or pass --force to re-pack it with new Kp/Kd."
            )
    if dst_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{dst_dir} already exists (pass --overwrite to replace)")
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)

    episode_paths = sorted((dst_dir / "episodes").glob("*.parquet"))
    n_episodes = 0
    n_frames = 0
    for path in episode_paths:
        table = pq.read_table(path)
        if "target_pose_4x4" in table.column_names:
            rows = [
                _pose_action_from_flat16(table.column("target_pose_4x4")[i].as_py(), kp, kd)
                for i in range(table.num_rows)
            ]
        elif "tcp_pose_4x4" in table.column_names:
            rows = [
                _pose_action_from_flat16(table.column("tcp_pose_4x4")[i].as_py(), kp, kd)
                for i in range(table.num_rows)
            ]
        elif "tcp_pos" in table.column_names and "tcp_quat" in table.column_names:
            rows = [
                _pose_action_from_pos_quat(
                    table.column("tcp_pos")[i].as_py(),
                    table.column("tcp_quat")[i].as_py(),
                    kp,
                    kd,
                )
                for i in range(table.num_rows)
            ]
        else:
            raise ValueError(
                f"{path}: missing target_pose_4x4, tcp_pose_4x4, and tcp_pos+tcp_quat"
            )
        action_col = pa.array(rows, type=pa.list_(pa.float32(), _ACTION_DIM))
        idx = table.column_names.index("action")
        new_table = table.set_column(idx, "action", action_col)
        pq.write_table(new_table, path, use_dictionary=False)
        n_episodes += 1
        n_frames += table.num_rows

    manifest_path = dst_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        collection = manifest.setdefault("collection", {})
        collection["action_dim"] = _ACTION_DIM
        collection["action_layout"] = "vic_pose_v1"
        manifest_path.write_text(json.dumps(manifest, indent=2))

    return {"episodes": n_episodes, "frames": n_frames}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-in", type=Path, required=True)
    parser.add_argument("--dataset-out", type=Path, required=True)
    parser.add_argument("--kp", type=float, nargs=6, required=True, metavar="Kp")
    parser.add_argument("--kd", type=float, nargs=6, required=True, metavar="Kd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-pack even when --dataset-in already carries 19D vic_pose actions.",
    )
    args = parser.parse_args(argv)

    try:
        stats = pack_vic_pose_actions(
            args.dataset_in,
            args.dataset_out,
            kp=tuple(args.kp),
            kd=tuple(args.kd),
            overwrite=bool(args.overwrite),
            force=bool(args.force),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

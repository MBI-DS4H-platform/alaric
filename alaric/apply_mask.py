#!/usr/bin/env python3
"""Apply one side of a mask-common-conformer result to its source pool."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

# Several Alaric runtime modules retain direct-script imports (``from npy_io``). Make
# those sibling modules visible when this entry point is run as ``python -m alaric.apply_mask``.
_ALARIC_SOURCE = str(Path(__file__).resolve().parent)
if _ALARIC_SOURCE not in sys.path:
    sys.path.insert(0, _ALARIC_SOURCE)

try:
    from .mask_common_conformer import pose_order_identity
    from .npy_io import find_npy, load_npy
    from .organize import organize_pose_dir
    from .pose_filter import BoolMaskSelector, filter_pose_dir
except ImportError:  # direct script execution with ALARIC_DIR on PYTHONPATH
    from mask_common_conformer import pose_order_identity
    from npy_io import find_npy, load_npy
    from organize import organize_pose_dir
    from pose_filter import BoolMaskSelector, filter_pose_dir


def _read_pair_metadata(mask_pair_dir: Path, side: int) -> dict:
    path = mask_pair_dir / "mask-pair.json"
    if not path.is_file():
        raise FileNotFoundError(f"mask-pair metadata missing: {path}")
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read mask-pair metadata: {exc}") from exc
    if not isinstance(data, dict) or type(data.get("schema")) is not int or data["schema"] != 1:
        raise ValueError("mask-pair metadata must declare schema 1")
    sources = data.get("sources")
    if not isinstance(sources, dict) or str(side) not in sources:
        raise ValueError(f"mask-pair metadata has no source identity for side {side}")
    source = sources[str(side)]
    if not isinstance(source, dict):
        raise ValueError(f"mask-pair side {side} source identity is malformed")
    count = source.get("poses")
    digest = source.get("order_sha256")
    if type(count) is not int or count < 0:
        raise ValueError(f"mask-pair side {side} pose count is malformed")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"mask-pair side {side} order digest is malformed")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(f"mask-pair side {side} order digest is malformed") from exc
    return source


def run_apply_mask(
    pose_dir: str | Path,
    mask_pair_dir: str | Path,
    output_dir: str | Path,
    side: int,
    *,
    force: bool = False,
    compress: bool = True,
    nprocs: int | None = None,
    chunk_poses: int = 10_000,
) -> dict[str, int]:
    """Filter a pool with its corresponding bool mask and preserve source provenance."""
    if type(side) is not int or side not in {1, 2}:
        raise ValueError("mask side must be the integer 1 or 2")
    pose_dir = Path(pose_dir)
    mask_pair_dir = Path(mask_pair_dir)
    output_dir = Path(output_dir)
    source_identity = _read_pair_metadata(mask_pair_dir, side)
    pose_count, order_digest = pose_order_identity(pose_dir)
    if pose_count != source_identity["poses"] or order_digest != source_identity["order_sha256"]:
        raise ValueError(f"input pool does not match mask-common-conformer side {side} ordering")

    mask_path = find_npy(mask_pair_dir / f"mask{side}.npy")
    if mask_path is None:
        raise FileNotFoundError(f"no mask{side}.npy found in {mask_pair_dir}")
    mask = np.asarray(load_npy(mask_path))
    if mask.dtype != np.dtype(np.bool_) or mask.ndim != 1:
        raise ValueError(f"{mask_path}: expected a one-dimensional boolean mask")
    if len(mask) != pose_count:
        raise ValueError(
            f"{mask_path}: mask covers {len(mask)} poses but input holds {pose_count}"
        )

    if output_dir.exists():
        if any(output_dir.iterdir()):
            if not force:
                raise ValueError(f"output directory is not empty: {output_dir}")
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = filter_pose_dir(
        pose_dir,
        output_dir,
        BoolMaskSelector(mask),
        provenance_path=output_dir / "provenance.npy",
        compress=compress,
        nprocs=nprocs,
        chunk_poses=chunk_poses,
    )
    organize_kwargs = {"compress": compress}
    if nprocs is not None:
        organize_kwargs["nprocs"] = nprocs
    organize_pose_dir(output_dir, **organize_kwargs)
    return {"input_poses": stats.total_poses, "kept_poses": stats.kept_poses}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Organized source pose pool")
    parser.add_argument("mask_pair_dir", help="mask-common-conformer result directory")
    parser.add_argument("output", help="Output directory for selected poses")
    parser.add_argument("--mask", required=True, type=int, choices=(1, 2), help="matching source side")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--nprocs", type=int)
    parser.add_argument("--chunk-poses", type=int, default=10_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_apply_mask(
        args.input,
        args.mask_pair_dir,
        args.output,
        args.mask,
        force=args.force,
        compress=args.compress,
        nprocs=args.nprocs,
        chunk_poses=args.chunk_poses,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

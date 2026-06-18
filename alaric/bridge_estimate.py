from __future__ import annotations

import argparse
from pathlib import Path
import sys

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from bridge import BridgeGrowConfig, _estimate_bridge_growth_once


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bridge-estimate")
    parser.add_argument("--lower-poses", required=True, type=Path)
    parser.add_argument("--upper-poses", required=True, type=Path)
    parser.add_argument("--lower-sequence", required=True)
    parser.add_argument("--middle-sequence", required=True)
    parser.add_argument("--upper-sequence", required=True)
    parser.add_argument("--lower-crmsd", required=True, type=float)
    parser.add_argument("--lower-ov-rmsd", required=True, type=float)
    parser.add_argument("--upper-crmsd", required=True, type=float)
    parser.add_argument("--upper-ov-rmsd", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--estimator-seed", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--pdb-exclude", nargs="*", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lower = BridgeGrowConfig(
        source_poses=args.lower_poses,
        source_sequence=args.lower_sequence,
        target_sequence=args.middle_sequence,
        direction="forward",
        crmsd=args.lower_crmsd,
        ov_rmsd=args.lower_ov_rmsd,
        pdb_exclude=tuple(args.pdb_exclude),
    )
    upper = BridgeGrowConfig(
        source_poses=args.upper_poses,
        source_sequence=args.upper_sequence,
        target_sequence=args.middle_sequence,
        direction="backward",
        crmsd=args.upper_crmsd,
        ov_rmsd=args.upper_ov_rmsd,
        pdb_exclude=tuple(args.pdb_exclude),
    )
    _estimate_bridge_growth_once(
        lower_config=lower,
        upper_config=upper,
        estimate_dir=args.output,
        estimator_seed=args.estimator_seed,
        sample_size=args.sample_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

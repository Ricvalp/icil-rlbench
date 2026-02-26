#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from icil.datasets.in_context_imitation_learning.build_dense_cache_per_variation import (
    build_cache_all_tasks,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build dense per-variation H5 cache for all discovered RLBench tasks."
    )
    parser.add_argument(
        "--root-raw",
        type=Path,
        default=None,
        help=(
            "Raw RLBench root containing task folders. "
            "Default resolution: --root-raw > ICIL_RAW_ROOT > parent(ICIL_CACHE_ROOT) > output_data_playground_v3."
        ),
    )
    parser.add_argument(
        "--root-cache",
        type=Path,
        default=None,
        help=(
            "Dense cache root. "
            "Default resolution: --root-cache > ICIL_CACHE_ROOT > <root-raw>/.rlbench_cache_dense."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional task names. If omitted, cache all tasks discovered under --root-raw.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--variations",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit variation ids to cache.",
    )
    group.add_argument(
        "--variation-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Optional inclusive variation range to cache.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=4096,
        help="Number of points sampled per frame in cache.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for variation-level parallel caching.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm episode progress bars.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    env_cache_root = os.environ.get("ICIL_CACHE_ROOT", "").strip()
    env_raw_root = os.environ.get("ICIL_RAW_ROOT", "").strip()

    if args.root_raw is not None:
        root_raw = args.root_raw
    elif env_raw_root:
        root_raw = Path(env_raw_root)
    elif env_cache_root:
        root_raw = Path(env_cache_root).resolve().parent
    else:
        root_raw = Path("output_data_playground_v3")

    if args.root_cache is not None:
        root_cache = args.root_cache
    elif env_cache_root:
        root_cache = Path(env_cache_root)
    else:
        root_cache = root_raw / ".rlbench_cache_dense"

    if args.variation_range is not None:
        start, end = args.variation_range
        variations = (int(start), int(end))
    elif args.variations is not None and len(args.variations) > 0:
        variations = [int(v) for v in args.variations]
    else:
        variations = None

    build_cache_all_tasks(
        root_raw=root_raw,
        root_cache=root_cache,
        tasks=args.tasks,
        variations=variations,
        N=int(args.num_points),
        show_progress=not args.no_progress,
        num_workers=int(args.num_workers),
    )


if __name__ == "__main__":
    main()

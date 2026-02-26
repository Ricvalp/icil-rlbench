from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import h5py
from icil.datasets.common import MASK_LABEL_MAP_FILENAME
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm optional
    tqdm = None

try:
    from .cache_variation_h5 import CacheSpec, cache_episode_into_variation_file
except ImportError:  # pragma: no cover - allow direct script execution
    from cache_variation_h5 import CacheSpec, cache_episode_into_variation_file


def discover_tasks(root_raw: Path) -> List[str]:
    if not root_raw.is_dir():
        return []
    tasks = []
    for p in sorted(root_raw.iterdir()):
        if p.is_dir() and any(p.glob("variation*")):
            tasks.append(p.name)
    return tasks


def _variation_id_from_dir(var_dir: Path) -> int:
    return int(var_dir.name.replace("variation", ""))


def _selected_variation_dirs(task_root: Path, variations) -> List[Path]:
    if variations is None:
        var_dirs = [p for p in task_root.glob("variation*") if p.is_dir()]
    elif isinstance(variations, tuple):
        start, end = variations
        if start > end:
            raise ValueError("variation range start must be <= end")
        var_dirs = [task_root / f"variation{v}" for v in range(start, end + 1)]
    else:
        var_dirs = [task_root / f"variation{int(v)}" for v in variations]

    # Deduplicate by variation id and keep only existing directories.
    unique: dict[int, Path] = {}
    for p in var_dirs:
        if not p.is_dir():
            continue
        unique[_variation_id_from_dir(p)] = p
    return [unique[v] for v in sorted(unique)]


def _split_by_mask_map(var_dirs: List[Path]) -> Tuple[List[Path], List[Path]]:
    valid: List[Path] = []
    missing: List[Path] = []
    for p in var_dirs:
        if (p / MASK_LABEL_MAP_FILENAME).is_file():
            valid.append(p)
        else:
            missing.append(p)
    return valid, missing


def build_cache(
    root_raw: Path,
    root_cache: Path,
    task: str,
    variations=None,
    N: int = 4096,
    show_progress: bool = True,
) -> int:
    task_root = root_raw / task
    selected_var_dirs = _selected_variation_dirs(task_root, variations)
    var_dirs, _missing_map = _split_by_mask_map(selected_var_dirs)

    spec = CacheSpec(N=N, store_rgb=True, store_mask_id=True)
    proprio_keys = ("gripper_pose", "gripper_open")
    action_keys = ("gripper_pose", "gripper_open")

    cached_episodes = 0
    for var_dir in var_dirs:
        if not var_dir.is_dir():
            continue
        variation_id = int(var_dir.name.replace("variation", ""))
        out_h5 = root_cache / task / f"variation{variation_id}.h5"

        episodes_dir = var_dir / "episodes"
        if not episodes_dir.is_dir():
            continue
        ep_dirs = sorted(episodes_dir.glob("episode*"))
        iterator = ep_dirs
        if show_progress and tqdm is not None:
            iterator = tqdm(
                ep_dirs,
                desc=f"[cache-dense] {task} v{variation_id}",
                unit="ep",
                leave=False,
            )
        for ep_dir in iterator:
            # low_dim_obs.pkl is written at the end of episode export; if missing,
            # treat this as in-progress and skip for now.
            if not (ep_dir / "low_dim_obs.pkl").is_file():
                continue
            episode_id = int(ep_dir.name.replace("episode", ""))
            # skip if already cached
            already_cached = False
            if out_h5.exists():
                try:
                    with h5py.File(out_h5, "r") as f:
                        if "episodes" in f and str(episode_id) in f["episodes"]:
                            already_cached = True
                except (BlockingIOError, OSError):
                    # Another process may hold an HDF5 lock; skip and retry in a
                    # future pass.
                    continue
            if already_cached:
                continue

            try:
                cache_episode_into_variation_file(
                    episode_dir=ep_dir,
                    variation_dir=var_dir,
                    out_variation_h5=out_h5,
                    task_name=task,
                    variation_id=variation_id,
                    episode_id=episode_id,
                    proprio_keys=proprio_keys,
                    action_keys=action_keys,
                    spec=spec,
                    seed=episode_id,
                )
                cached_episodes += 1
            except (FileNotFoundError, EOFError, OSError, RuntimeError):
                # Episode may still be in-flight while raw generation is running.
                continue
    return cached_episodes


def _count_variation_remaining_episodes(
    root_raw: Path,
    root_cache: Path,
    task: str,
    variation_id: int,
) -> Tuple[int, int]:
    """
    Returns (eligible_complete_episodes, remaining_to_cache) for one variation.
    Snapshot-only: if raw generation is still running this can change over time.
    """
    var_dir = root_raw / task / f"variation{int(variation_id)}"
    episodes_dir = var_dir / "episodes"
    if not episodes_dir.is_dir():
        return 0, 0

    eligible_episode_ids: List[int] = []
    for ep_dir in episodes_dir.glob("episode*"):
        if not ep_dir.is_dir():
            continue
        if not (ep_dir / "low_dim_obs.pkl").is_file():
            continue
        try:
            episode_id = int(ep_dir.name.replace("episode", ""))
        except ValueError:
            continue
        eligible_episode_ids.append(episode_id)

    if not eligible_episode_ids:
        return 0, 0

    out_h5 = root_cache / task / f"variation{int(variation_id)}.h5"
    cached_ids: set[int] = set()
    if out_h5.exists():
        try:
            with h5py.File(out_h5, "r") as f:
                if "episodes" in f:
                    for key in f["episodes"].keys():
                        try:
                            cached_ids.add(int(key))
                        except ValueError:
                            continue
        except (BlockingIOError, OSError):
            # If the file is temporarily locked, treat as uncached for snapshot.
            cached_ids = set()

    remaining = sum(1 for eid in eligible_episode_ids if eid not in cached_ids)
    return len(eligible_episode_ids), remaining


def _cache_variation_job(
    args: Tuple[Path, Path, str, int, int, int],
) -> Tuple[str, int, int, Optional[str]]:
    root_raw, root_cache, task, variation_id, n_points, _snapshot_remaining = args
    try:
        cached_episodes = build_cache(
            root_raw=root_raw,
            root_cache=root_cache,
            task=task,
            variations=[int(variation_id)],
            N=int(n_points),
            show_progress=False,
        )
        return task, int(variation_id), int(cached_episodes), None
    except Exception as exc:  # pragma: no cover - defensive for worker isolation
        return task, int(variation_id), 0, f"{type(exc).__name__}: {exc}"


def build_cache_all_tasks(
    root_raw: Path,
    root_cache: Path,
    tasks: Optional[Iterable[str]] = None,
    variations=None,
    N: int = 4096,
    show_progress: bool = True,
    num_workers: int = 1,
) -> List[str]:
    task_list = list(tasks) if tasks else discover_tasks(root_raw)
    if not task_list:
        raise RuntimeError(f"No tasks found in raw dataset root: {root_raw}")

    print(f"[cache-dense] raw_root={root_raw}")
    print(f"[cache-dense] cache_root={root_cache}")
    print(f"[cache-dense] tasks={len(task_list)}")

    jobs: List[Tuple[Path, Path, str, int, int, int]] = []
    skipped_missing_map = 0
    skipped_preview: List[str] = []
    snapshot_eligible_episodes = 0
    snapshot_remaining_episodes = 0
    for task in task_list:
        task_root = root_raw / task
        selected_var_dirs = _selected_variation_dirs(task_root, variations)
        valid_var_dirs, missing_map_dirs = _split_by_mask_map(selected_var_dirs)
        skipped_missing_map += len(missing_map_dirs)
        for p in missing_map_dirs[:5]:
            if len(skipped_preview) < 12:
                skipped_preview.append(f"{task}/{p.name}")
        for p in valid_var_dirs:
            vid = _variation_id_from_dir(p)
            eligible, remaining = _count_variation_remaining_episodes(
                root_raw=root_raw,
                root_cache=root_cache,
                task=task,
                variation_id=int(vid),
            )
            snapshot_eligible_episodes += int(eligible)
            snapshot_remaining_episodes += int(remaining)
            jobs.append((root_raw, root_cache, task, int(vid), int(N), int(remaining)))

    print(f"[cache-dense] variation-jobs={len(jobs)}")
    print(f"[cache-dense] pre-scan eligible-complete-episodes={snapshot_eligible_episodes}")
    print(f"[cache-dense] pre-scan remaining-to-cache={snapshot_remaining_episodes}")
    if skipped_missing_map > 0:
        extra = f" (e.g. {', '.join(skipped_preview)})" if skipped_preview else ""
        print(
            f"[cache-dense] skipped {skipped_missing_map} variation(s) missing "
            f"{MASK_LABEL_MAP_FILENAME}{extra}"
        )
    if not jobs:
        print("[cache-dense] no matching task/variation jobs found")
        return task_list

    failures: List[Tuple[str, int, str]] = []
    cached_episode_total = 0
    workers = max(1, int(num_workers))
    pbar_episodes = None
    if show_progress and tqdm is not None and snapshot_remaining_episodes > 0:
        pbar_episodes = tqdm(
            total=int(snapshot_remaining_episodes),
            desc="[cache-dense] episodes",
            unit="ep",
        )
    if workers == 1:
        for idx, (root_raw_j, root_cache_j, task, vid, n_points, snapshot_remaining) in enumerate(jobs, start=1):
            print(f"[cache-dense] ({idx}/{len(jobs)}) {task} variation{vid}")
            task_name, variation_id, cached_eps, err = _cache_variation_job(
                (root_raw_j, root_cache_j, task, vid, n_points, snapshot_remaining)
            )
            cached_episode_total += int(cached_eps)
            if pbar_episodes is not None and cached_eps > 0:
                pbar_episodes.update(int(cached_eps))
            if err is not None:
                failures.append((task_name, variation_id, err))
    else:
        print(f"[cache-dense] multiprocessing workers={workers}")
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
            future_to_job = {ex.submit(_cache_variation_job, job): job for job in jobs}
            for fut in as_completed(future_to_job):
                task_name, variation_id, cached_eps, err = fut.result()
                cached_episode_total += int(cached_eps)
                if err is not None:
                    failures.append((task_name, variation_id, err))
                if pbar_episodes is not None and cached_eps > 0:
                    pbar_episodes.update(int(cached_eps))
                elif err is not None:
                    print(f"[cache-dense] failed {task_name} variation{variation_id}: {err}")
    if pbar_episodes is not None:
        pbar_episodes.close()

    if failures:
        preview = "\n".join(
            f"  - {task} variation{vid}: {err}" for task, vid, err in failures[:20]
        )
        raise RuntimeError(
            f"[cache-dense] {len(failures)} variation job(s) failed:\n{preview}"
        )

    print(f"[cache-dense] cached-episodes-this-run={cached_episode_total}")
    print("[cache-dense] done")
    return task_list

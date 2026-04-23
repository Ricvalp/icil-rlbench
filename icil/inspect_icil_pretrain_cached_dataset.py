#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from icil.action_representation import decode_action_chunk
from icil.datasets.in_context_imitation_learning.icil_datasets import (
    ICILConfig,
    ICILPretrainBatchIterable,
)
from icil.datasets.in_context_imitation_learning.variation_store import (
    VariationStore,
    build_variation_keys,
)

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "plotly is required for HTML point-cloud visualization. Install with: pip install plotly"
    ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect ICILPretrainBatchIterable batches and save visualizations."
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("output_data_playground_v3/.rlbench_cache_dense"),
        help="Root cache directory with <task>/variation*.h5 files.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional task subset. If omitted, all cached tasks are used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_data_playground_v3/.inspection/icil_pretrain"),
        help="Directory to store plots and summaries.",
    )

    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--T-obs", dest="T_obs", type=int, default=2)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--action-representation",
        type=str,
        default="absolute",
        choices=("absolute", "delta_xyz"),
        help="Interpret target_action/cond_traj as absolute positions or delta xyz actions.",
    )

    parser.add_argument("--batch-size", type=int, default=2, help="Pretrain batch size B.")
    parser.add_argument("--num-batches", type=int, default=4, help="Number of batches to inspect.")
    parser.add_argument(
        "--samples-per-batch",
        type=int,
        default=2,
        help="How many samples to visualize from each batch (<= batch-size).",
    )
    parser.add_argument(
        "--max-points-per-trace",
        type=int,
        default=1200,
        help="Downsample points per frame trace for HTML rendering speed.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-tries-per-item", type=int, default=100)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--keep-open-per-worker", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _unwrap_single(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    return batch_list[0]


def _discover_cached_tasks(cache_root: Path) -> List[str]:
    if not cache_root.is_dir():
        return []
    out: List[str] = []
    for p in sorted(cache_root.iterdir()):
        if p.is_dir() and any(p.glob("variation*.h5")):
            out.append(p.name)
    return out


def _build_store(
    cache_root: Path,
    tasks: Sequence[str] | None,
    keep_open_per_worker: bool,
) -> Tuple[VariationStore, List[str]]:
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Cache root not found: {cache_root}")

    selected = list(tasks) if tasks else _discover_cached_tasks(cache_root)
    if not selected:
        raise RuntimeError(f"No cached tasks found in {cache_root}")

    keys = []
    missing = []
    for task in selected:
        task_keys = build_variation_keys(cache_root, task)
        if not task_keys:
            missing.append(task)
            continue
        keys.extend(task_keys)
    if missing:
        raise RuntimeError(f"Missing variation*.h5 for tasks: {', '.join(sorted(missing))}")
    if not keys:
        raise RuntimeError(f"No variation*.h5 files found under {cache_root}")

    return VariationStore(keys, keep_open_per_worker=keep_open_per_worker), selected


def _infer_dims(store: VariationStore) -> Tuple[int, int, int]:
    for vidx in range(len(store)):
        eids = store.list_episode_ids(vidx)
        if eids.shape[0] == 0:
            continue
        eid = int(eids[0])
        sample = store.load_episode_slices(
            vidx,
            eid,
            np.asarray([0], dtype=np.int64),
            load_rgb=False,
            load_mask_id=False,
        )
        N = int(sample["xyz"].shape[1])
        S = int(sample["state"].shape[1])
        A = int(sample["action"].shape[1])
        return N, S, A
    raise RuntimeError("Could not infer dimensions from cache (no non-empty episodes found).")


def _downsample_indices(n: int, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=max_n, replace=False)).astype(np.int64)


def _color_for_idx(idx: int, total: int) -> str:
    if total <= 1:
        h = 0.0
    else:
        h = float(idx) / float(total)
    # HSV -> RGB (s=0.7, v=0.9)
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = 0.27
    q = 0.9 * (1.0 - 0.7 * f)
    t = 0.9 * (1.0 - 0.7 * (1.0 - f))
    i = i % 6
    if i == 0:
        r, g, b = 0.9, t, p
    elif i == 1:
        r, g, b = q, 0.9, p
    elif i == 2:
        r, g, b = p, 0.9, t
    elif i == 3:
        r, g, b = p, q, 0.9
    elif i == 4:
        r, g, b = t, p, 0.9
    else:
        r, g, b = 0.9, p, q
    return f"rgb({int(255*r)},{int(255*g)},{int(255*b)})"


def _save_cloud_sequence_html(
    out_path: Path,
    xyz_seq: torch.Tensor,      # [T,N,3]
    valid_seq: torch.Tensor,    # [T,N]
    title: str,
    rng: np.random.Generator,
    max_points_per_trace: int,
    action_xyz: np.ndarray | None = None,  # [H,3]
) -> None:
    xyz_np = xyz_seq.detach().cpu().numpy()
    valid_np = valid_seq.detach().cpu().numpy().astype(bool)
    T = int(xyz_np.shape[0])

    fig = go.Figure()
    for t in range(T):
        pts = xyz_np[t]
        mask = valid_np[t]
        if mask.ndim != 1:
            mask = mask.reshape(-1)
        pts = pts[mask]
        if pts.shape[0] == 0:
            continue
        idx = _downsample_indices(pts.shape[0], max_points_per_trace, rng)
        p = pts[idx]
        fig.add_trace(
            go.Scatter3d(
                x=p[:, 0],
                y=p[:, 1],
                z=p[:, 2],
                mode="markers",
                marker=dict(size=1.8, color=_color_for_idx(t, T), opacity=0.8),
                name=f"t{t}",
            )
        )

    if action_xyz is not None and action_xyz.shape[0] > 0:
        fig.add_trace(
            go.Scatter3d(
                x=action_xyz[:, 0],
                y=action_xyz[:, 1],
                z=action_xyz[:, 2],
                mode="lines+markers",
                line=dict(color="black", width=6),
                marker=dict(size=3, color="black"),
                name="target_action_xyz",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def _save_state_action_plot(
    out_path: Path,
    cond_state: torch.Tensor,   # [K,L,S]
    query_state: torch.Tensor,  # [T_obs,S]
    target_action: torch.Tensor,  # [H,A]
    title: str,
) -> None:
    cond = cond_state.detach().cpu().numpy()
    query = query_state.detach().cpu().numpy()
    action = target_action.detach().cpu().numpy()
    K, L, S = cond.shape
    H, A = action.shape

    cond_flat = cond.reshape(K * L, S)
    max_lines = min(16, max(S, A))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), dpi=120, constrained_layout=True)
    axes[0].set_title("Support state (flattened over K*L)")
    for d in range(min(S, max_lines)):
        axes[0].plot(cond_flat[:, d], linewidth=1.0, label=f"s{d}")
    axes[0].set_xlabel("support step (k,l -> flat)")
    axes[0].set_ylabel("value")
    if S <= 10:
        axes[0].legend(ncol=5, fontsize=7)

    axes[1].set_title("Query state (T_obs)")
    for d in range(min(S, max_lines)):
        axes[1].plot(query[:, d], linewidth=1.2, label=f"s{d}")
    axes[1].set_xlabel("query t")
    axes[1].set_ylabel("value")
    if S <= 10:
        axes[1].legend(ncol=5, fontsize=7)

    axes[2].set_title("Target action (H)")
    for d in range(min(A, max_lines)):
        axes[2].plot(action[:, d], linewidth=1.2, label=f"a{d}")
    axes[2].set_xlabel("horizon step")
    axes[2].set_ylabel("value")
    if A <= 10:
        axes[2].legend(ncol=5, fontsize=7)

    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return str(obj)


def _summarize_store(store: VariationStore, tasks_used: Sequence[str]) -> Dict[str, Any]:
    per_variation: List[Dict[str, Any]] = []
    total_episodes = 0
    all_lengths: List[int] = []
    for vidx, key in enumerate(store.keys):
        eids = store.list_episode_ids(vidx)
        ep_count = int(eids.shape[0])
        lengths = [int(store.episode_length(vidx, int(eid))) for eid in eids]
        total_episodes += ep_count
        all_lengths.extend(lengths)
        per_variation.append(
            {
                "vidx": vidx,
                "task": key.task,
                "variation": int(key.variation),
                "path": key.path,
                "episodes": ep_count,
                "length_min": int(min(lengths)) if lengths else 0,
                "length_max": int(max(lengths)) if lengths else 0,
                "length_mean": float(np.mean(lengths)) if lengths else 0.0,
            }
        )
    summary = {
        "num_tasks": len(tasks_used),
        "tasks": list(tasks_used),
        "num_variations": len(store),
        "total_episodes": int(total_episodes),
        "episode_length_min": int(min(all_lengths)) if all_lengths else 0,
        "episode_length_max": int(max(all_lengths)) if all_lengths else 0,
        "episode_length_mean": float(np.mean(all_lengths)) if all_lengths else 0.0,
        "variations": per_variation,
    }
    return summary


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    store, tasks_used = _build_store(
        cache_root=args.cache_root,
        tasks=args.tasks,
        keep_open_per_worker=bool(args.keep_open_per_worker),
    )

    try:
        N, S, A = _infer_dims(store)
        cfg = ICILConfig(
            K=int(args.K),
            L=int(args.L),
            T_obs=int(args.T_obs),
            H=int(args.H),
            stride=int(args.stride),
            action_representation=str(args.action_representation),
        )

        pretrain_ds = ICILPretrainBatchIterable(
            store=store,
            cfg=cfg,
            batch_size_B=int(args.batch_size),
            num_batches=int(args.num_batches),
            seed=int(args.seed),
            num_tries_per_item=int(args.num_tries_per_item),
        )
        loader = DataLoader(
            pretrain_ds,
            batch_size=1,
            collate_fn=_unwrap_single,
            num_workers=int(args.num_workers),
            pin_memory=False,
            persistent_workers=(int(args.num_workers) > 0),
        )

        summary = {
            "args": _to_serializable(vars(args)),
            "dims": {"N": N, "state_dim": S, "action_dim": A},
            "cfg": _to_serializable(cfg.__dict__),
            "store": _summarize_store(store, tasks_used),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        batch_iter = enumerate(loader)
        if not args.no_progress:
            try:
                from tqdm.auto import tqdm
                batch_iter = enumerate(tqdm(loader, total=int(args.num_batches), desc="inspect-batches"))
            except Exception:
                pass

        index_lines: List[str] = []
        index_lines.append("# ICIL Pretrain Cache Inspection")
        index_lines.append("")
        index_lines.append(f"- cache_root: `{args.cache_root}`")
        index_lines.append(f"- output_dir: `{output_dir}`")
        index_lines.append(f"- tasks_used: `{len(tasks_used)}`")
        index_lines.append(f"- variations: `{len(store)}`")
        index_lines.append(f"- dims: N={N}, S={S}, A={A}")
        index_lines.append("")

        for batch_idx, batch in batch_iter:
            batch_dir = output_dir / f"batch_{batch_idx:04d}"
            batch_dir.mkdir(parents=True, exist_ok=True)

            B = int(batch["cond_xyz"].shape[0])
            batch_meta = {
                "batch_idx": batch_idx,
                "shapes": {k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)},
                "meta": _to_serializable(batch.get("meta", [])),
            }
            (batch_dir / "batch_meta.json").write_text(
                json.dumps(batch_meta, indent=2),
                encoding="utf-8",
            )
            index_lines.append(f"## batch_{batch_idx:04d}")
            index_lines.append(f"- samples: {B}")
            index_lines.append(f"- metadata: `batch_{batch_idx:04d}/batch_meta.json`")

            max_samples = min(B, int(args.samples_per_batch))
            for sample_idx in range(max_samples):
                sample_dir = batch_dir / f"sample_{sample_idx:03d}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                cond_xyz = batch["cond_xyz"][sample_idx]       # [K,L,N,3]
                cond_valid = batch["cond_valid"][sample_idx]   # [K,L,N]
                query_xyz = batch["query_xyz"][sample_idx]     # [T_obs,N,3]
                query_valid = batch["query_valid"][sample_idx] # [T_obs,N]
                cond_state = batch["cond_state"][sample_idx]   # [K,L,S]
                query_state = batch["query_state"][sample_idx] # [T_obs,S]
                target_action = batch["target_action"][sample_idx]  # [H,A]

                K = int(cond_xyz.shape[0])
                L = int(cond_xyz.shape[1])
                support_xyz_seq = cond_xyz.reshape(K * L, cond_xyz.shape[2], 3)
                support_valid_seq = cond_valid.reshape(K * L, cond_valid.shape[2])

                action_xyz = None
                if target_action.shape[1] >= 3:
                    target_action_for_plot = decode_action_chunk(
                        target_action.unsqueeze(0),
                        query_state=query_state.unsqueeze(0),
                        representation=str(cfg.action_representation),
                    )[0]
                    action_xyz = target_action_for_plot[:, :3].detach().cpu().numpy()

                support_html = sample_dir / "support_frames.html"
                query_html = sample_dir / "query_frames_with_actions.html"
                state_png = sample_dir / "states_actions.png"

                _save_cloud_sequence_html(
                    out_path=support_html,
                    xyz_seq=support_xyz_seq,
                    valid_seq=support_valid_seq,
                    title=f"batch {batch_idx} sample {sample_idx} | support frames (K*L={K*L})",
                    rng=rng,
                    max_points_per_trace=int(args.max_points_per_trace),
                )
                _save_cloud_sequence_html(
                    out_path=query_html,
                    xyz_seq=query_xyz,
                    valid_seq=query_valid,
                    title=(
                        f"batch {batch_idx} sample {sample_idx} | "
                        f"query frames + target action xyz ({cfg.action_representation})"
                    ),
                    rng=rng,
                    max_points_per_trace=int(args.max_points_per_trace),
                    action_xyz=action_xyz,
                )
                _save_state_action_plot(
                    out_path=state_png,
                    cond_state=cond_state,
                    query_state=query_state,
                    target_action=target_action,
                    title=f"batch {batch_idx} sample {sample_idx}",
                )

                sample_meta = {
                    "batch_idx": batch_idx,
                    "sample_idx": sample_idx,
                    "cond_xyz_shape": list(cond_xyz.shape),
                    "query_xyz_shape": list(query_xyz.shape),
                    "cond_state_shape": list(cond_state.shape),
                    "query_state_shape": list(query_state.shape),
                    "target_action_shape": list(target_action.shape),
                    "action_representation": str(cfg.action_representation),
                    "meta": _to_serializable(batch.get("meta", [])[sample_idx] if sample_idx < len(batch.get("meta", [])) else {}),
                }
                (sample_dir / "sample_meta.json").write_text(
                    json.dumps(sample_meta, indent=2),
                    encoding="utf-8",
                )
                index_lines.append(f"- sample_{sample_idx:03d}:")
                index_lines.append(f"  - `batch_{batch_idx:04d}/sample_{sample_idx:03d}/support_frames.html`")
                index_lines.append(f"  - `batch_{batch_idx:04d}/sample_{sample_idx:03d}/query_frames_with_actions.html`")
                index_lines.append(f"  - `batch_{batch_idx:04d}/sample_{sample_idx:03d}/states_actions.png`")
                index_lines.append(f"  - `batch_{batch_idx:04d}/sample_{sample_idx:03d}/sample_meta.json`")
            index_lines.append("")

        (output_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
        print(f"[inspect] wrote inspection outputs to {output_dir}")
        print(f"[inspect] summary: {output_dir / 'summary.json'}")
        print(f"[inspect] index:   {output_dir / 'index.md'}")
    finally:
        store.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from icil.datasets.in_context_imitation_learning.icil_datasets import ICILConfig, ICILPretrainBatchIterable
from icil.datasets.in_context_imitation_learning.variation_store import VariationStore, build_variation_keys
from icil.models.common import SupernodeFrameTokenizer, SupernodeFrameTokenizerConfig

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover
    raise ImportError("plotly is required for HTML point-cloud visualization. Install with: pip install plotly") from exc


_BUCKET_NAMES = {
    -1: "duplicate/pad",
    0: "global_fps",
    1: "gripper_quota",
    2: "mask_quota",
}
_BUCKET_STYLE = {
    -1: ("black", "x"),
    0: ("royalblue", "diamond"),
    1: ("crimson", "diamond"),
    2: ("darkorange", "diamond"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the UPT-style supernode tokenizer on cached ICIL point-cloud frames."
    )
    parser.add_argument("--cache-root", type=Path, default=Path("output_data_playground_v3/.rlbench_cache_dense"))
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional task subset. If omitted, use all cached tasks.")
    parser.add_argument("--output-dir", type=Path, default=Path("output_data_playground_v3/.inspection/supernode_tokenizer"))
    parser.add_argument("--device", default="cpu", help="cpu or cuda. CPU is usually enough for diagnostics.")

    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--T-obs", dest="T_obs", type=int, default=2)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--samples-per-batch", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-tries-per-item", type=int, default=100)
    parser.add_argument("--keep-open-per-worker", action="store_true")
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--branch", choices=("demo", "query", "both"), default="both")
    parser.add_argument("--max-support-frames", type=int, default=8)
    parser.add_argument("--max-query-frames", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=6000, help="Max original points rendered per frame.")
    parser.add_argument("--max-supernodes-with-edges", type=int, default=32)
    parser.add_argument("--max-neighbors-per-supernode", type=int, default=16)

    parser.add_argument("--d-model", type=int, default=64, help="Tokenizer width for diagnostics; sampling is width-independent.")
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--demo-supernodes", type=int, default=32)
    parser.add_argument("--query-supernodes", type=int, default=32)
    parser.add_argument("--demo-frame-tokens-out", type=int, default=64)
    parser.add_argument("--query-frame-tokens-out", type=int, default=128)
    parser.add_argument("--neighbors-per-supernode", type=int, default=32)
    parser.add_argument(
        "--supernode-sampling-mode",
        choices=("fps", "exact_fps", "fast", "fast_random"),
        default="fps",
        help="Use the same sampler as training. fast_random avoids the slow per-frame FPS fill.",
    )
    parser.add_argument("--demo-refine-layers", type=int, default=1)
    parser.add_argument("--query-refine-layers", type=int, default=2)
    parser.add_argument("--supernode-pool-layers", type=int, default=1)
    parser.add_argument("--no-compress-demo", action="store_true")
    parser.add_argument("--no-compress-query", action="store_true")
    parser.add_argument("--min-gripper-supernodes", type=int, default=2)
    parser.add_argument("--min-mask-supernodes", type=int, default=4)
    parser.add_argument("--gripper-radius", type=float, default=0.10)
    parser.add_argument("--gripper-xyz-state-start", type=int, default=0)
    parser.add_argument("--use-gripper-point-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-mask-instance-quota", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-mask-embedding", action="store_true", help="Off by default; mask ids are otherwise quota-only.")
    return parser.parse_args()


def _unwrap_single(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    return batch_list[0]


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _discover_cached_tasks(cache_root: Path) -> List[str]:
    if not cache_root.is_dir():
        return []
    return [p.name for p in sorted(cache_root.iterdir()) if p.is_dir() and any(p.glob("variation*.h5"))]


def _build_store(cache_root: Path, tasks: Sequence[str] | None, keep_open_per_worker: bool) -> Tuple[VariationStore, List[str]]:
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
        keys.extend(task_keys)
    if missing:
        raise RuntimeError(f"Missing variation*.h5 for tasks: {', '.join(sorted(missing))}")
    return VariationStore(keys, keep_open_per_worker=keep_open_per_worker), selected


def _infer_dims(store: VariationStore) -> Tuple[int, int, int]:
    for vidx in range(len(store)):
        episode_ids = store.list_episode_ids(vidx)
        if episode_ids.shape[0] == 0:
            continue
        sample = store.load_episode_slices(
            vidx,
            int(episode_ids[0]),
            np.asarray([0], dtype=np.int64),
            load_rgb=False,
            load_mask_id=False,
        )
        return int(sample["xyz"].shape[1]), int(sample["state"].shape[-1]), int(sample["action"].shape[-1])
    raise RuntimeError("Could not infer dimensions from cache.")


def _frame_indices(total: int, max_frames: int) -> List[int]:
    if total <= 0 or max_frames <= 0:
        return []
    n = min(total, max_frames)
    return np.linspace(0, total - 1, n, dtype=np.int64).tolist()


def _downsample_valid_indices(valid: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(valid.astype(bool))
    if max_points > 0 and idx.shape[0] > max_points:
        idx = np.sort(rng.choice(idx, size=max_points, replace=False)).astype(np.int64)
    return idx.astype(np.int64)


def _rgb_strings(rgb: np.ndarray) -> List[str]:
    rgb = np.asarray(rgb)
    if rgb.max(initial=0.0) <= 1.5:
        rgb = rgb * 255.0
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in rgb[:, :3]]


def _tokenizer_cfg(args: argparse.Namespace, *, branch: str) -> SupernodeFrameTokenizerConfig:
    if branch == "demo":
        num_supernodes = int(args.demo_supernodes)
        frame_tokens_out = int(args.demo_frame_tokens_out)
        refine_layers = int(args.demo_refine_layers)
        compress = not bool(args.no_compress_demo)
    elif branch == "query":
        num_supernodes = int(args.query_supernodes)
        frame_tokens_out = int(args.query_frame_tokens_out)
        refine_layers = int(args.query_refine_layers)
        compress = not bool(args.no_compress_query)
    else:
        raise ValueError(f"Unsupported branch={branch!r}")
    return SupernodeFrameTokenizerConfig(
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        dropout=0.0,
        num_supernodes=num_supernodes,
        frame_tokens_out=frame_tokens_out,
        neighbors_per_supernode=int(args.neighbors_per_supernode),
        supernode_refine_layers=refine_layers,
        compress_supernodes=compress,
        supernode_pool_layers=int(args.supernode_pool_layers),
        use_mask_id=True,
        use_mask_embedding=bool(args.use_mask_embedding),
        mask_hash_buckets=2048 if bool(args.use_mask_embedding) else 1,
        supernode_sampling_mode=str(args.supernode_sampling_mode),
        use_mask_instance_quota=bool(args.use_mask_instance_quota),
        min_mask_supernodes=int(args.min_mask_supernodes),
        use_gripper_point_features=bool(args.use_gripper_point_features),
        gripper_xyz_state_start=int(args.gripper_xyz_state_start),
        gripper_alpha_init=1.0,
        min_gripper_supernodes=int(args.min_gripper_supernodes),
        gripper_sampling_radius=float(args.gripper_radius),
        rgb_alpha_init=1.0,
    )


def _coverage_summary(diag: Dict[str, Any], mask_id: torch.Tensor | None, valid: torch.Tensor) -> Dict[str, Any]:
    bucket = diag["sample_bucket"][0].detach().cpu().numpy().astype(np.int64)
    super_mask = diag["supernode_mask"][0].detach().cpu().numpy().astype(bool)
    neighbor_mask = diag["neighbor_mask"][0].detach().cpu().numpy().astype(bool)
    neighbor_idx = diag["neighbor_idx"][0].detach().cpu().numpy().astype(np.int64)
    covered = np.unique(neighbor_idx[neighbor_mask]) if neighbor_mask.any() else np.asarray([], dtype=np.int64)
    valid_np = valid.detach().cpu().numpy().astype(bool)
    out: Dict[str, Any] = {
        "num_valid_points": int(valid_np.sum()),
        "num_valid_supernodes": int(super_mask.sum()),
        "bucket_counts": {
            _BUCKET_NAMES.get(int(b), str(int(b))): int(((bucket == int(b)) & super_mask).sum())
            for b in sorted(set(bucket.tolist()))
        },
        "covered_valid_points": int(valid_np[covered].sum()) if covered.size else 0,
        "coverage_fraction_valid": float(valid_np[covered].sum() / max(1, valid_np.sum())) if covered.size else 0.0,
    }
    neighbor_counts = neighbor_mask.sum(axis=1)
    out["neighbor_count_min"] = int(neighbor_counts.min()) if neighbor_counts.size else 0
    out["neighbor_count_max"] = int(neighbor_counts.max()) if neighbor_counts.size else 0
    out["neighbor_count_mean"] = float(neighbor_counts.mean()) if neighbor_counts.size else 0.0
    if mask_id is not None:
        mask_np = mask_id.detach().cpu().numpy()
        visible_ids = np.unique(mask_np[valid_np]).astype(np.int64)
        selected_idx = diag["supernode_idx"][0].detach().cpu().numpy().astype(np.int64)[super_mask]
        selected_ids = np.unique(mask_np[selected_idx]).astype(np.int64) if selected_idx.size else np.asarray([], dtype=np.int64)
        out["num_visible_mask_ids"] = int(visible_ids.shape[0])
        out["visible_mask_ids"] = visible_ids.tolist()
        out["selected_mask_ids"] = selected_ids.tolist()
        out["mask_id_coverage_fraction"] = float(selected_ids.shape[0] / max(1, visible_ids.shape[0]))
    return out


def _save_supernode_html(
    out_path: Path,
    *,
    xyz: torch.Tensor,
    valid: torch.Tensor,
    rgb: torch.Tensor | None,
    mask_id: torch.Tensor | None,
    diag: Dict[str, Any],
    title: str,
    rng: np.random.Generator,
    max_points: int,
    max_supernodes_with_edges: int,
    max_neighbors_per_supernode: int,
) -> Dict[str, Any]:
    xyz_np = xyz.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy().astype(bool)
    point_idx = _downsample_valid_indices(valid_np, int(max_points), rng)
    p = xyz_np[point_idx]

    fig = go.Figure()
    marker: Dict[str, Any] = {"size": 1.8, "opacity": 0.55}
    if rgb is not None:
        rgb_np = rgb.detach().cpu().numpy()[point_idx]
        marker["color"] = _rgb_strings(rgb_np)
        marker["showscale"] = False
    elif mask_id is not None:
        mask_np = mask_id.detach().cpu().numpy()[point_idx]
        marker.update({"color": mask_np, "colorscale": "Turbo", "showscale": True, "colorbar": {"title": "mask_id"}})
    else:
        marker["color"] = "lightgray"

    fig.add_trace(
        go.Scatter3d(
            x=p[:, 0],
            y=p[:, 1],
            z=p[:, 2],
            mode="markers",
            marker=marker,
            name="valid_points",
        )
    )

    super_xyz = diag["supernode_xyz"][0].detach().cpu().numpy()
    super_idx = diag["supernode_idx"][0].detach().cpu().numpy().astype(np.int64)
    super_mask = diag["supernode_mask"][0].detach().cpu().numpy().astype(bool)
    bucket = diag["sample_bucket"][0].detach().cpu().numpy().astype(np.int64)
    neighbor_idx = diag["neighbor_idx"][0].detach().cpu().numpy().astype(np.int64)
    neighbor_mask = diag["neighbor_mask"][0].detach().cpu().numpy().astype(bool)

    for b in sorted(set(bucket.tolist())):
        keep = (bucket == int(b)) & super_mask
        if not keep.any():
            continue
        color, symbol = _BUCKET_STYLE.get(int(b), ("purple", "diamond"))
        pts = super_xyz[keep]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker={"size": 6.0, "color": color, "symbol": symbol, "opacity": 0.95},
                name=f"supernodes:{_BUCKET_NAMES.get(int(b), int(b))}",
                text=[f"point_idx={int(i)} bucket={_BUCKET_NAMES.get(int(b), int(b))}" for i in super_idx[keep]],
            )
        )

    valid_super = np.flatnonzero(super_mask)
    if valid_super.size > 0 and max_supernodes_with_edges > 0:
        if valid_super.size > max_supernodes_with_edges:
            chosen_super = np.linspace(0, valid_super.size - 1, int(max_supernodes_with_edges), dtype=np.int64)
            valid_super = valid_super[chosen_super]
        line_x: List[float | None] = []
        line_y: List[float | None] = []
        line_z: List[float | None] = []
        covered_points: List[int] = []
        for sidx in valid_super:
            neigh = neighbor_idx[sidx][neighbor_mask[sidx]]
            if neigh.size > max_neighbors_per_supernode:
                neigh = neigh[: int(max_neighbors_per_supernode)]
            covered_points.extend(int(i) for i in neigh.tolist())
            s = super_xyz[sidx]
            for nidx in neigh:
                q = xyz_np[int(nidx)]
                line_x.extend([float(s[0]), float(q[0]), None])
                line_y.extend([float(s[1]), float(q[1]), None])
                line_z.extend([float(s[2]), float(q[2]), None])
        if line_x:
            fig.add_trace(
                go.Scatter3d(
                    x=line_x,
                    y=line_y,
                    z=line_z,
                    mode="lines",
                    line={"color": "rgba(20,20,20,0.28)", "width": 2},
                    name="message_edges_subset",
                )
            )
        if covered_points:
            covered = np.asarray(sorted(set(covered_points)), dtype=np.int64)
            cpts = xyz_np[covered]
            fig.add_trace(
                go.Scatter3d(
                    x=cpts[:, 0],
                    y=cpts[:, 1],
                    z=cpts[:, 2],
                    mode="markers",
                    marker={"size": 3.0, "color": "rgba(0,120,0,0.45)", "opacity": 0.45},
                    name="covered_neighbor_points_subset",
                )
            )

    summary = _coverage_summary(diag, mask_id=mask_id, valid=valid)
    fig.update_layout(
        title=f"{title}<br>{json.dumps(summary, sort_keys=True)}",
        scene={"xaxis_title": "x", "yaxis_title": "y", "zaxis_title": "z", "aspectmode": "data"},
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "b": 0, "t": 80},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return summary


def _run_tokenizer(
    tokenizer: SupernodeFrameTokenizer,
    *,
    device: torch.device,
    xyz: torch.Tensor,
    valid: torch.Tensor,
    state: torch.Tensor,
    rgb: torch.Tensor | None,
    mask_id: torch.Tensor | None,
) -> Dict[str, Any]:
    tokenizer.eval()
    with torch.no_grad():
        _, diag = tokenizer(
            xyz=xyz.unsqueeze(0).to(device),
            valid=valid.unsqueeze(0).to(device),
            state=state.unsqueeze(0).to(device),
            rgb=None if rgb is None else rgb.unsqueeze(0).to(device),
            mask_id=None if mask_id is None else mask_id.unsqueeze(0).to(device),
            return_diagnostics=True,
        )
    return diag


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is not available.")

    store, tasks_used = _build_store(args.cache_root.expanduser().resolve(), args.tasks, bool(args.keep_open_per_worker))
    try:
        N, state_dim, action_dim = _infer_dims(store)
        cfg = ICILConfig(K=int(args.K), L=int(args.L), T_obs=int(args.T_obs), H=int(args.H), stride=int(args.stride))
        dataset = ICILPretrainBatchIterable(
            store=store,
            cfg=cfg,
            batch_size_B=int(args.batch_size),
            num_batches=int(args.num_batches),
            seed=int(args.seed),
            num_tries_per_item=int(args.num_tries_per_item),
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=_unwrap_single,
            num_workers=int(args.num_workers),
            pin_memory=False,
            persistent_workers=(int(args.num_workers) > 0),
        )
        demo_tokenizer = SupernodeFrameTokenizer(cfg=_tokenizer_cfg(args, branch="demo"), state_dim=state_dim).to(device)
        query_tokenizer = SupernodeFrameTokenizer(cfg=_tokenizer_cfg(args, branch="query"), state_dim=state_dim).to(device)

        batch_iter = enumerate(loader)
        if not bool(args.no_progress):
            try:
                from tqdm.auto import tqdm

                batch_iter = enumerate(tqdm(loader, total=int(args.num_batches), desc="supernode-inspect"))
            except Exception:
                pass

        summary = {
            "args": _to_serializable(vars(args)),
            "dims": {"N": N, "state_dim": state_dim, "action_dim": action_dim},
            "tasks_used": tasks_used,
            "num_variations": len(store),
            "demo_tokenizer": _to_serializable(_tokenizer_cfg(args, branch="demo").__dict__),
            "query_tokenizer": _to_serializable(_tokenizer_cfg(args, branch="query").__dict__),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        index_lines = [
            "# Supernode Tokenizer Inspection",
            "",
            f"- cache_root: `{args.cache_root}`",
            f"- output_dir: `{output_dir}`",
            f"- tasks_used: `{len(tasks_used)}`",
            f"- variations: `{len(store)}`",
            f"- dims: N={N}, state_dim={state_dim}, action_dim={action_dim}",
            "",
        ]
        all_frame_summaries: List[Dict[str, Any]] = []

        for batch_idx, batch in batch_iter:
            B = int(batch["cond_xyz"].shape[0])
            batch_dir = output_dir / f"batch_{batch_idx:04d}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            index_lines.append(f"## batch_{batch_idx:04d}")
            index_lines.append(f"- metadata: `batch_{batch_idx:04d}/batch_meta.json`")
            (batch_dir / "batch_meta.json").write_text(
                json.dumps(
                    {
                        "batch_idx": batch_idx,
                        "shapes": {k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)},
                        "meta": _to_serializable(batch.get("meta", [])),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            for sample_idx in range(min(B, int(args.samples_per_batch))):
                sample_dir = batch_dir / f"sample_{sample_idx:03d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                sample_meta = batch.get("meta", [])[sample_idx] if sample_idx < len(batch.get("meta", [])) else {}
                index_lines.append(f"- sample_{sample_idx:03d}: `{batch_dir.name}/sample_{sample_idx:03d}/sample_meta.json`")

                if args.branch in ("demo", "both"):
                    cond_xyz = batch["cond_xyz"][sample_idx]
                    cond_valid = batch["cond_valid"][sample_idx]
                    cond_state = batch["cond_state"][sample_idx]
                    cond_rgb = batch.get("cond_rgb", None)
                    cond_rgb = None if cond_rgb is None else cond_rgb[sample_idx]
                    cond_mask = batch.get("cond_mask_id", None)
                    cond_mask = None if cond_mask is None else cond_mask[sample_idx]
                    K, L = int(cond_xyz.shape[0]), int(cond_xyz.shape[1])
                    for flat_i in _frame_indices(K * L, int(args.max_support_frames)):
                        k = int(flat_i // L)
                        l = int(flat_i % L)
                        diag = _run_tokenizer(
                            demo_tokenizer,
                            device=device,
                            xyz=cond_xyz[k, l],
                            valid=cond_valid[k, l],
                            state=cond_state[k, l],
                            rgb=None if cond_rgb is None else cond_rgb[k, l],
                            mask_id=None if cond_mask is None else cond_mask[k, l],
                        )
                        rel = f"support_k{k:02d}_l{l:02d}.html"
                        frame_summary = _save_supernode_html(
                            sample_dir / rel,
                            xyz=cond_xyz[k, l],
                            valid=cond_valid[k, l],
                            rgb=None if cond_rgb is None else cond_rgb[k, l],
                            mask_id=None if cond_mask is None else cond_mask[k, l],
                            diag=diag,
                            title=f"batch {batch_idx} sample {sample_idx} support k={k} l={l}",
                            rng=rng,
                            max_points=int(args.max_points),
                            max_supernodes_with_edges=int(args.max_supernodes_with_edges),
                            max_neighbors_per_supernode=int(args.max_neighbors_per_supernode),
                        )
                        frame_summary.update({"batch_idx": batch_idx, "sample_idx": sample_idx, "branch": "demo", "k": k, "l": l, "html": rel})
                        all_frame_summaries.append(frame_summary)
                        index_lines.append(f"  - support k={k} l={l}: `{batch_dir.name}/sample_{sample_idx:03d}/{rel}`")

                if args.branch in ("query", "both"):
                    query_xyz = batch["query_xyz"][sample_idx]
                    query_valid = batch["query_valid"][sample_idx]
                    query_state = batch["query_state"][sample_idx]
                    query_rgb = batch.get("query_rgb", None)
                    query_rgb = None if query_rgb is None else query_rgb[sample_idx]
                    query_mask = batch.get("query_mask_id", None)
                    query_mask = None if query_mask is None else query_mask[sample_idx]
                    Tobs = int(query_xyz.shape[0])
                    for t in _frame_indices(Tobs, int(args.max_query_frames)):
                        diag = _run_tokenizer(
                            query_tokenizer,
                            device=device,
                            xyz=query_xyz[t],
                            valid=query_valid[t],
                            state=query_state[t],
                            rgb=None if query_rgb is None else query_rgb[t],
                            mask_id=None if query_mask is None else query_mask[t],
                        )
                        rel = f"query_t{t:02d}.html"
                        frame_summary = _save_supernode_html(
                            sample_dir / rel,
                            xyz=query_xyz[t],
                            valid=query_valid[t],
                            rgb=None if query_rgb is None else query_rgb[t],
                            mask_id=None if query_mask is None else query_mask[t],
                            diag=diag,
                            title=f"batch {batch_idx} sample {sample_idx} query t={t}",
                            rng=rng,
                            max_points=int(args.max_points),
                            max_supernodes_with_edges=int(args.max_supernodes_with_edges),
                            max_neighbors_per_supernode=int(args.max_neighbors_per_supernode),
                        )
                        frame_summary.update({"batch_idx": batch_idx, "sample_idx": sample_idx, "branch": "query", "t": int(t), "html": rel})
                        all_frame_summaries.append(frame_summary)
                        index_lines.append(f"  - query t={t}: `{batch_dir.name}/sample_{sample_idx:03d}/{rel}`")

                (sample_dir / "sample_meta.json").write_text(
                    json.dumps({"meta": _to_serializable(sample_meta)}, indent=2),
                    encoding="utf-8",
                )
            index_lines.append("")

        (output_dir / "frame_summaries.json").write_text(json.dumps(all_frame_summaries, indent=2), encoding="utf-8")
        (output_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
        print(f"[supernode-inspect] wrote outputs to {output_dir}")
        print(f"[supernode-inspect] index: {output_dir / 'index.md'}")
    finally:
        store.close()


if __name__ == "__main__":
    main()

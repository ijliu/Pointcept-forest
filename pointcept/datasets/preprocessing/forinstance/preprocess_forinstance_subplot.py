import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def find_plot_dirs(input_root: Path):
    """Find directories containing coord.npy. segment.npy is optional."""
    plot_dirs = []
    for coord_path in input_root.rglob("coord.npy"):
        plot_dirs.append(coord_path.parent)
    return sorted(set(plot_dirs))


def query_cylinder_xy(tree: cKDTree, xy: np.ndarray, center_xy: np.ndarray, radius: float) -> np.ndarray:
    """Return point indices inside an XY cylinder (circle in XY, full height in Z)."""
    idx = tree.query_ball_point(center_xy, r=radius)
    if len(idx) == 0:
        return np.empty((0,), dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    return np.sort(idx)


def query_square_xy(tree: cKDTree, xy: np.ndarray, center_xy: np.ndarray, half_size: float) -> np.ndarray:
    """
    Return point indices inside an XY square centered at center_xy.
    We first use a circumscribed circle query, then filter by square bounds.
    """
    search_r = half_size * math.sqrt(2.0)
    cand = tree.query_ball_point(center_xy, r=search_r)
    if len(cand) == 0:
        return np.empty((0,), dtype=np.int64)
    cand = np.asarray(cand, dtype=np.int64)
    delta = np.abs(xy[cand] - center_xy[None, :])
    keep = (delta[:, 0] <= half_size) & (delta[:, 1] <= half_size)
    return np.sort(cand[keep])


def generate_centers_1d(max_coord: float, tile_size: float, pass_id: int) -> np.ndarray:
    """
    pass 0: centers at tile_size/2, 3*tile_size/2, ...
    pass 1: centers at 0, tile_size, 2*tile_size, ...

    This guarantees pass 0 already covers [0, max_coord] with output cylinders of radius tile_size/2.
    pass 1 provides the half-window shifted overlap.
    """
    if pass_id == 0:
        num = int(math.floor(max_coord / tile_size)) + 1
        return tile_size / 2.0 + np.arange(num, dtype=np.float64) * tile_size
    elif pass_id == 1:
        num = int(math.floor(max_coord / tile_size)) + 2
        return np.arange(num, dtype=np.float64) * tile_size
    else:
        raise ValueError(f"Unsupported pass_id: {pass_id}")


def process_one_plot(
    plot_dir: Path,
    input_root: Path,
    output_root: Path,
    tile_size: float,
    min_points: int,
    max_input_half_size: float,
    expand_step: float,
    overwrite: bool,
):
    coord_path = plot_dir / "coord.npy"
    segment_path = plot_dir / "segment.npy"
    point_index_path = plot_dir / "point_index.npy"

    coord = np.load(coord_path)  # [N, 3]
    segment = np.load(segment_path) if segment_path.exists() else None
    point_index = np.load(point_index_path) if point_index_path.exists() else np.arange(coord.shape[0], dtype=np.int64)

    if coord.ndim != 2 or coord.shape[1] < 3:
        raise ValueError(f"coord.npy must have shape [N, 3], got {coord.shape} in {plot_dir}")
    if segment is not None and segment.shape[0] != coord.shape[0]:
        raise ValueError(f"segment.npy length mismatch in {plot_dir}")
    if point_index.shape[0] != coord.shape[0]:
        raise ValueError(f"point_index.npy length mismatch in {plot_dir}")

    scene_id = plot_dir.name

    # Flat output layout: save all subtiles directly under output_root.
    # Each tile folder is named as: <scene_id>_<pass_id>_<row>_<col>
    # This avoids nesting the original plot parent folders in the output.
    if overwrite:
        import shutil
        for old_tile_dir in output_root.glob(f"{scene_id}_[01]_[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9]"):
            if old_tile_dir.is_dir():
                shutil.rmtree(old_tile_dir)
        old_summary = output_root / f"{scene_id}_split_summary.json"
        if old_summary.exists():
            old_summary.unlink()

    xy = coord[:, :2].astype(np.float64)
    xy_local = xy - xy.min(axis=0, keepdims=True)
    tree = cKDTree(xy_local)

    x_max = float(xy_local[:, 0].max())
    y_max = float(xy_local[:, 1].max())

    output_radius = tile_size / 2.0
    base_input_half_size = tile_size / 2.0
    if max_input_half_size < base_input_half_size:
        raise ValueError("max_input_half_size must be >= tile_size / 2")

    total_tiles = 0
    expand_counter = 0

    for pass_id in (0, 1):
        center_x = generate_centers_1d(x_max, tile_size, pass_id)
        center_y = generate_centers_1d(y_max, tile_size, pass_id)

        for row_idx, cy in enumerate(center_y, start=1):
            for col_idx, cx in enumerate(center_x, start=1):
                center_xy = np.array([cx, cy], dtype=np.float64)

                # Output region: fixed inscribed cylinder in XY.
                keep_idx = query_cylinder_xy(tree, xy_local, center_xy, output_radius)
                if keep_idx.size == 0:
                    continue

                # Input region: adaptive square support in XY.
                input_half_size = base_input_half_size
                input_idx = query_square_xy(tree, xy_local, center_xy, input_half_size)
                while input_idx.size < min_points and input_half_size < max_input_half_size:
                    input_half_size = min(input_half_size + expand_step, max_input_half_size)
                    input_idx = query_square_xy(tree, xy_local, center_xy, input_half_size)

                if input_idx.size == 0:
                    # In practice this should not happen if keep_idx is non-empty,
                    # but keep the guard for safety.
                    continue

                keep_mask = np.isin(input_idx, keep_idx, assume_unique=False)
                if not np.any(keep_mask):
                    continue

                tile_name = f"{scene_id}_{pass_id}_{row_idx:04d}_{col_idx:04d}"
                tile_dir = output_root / tile_name
                tile_dir.mkdir(parents=True, exist_ok=True)

                np.save(tile_dir / "coord.npy", coord[input_idx].astype(np.float32))
                np.save(tile_dir / "point_index.npy", point_index[input_idx].astype(np.int64))
                np.save(tile_dir / "keep_mask.npy", keep_mask.astype(np.bool_))

                if segment is not None:
                    np.save(tile_dir / "segment.npy", segment[input_idx])

                meta = {
                    "scene_id": scene_id,
                    "tile_name": tile_name,
                    "pass_id": int(pass_id),
                    "row": int(row_idx),
                    "col": int(col_idx),
                    "center_xy_local": [float(cx), float(cy)],
                    "tile_size": float(tile_size),
                    "output_radius": float(output_radius),
                    "base_input_half_size": float(base_input_half_size),
                    "input_half_size_used": float(input_half_size),
                    "min_points": int(min_points),
                    "num_input_points": int(input_idx.size),
                    "num_keep_points": int(keep_mask.sum()),
                }
                with open(tile_dir / "meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)

                total_tiles += 1
                if input_half_size > base_input_half_size:
                    expand_counter += 1

    summary = {
        "scene_id": scene_id,
        "num_points": int(coord.shape[0]),
        "tile_size": float(tile_size),
        "output_radius": float(output_radius),
        "min_points": int(min_points),
        "max_input_half_size": float(max_input_half_size),
        "expand_step": float(expand_step),
        "total_tiles": int(total_tiles),
        "expanded_tiles": int(expand_counter),
    }
    with open(output_root / f"{scene_id}_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Split plot-level coord/segment npy files into adaptive-support subtiles with fixed output cylinders."
    )
    parser.add_argument("--input_root", type=str, required=True,
                        help="Root containing plot folders with coord.npy (and optional segment.npy / point_index.npy)")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root to save the generated subtiles")
    parser.add_argument("--tile_size", type=float, default=8.0,
                        help="Base square window side length in meters. Default: 8.0")
    parser.add_argument("--min_points", type=int, default=1000,
                        help="Minimum number of input points required for a subtile support region. Default: 1000")
    parser.add_argument("--max_input_half_size", type=float, default=8.0,
                        help="Maximum XY half-size of the adaptive input square. Default: 8.0 (i.e. up to 16m x 16m support)")
    parser.add_argument("--expand_step", type=float, default=2.0,
                        help="Expansion step of input half-size in meters. Default: 2.0")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Parallel workers across plots. Default: 1")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing plot output folders")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    plot_dirs = find_plot_dirs(input_root)
    if len(plot_dirs) == 0:
        raise RuntimeError(f"No plot folders with coord.npy were found under: {input_root}")

    print(f"Found {len(plot_dirs)} plot folders.")

    summaries = []
    if args.num_workers <= 1:
        for i, plot_dir in enumerate(plot_dirs, start=1):
            summary = process_one_plot(
                plot_dir=plot_dir,
                input_root=input_root,
                output_root=output_root,
                tile_size=args.tile_size,
                min_points=args.min_points,
                max_input_half_size=args.max_input_half_size,
                expand_step=args.expand_step,
                overwrite=args.overwrite,
            )
            summaries.append(summary)
            print(f"[{i}/{len(plot_dirs)}] Done: {summary['scene_id']} | tiles={summary['total_tiles']} | expanded={summary['expanded_tiles']}")
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [
                ex.submit(
                    process_one_plot,
                    plot_dir,
                    input_root,
                    output_root,
                    args.tile_size,
                    args.min_points,
                    args.max_input_half_size,
                    args.expand_step,
                    args.overwrite,
                )
                for plot_dir in plot_dirs
            ]
            for i, fut in enumerate(as_completed(futures), start=1):
                summary = fut.result()
                summaries.append(summary)
                print(f"[{i}/{len(plot_dirs)}] Done: {summary['scene_id']} | tiles={summary['total_tiles']} | expanded={summary['expanded_tiles']}")

    with open(output_root / "all_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print("All done.")


if __name__ == "__main__":
    main()

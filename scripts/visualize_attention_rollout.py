"""Overlay exported pi0 attention maps on LIBERO rollout images.

Example:
    python scripts/visualize_attention_rollout.py \
        --data-dir data/libero_plus \
        --rollout-name rollout_NAME \
        --view agentview \
        --output-dir data/libero_plus/attention_videos

To process every rollout below ``data/libero_plus``:
    python scripts/visualize_attention_rollout.py \
        --data-dir data/libero_plus \
        --all-rollouts \
        --view wristview

The default visualization averages heads 0 and 1 in layers 9, 10, 11, and
12, and averages all denoising snapshots in each .pt file.  ``--branch object``
can be used for pi0_object files that contain an ``object`` branch.  Use
``--heads`` to override the selected head indices.
"""

from __future__ import annotations

import argparse
import math
import pathlib
from typing import Any

import cv2
import imageio
import numpy as np
import torch


DEFAULT_LAYERS = (9, 10, 11, 12)


def _numeric_pt_key(path: pathlib.Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return math.inf, path.name


def _parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not layers:
        raise argparse.ArgumentTypeError("layers must contain at least one integer")
    return layers


def _parse_heads(value: str) -> tuple[int, ...]:
    heads = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not heads or any(head < 0 for head in heads):
        raise argparse.ArgumentTypeError("heads must contain one or more non-negative integers")
    return heads


def _view_index(view_names: list[str], view: str) -> int:
    if view == "agentview":
        candidates = ("base_0_rgb", "base_1_rgb")
    else:
        candidates = ("left_wrist_0_rgb", "wrist_0_rgb", "right_wrist_0_rgb")

    for candidate in candidates:
        if candidate in view_names:
            return view_names.index(candidate)

    raise ValueError(f"Cannot map --view {view!r} to exported view_names={view_names}")


def _select_snapshots(snapshots: list[dict[str, Any]], mode: str, index: int | None) -> list[dict[str, Any]]:
    if not snapshots:
        raise ValueError("The attention file contains no snapshots")
    if mode == "mean":
        return snapshots
    if mode == "first":
        return snapshots[:1]
    if mode == "last":
        return snapshots[-1:]
    if mode == "index":
        if index is None:
            raise ValueError("--snapshot-index is required when --snapshot index is used")
        if not -len(snapshots) <= index < len(snapshots):
            raise IndexError(f"snapshot index {index} is outside [ {-len(snapshots)}, {len(snapshots) - 1} ]")
        return [snapshots[index]]
    raise ValueError(f"Unknown snapshot mode: {mode}")


def _load_attention_file(path: pathlib.Path) -> dict[str, Any]:
    """Load an exporter-generated file across old and new PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        # PyTorch versions whose Unpickler predates the weights_only argument.
        # These files are local exporter outputs and are expected to contain a
        # plain metadata dictionary plus CPU tensors.
        return torch.load(path, map_location="cpu")


def _attention_vector(
    attention_payload: dict[str, Any],
    *,
    layers: tuple[int, ...],
    heads: tuple[int, ...],
    branch: str,
    view_index: int,
    snapshot_mode: str,
    snapshot_index: int | None,
) -> np.ndarray:
    view_names = list(attention_payload["view_names"])
    if not 0 <= view_index < len(view_names):
        raise IndexError(f"view index {view_index} is invalid for {view_names}")

    snapshots = _select_snapshots(attention_payload["snapshots"], snapshot_mode, snapshot_index)
    snapshot_maps = []
    for snapshot in snapshots:
        branch_maps = snapshot.get(branch)
        if branch_maps is None:
            available = [key for key in ("origin", "object") if key in snapshot]
            raise KeyError(
                f"Attention branch {branch!r} is not present. Available branches: {available}. "
                "Use --branch origin for pi0_raw, or regenerate pi0_object with object-head capture enabled."
            )

        layer_maps = []
        for layer in layers:
            key = str(layer)
            if key not in branch_maps:
                raise KeyError(f"Layer {layer} is not present in branch {branch!r}")
            tensor = branch_maps[key]
            if not torch.is_tensor(tensor):
                tensor = torch.as_tensor(tensor)
            if tensor.ndim != 3:
                raise ValueError(f"Expected [heads, views, patches], got {tuple(tensor.shape)} for layer {layer}")
            invalid_heads = [head for head in heads if head >= tensor.shape[0]]
            if invalid_heads:
                raise IndexError(
                    f"Head index/indices {invalid_heads} are invalid for layer {layer} "
                    f"with {tensor.shape[0]} stored heads"
                )
            if view_index >= tensor.shape[1]:
                raise IndexError(f"View index {view_index} is invalid for layer {layer} shape {tuple(tensor.shape)}")

            # Select the requested heads, average them, then select the view.
            layer_maps.append(tensor.float()[list(heads)].mean(dim=0)[view_index])

        # Mean over the requested layers for this denoising snapshot.
        snapshot_maps.append(torch.stack(layer_maps).mean(dim=0))

    # Mean over selected denoising snapshots.
    return torch.stack(snapshot_maps).mean(dim=0).cpu().numpy()


def _overlay(image: np.ndarray, attention_vector: np.ndarray, alpha: float) -> np.ndarray:
    patch_count = int(attention_vector.size)
    grid_size = math.isqrt(patch_count)
    if grid_size * grid_size != patch_count:
        raise ValueError(f"Patch count {patch_count} is not a square; cannot reshape to a 2D heatmap")

    heatmap = attention_vector.reshape(grid_size, grid_size)
    if not np.isfinite(heatmap).any():
        raise ValueError("Attention map contains no finite values; regenerate the .pt files after fixing export precision")
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    heatmap_min = float(heatmap.min())
    heatmap_max = float(heatmap.max())
    if heatmap_max > heatmap_min:
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap = np.zeros_like(heatmap)

    height, width = image.shape[:2]
    heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
    heatmap_u8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    heatmap_rgb = cv2.cvtColor(cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1.0 - alpha, heatmap_rgb, alpha, 0.0)


def _visualize_rollout(
    attention_dir: pathlib.Path,
    image_dir: pathlib.Path,
    output: pathlib.Path,
    *,
    view: str,
    branch: str,
    layers: tuple[int, ...],
    heads: tuple[int, ...],
    snapshot_mode: str,
    snapshot_index: int | None,
    alpha: float,
    fps: float,
) -> tuple[int, int]:
    attention_files = sorted(attention_dir.glob("*.pt"), key=_numeric_pt_key)
    if not attention_files:
        raise FileNotFoundError(f"No .pt files found under {attention_dir}")

    view_dir = image_dir / view
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with imageio.get_writer(output, fps=fps, codec="libx264", macro_block_size=1) as writer:
        for attention_file in attention_files:
            image_file = view_dir / f"{attention_file.stem}.png"
            if not image_file.exists():
                print(f"[warning] missing image for {attention_file.name}: {image_file}")
                skipped += 1
                continue

            payload = _load_attention_file(attention_file)
            attention_payload = payload["attention"]
            view_index = _view_index(list(attention_payload["view_names"]), view)
            vector = _attention_vector(
                attention_payload,
                layers=layers,
                heads=heads,
                branch=branch,
                view_index=view_index,
                snapshot_mode=snapshot_mode,
                snapshot_index=snapshot_index,
            )

            image = imageio.imread(image_file)
            if image.ndim == 2:
                image = np.repeat(image[..., None], 3, axis=2)
            image = image[..., :3].astype(np.uint8)
            writer.append_data(_overlay(image, vector, alpha))
            written += 1

    print(f"Wrote {written} frames to {output}")
    if skipped:
        print(f"Skipped {skipped} frames because the matching image was missing")
    return written, skipped


def _rollout_names(data_dir: pathlib.Path) -> list[str]:
    attention_root = data_dir / "rollout_attention"
    image_root = data_dir / "rollout_images"
    if not attention_root.is_dir():
        raise FileNotFoundError(f"Missing attention root: {attention_root}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing image root: {image_root}")

    attention_names = {path.name for path in attention_root.iterdir() if path.is_dir()}
    image_names = {path.name for path in image_root.iterdir() if path.is_dir()}
    missing_images = sorted(attention_names - image_names)
    missing_attention = sorted(image_names - attention_names)
    if missing_images:
        print(f"[warning] rollout(s) missing from rollout_images: {missing_images}")
    if missing_attention:
        print(f"[warning] rollout(s) missing from rollout_attention: {missing_attention}")

    names = sorted(attention_names & image_names)
    if not names:
        raise FileNotFoundError("No matching rollout directories found under rollout_attention and rollout_images")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        required=True,
        help="Root directory containing rollout_attention/ and rollout_images/.",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all-rollouts", action="store_true", help="Visualize every matching rollout")
    selection.add_argument("--rollout-name", type=str, help="Visualize one rollout directory by name")
    parser.add_argument("--view", choices=("agentview", "wristview"), default="agentview")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Output directory; defaults to <data-dir>/rollout_attention_videos/<view>.",
    )
    parser.add_argument("--branch", choices=("origin", "object"), default="origin")
    parser.add_argument("--layers", type=_parse_layers, default=DEFAULT_LAYERS)
    parser.add_argument(
        "--heads",
        type=_parse_heads,
        default=(0, 1),
        help="Comma-separated attention head indices to average, e.g. 0,1 or 0,1,2,3",
    )
    parser.add_argument("--snapshot", choices=("mean", "first", "last", "index"), default="mean")
    parser.add_argument("--snapshot-index", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be in [0, 1]")
    if args.fps <= 0:
        parser.error("--fps must be positive")

    data_dir = args.data_dir.expanduser().resolve()
    names = _rollout_names(data_dir) if args.all_rollouts else [args.rollout_name]
    assert all(name is not None for name in names)
    names = [name for name in names if name is not None]

    attention_root = data_dir / "rollout_attention"
    image_root = data_dir / "rollout_images"
    output_root = args.output_dir or data_dir / "rollout_attention_videos" / args.view

    failures = 0
    for name in names:
        attention_dir = attention_root / name / "attention"
        image_dir = image_root / name
        output = output_root / f"{name}.mp4"
        try:
            _visualize_rollout(
                attention_dir,
                image_dir,
                output,
                view=args.view,
                branch=args.branch,
                layers=args.layers,
                heads=args.heads,
                snapshot_mode=args.snapshot,
                snapshot_index=args.snapshot_index,
                alpha=args.alpha,
                fps=args.fps,
            )
        except Exception as exc:
            failures += 1
            print(f"[error] failed to visualize {name}: {type(exc).__name__}: {exc}")

    if failures:
        raise SystemExit(f"{failures} rollout(s) failed")


if __name__ == "__main__":
    main()

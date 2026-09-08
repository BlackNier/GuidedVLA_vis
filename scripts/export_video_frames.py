"""Export video frames at a fixed frame interval.

Examples:
    # Export frames 0, 6, 12, ... from one video.
    python scripts/export_video_frames.py \
        --input rollout.mp4 \
        --output-dir rollout_frames \
        --interval 6

    # Export every 10th frame from all videos in a directory.
    python scripts/export_video_frames.py \
        --input videos \
        --output-dir extracted_frames \
        --interval 10
"""

from __future__ import annotations

import argparse
import pathlib

import cv2


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _video_paths(input_path: pathlib.Path, recursive: bool) -> list[pathlib.Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)


def export_video(
    video_path: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    interval: int,
    image_format: str,
    jpeg_quality: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    saved = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # This produces frame indices 0, interval, 2*interval, ... .
            if frame_index % interval == 0:
                output_path = output_dir / f"frame_{frame_index:06d}.{image_format}"
                if image_format == "jpg":
                    write_ok = cv2.imwrite(
                        str(output_path),
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
                    )
                else:
                    write_ok = cv2.imwrite(str(output_path), frame)
                if not write_ok:
                    raise RuntimeError(f"Could not write frame: {output_path}")
                saved += 1

            frame_index += 1
    finally:
        capture.release()

    print(f"{video_path.name}: saved {saved} frames to {output_dir}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        required=True,
        help="A video file or a directory containing videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Output directory. Defaults to <video>_frames for one video, or <input>/frames for a directory.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=6,
        help="Save one frame every N input frames; frame indices are 0, N, 2N, ... .",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When --input is a directory, search for videos recursively.",
    )
    parser.add_argument(
        "--format",
        choices=("jpg", "png"),
        default="jpg",
        dest="image_format",
        help="Output image format.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality from 0 to 100; ignored for PNG.",
    )
    args = parser.parse_args()

    if args.interval <= 0:
        parser.error("--interval must be a positive integer")
    if not 0 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 0 and 100")

    input_path = args.input.expanduser().resolve()
    videos = _video_paths(input_path, args.recursive)
    if not videos:
        raise SystemExit(f"No supported videos found under: {input_path}")

    if args.output_dir is None:
        if input_path.is_file():
            output_root = input_path.parent / f"{input_path.stem}_frames"
        else:
            output_root = input_path / "frames"
    else:
        output_root = args.output_dir.expanduser().resolve()

    multiple_videos = len(videos) > 1 or input_path.is_dir()
    total_saved = 0
    for video_path in videos:
        if multiple_videos:
            video_output_dir = output_root / video_path.stem
        else:
            video_output_dir = output_root
        total_saved += export_video(
            video_path,
            video_output_dir,
            interval=args.interval,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
        )

    print(f"Done: exported {total_saved} frames from {len(videos)} video(s)")


if __name__ == "__main__":
    main()

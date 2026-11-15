"""CLI interface for running the Sign Language & Gesture Translator pipeline.

Supports input from:
1. Synthetic frame generator (zero external files required).
2. Pre-recorded video files (.mp4, .avi, .mov).
3. Live webcam streams (camera device index).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sign_translator.collect_data import OpenCVCaptureSource, SyntheticCaptureSource
from sign_translator.live_inference import ONNXModelRunner, RealtimeGesturePipeline
from sign_translator.schemas import PipelineConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-Time Sign Language & Gesture Translator CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="synthetic",
        help="Input video source: 'synthetic', path to a video file (.mp4/.avi), or camera index (e.g. '0')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/exported/gesture_model.onnx",
        help="Path to trained model (.onnx or .pt checkpoint)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process before stopping (default: run until stream ends)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device for inference: 'auto' (detects CUDA or falls back to CPU), 'cuda', or 'cpu'",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable Text-to-Speech audio output",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Debounce cooldown in seconds between audio trigger events",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("=" * 70)
    print(" Real-Time Sign Language & Gesture Translator")
    print("=" * 70)
    print(f" Source:      {args.source}")
    print(f" Model:       {args.model}")
    print(f" Device:      {args.device}")
    print(f" TTS Output:  {'Disabled' if args.no_tts else 'Enabled'}")
    print("=" * 70)

    # 1. Resolve Capture Source
    if args.source.lower() == "synthetic":
        total_frames = args.max_frames or 100
        print(f"==> Using synthetic stream ({total_frames} frames)...")
        capture_source = SyntheticCaptureSource(max_frames=total_frames)
    elif args.source.isdigit():
        camera_idx = int(args.source)
        print(f"==> Opening live webcam (device index {camera_idx})...")
        capture_source = OpenCVCaptureSource(source=camera_idx)
    else:
        video_path = Path(args.source)
        if not video_path.exists():
            print(f"Error: Video file not found: {video_path}", file=sys.stderr)
            return 1
        print(f"==> Opening video file: {video_path}...")
        capture_source = OpenCVCaptureSource(source=str(video_path))

    # 2. Resolve Model Runner
    model_path = Path(args.model)
    model_runner = None
    if model_path.exists() and model_path.suffix.lower() == ".onnx":
        print(f"==> Loading ONNX inference runtime: {model_path} (device: {args.device})...")
        model_runner = ONNXModelRunner(model_path, device=args.device)
    elif model_path.exists() and model_path.suffix.lower() == ".pt":
        from sign_translator.live_inference import PyTorchModelRunner
        from sign_translator.train import load_checkpoint

        print(f"==> Loading PyTorch checkpoint: {model_path} (device: {args.device})...")
        pt_model, _ = load_checkpoint(model_path)
        model_runner = PyTorchModelRunner(pt_model, device=args.device)

    config = PipelineConfig(
        cooldown_seconds=args.cooldown,
        device=args.device,
    )

    pipeline = RealtimeGesturePipeline(
        config=config,
        model=model_runner,
        capture_source=capture_source,
        enable_tts=not args.no_tts,
    )

    print("\nStarting inference loop (press Ctrl+C to stop)...")
    print("-" * 70)
    print(
        f"{'Frame':<8} | {'Prediction':<15} | {'Confidence':<10} | {'Stable':<8} | {'Latency':<10}"
    )
    print("-" * 70)

    frame_count = 0
    stable_events = 0
    try:
        while True:
            if args.max_frames is not None and frame_count >= args.max_frames:
                break
            ret, frame = capture_source.read()
            if not ret or frame is None:
                break
            frame_count += 1
            pred = pipeline.process_frame(frame)

            status_mark = " [*TTS*]" if pred.triggered_tts else ""
            if pred.is_stable and pred.label != "buffering":
                stable_events += 1

            if frame_count % 5 == 0 or pred.triggered_tts:
                print(
                    f"{frame_count:<8} | {pred.label:<15} | {pred.confidence:<10.3f} | "
                    f"{str(pred.is_stable):<8} | {pred.latency_ms:<8.2f} ms{status_mark}"
                )

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    finally:
        pipeline.close()

    print("-" * 70)
    print(f"Processed {frame_count} frames total. Detected {stable_events} stable gesture events.")
    print("Pipeline closed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

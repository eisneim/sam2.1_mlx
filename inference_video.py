"""SAM2.1 Video Object Tracking — click on an object in the first frame, track it through the video.

Usage:
    python inference_video.py -i path/to/video.mp4
    python inference_video.py -i path/to/video.mp4 --model small --weights weights/sam2.1_hiera_small.safetensors
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx
from sam2.build import build_model
from sam2.predictor import SAM2VideoPredictor


def load_video(video_path: str):
    """Load video frames as RGB numpy array and fps."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames), fps


def pick_point(frame_rgb: np.ndarray) -> tuple:
    """Show frame in a window, let user click on the object. Returns (x, y)."""
    display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    click_pos = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos.clear()
            click_pos.extend([x, y])

    win = "Click on the object to track (then press any key)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)
    cv2.setMouseCallback(win, on_click)

    while True:
        vis = display.copy()
        if click_pos:
            cv2.circle(vis, (click_pos[0], click_pos[1]), 8, (0, 0, 255), -1)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            return None
        if click_pos and key != 255:  # any key pressed after click
            break

    cv2.destroyAllWindows()
    return click_pos[0], click_pos[1]


def main():
    parser = argparse.ArgumentParser(description="SAM2.1 Video Object Tracking")
    parser.add_argument("-i", "--video", required=True, help="Input video path")
    parser.add_argument("--output", default="output", help="Output directory (default: output/)")
    parser.add_argument("--model", default="base_plus", choices=["small", "base_plus", "large"],
                        help="Model size (default: base_plus)")
    parser.add_argument("--weights", default=None,
                        help="Path to .safetensors weights (default: weights/sam2.1_hiera_{model}.safetensors)")
    args = parser.parse_args()

    # Load video
    print(f"Loading video: {args.video}")
    video, fps = load_video(args.video)
    N, H, W = video.shape[:3]
    print(f"  {N} frames, {W}x{H}, {fps:.1f} fps")

    # Let user click on the object
    click = pick_point(video[0])
    if click is None:
        print("No click — exiting.")
        return
    print(f"  Click at: {click}")

    # Build model
    print(f"Loading model ({args.model})...")
    t0 = time.time()
    sam = build_model(args.model, args.weights)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Init predictor and preload features
    pred = SAM2VideoPredictor(sam)
    print("Preloading features for all frames...")
    t0 = time.time()
    pred.init_state(
        video, preload_features=True,
        progress_callback=lambda d, t: print(f"  {d}/{t}", end="\r"),
    )
    print(f"\n  Features preloaded in {time.time() - t0:.1f}s")

    # Add click on frame 0
    pred.add_new_points(
        0, 1,
        np.array([list(click)], dtype=np.float32),
        np.array([1], dtype=np.int32),
    )

    # Propagate and write output
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_path = out_dir / "tracked_overlay.mp4"
    mask_path = out_dir / "tracked_mask.mp4"
    over_w = cv2.VideoWriter(str(overlay_path), fourcc, fps, (W, H), True)
    mask_w = cv2.VideoWriter(str(mask_path), fourcc, fps, (W, H), False)

    print("Tracking...")
    t0 = time.time()
    for f, ids, masks, ious in pred.propagate_in_video(start_frame=0, max_frames=N):
        logits = np.asarray(masks[0])   # [3, H, W]
        ious_arr = np.asarray(ious[0])  # [3]
        best = int(np.argmax(ious_arr))
        sigm = 1.0 / (1.0 + np.exp(-np.clip(logits[best], -30, 30)))
        binary = (sigm > 0.5).astype(np.uint8)

        # Overlay
        bgr = cv2.cvtColor(video[f], cv2.COLOR_RGB2BGR).copy()
        green = bgr.copy()
        green[binary > 0] = (0, 255, 0)
        ov = cv2.addWeighted(bgr, 0.55, green, 0.45, 0)
        if f == 0:
            cv2.circle(ov, click, 6, (0, 0, 255), -1)
        over_w.write(ov)
        mask_w.write(binary * 255)

    over_w.release()
    mask_w.release()
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({N / elapsed:.1f} fps)")
    print(f"  Output: {overlay_path}")
    print(f"  Mask:   {mask_path}")


if __name__ == "__main__":
    main()

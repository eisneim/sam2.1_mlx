"""SAM2.1 Image Segmentation — click on an object, get its mask.

Usage:
    python inference_image.py --image path/to/image.jpg
    python inference_image.py --image path/to/image.jpg --model small --weights weights/sam2.1_hiera_small.safetensors
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from sam2.build import build_model
from sam2.predictor import SAM2ImagePredictor


def pick_point(image_bgr: np.ndarray) -> tuple:
    """Show image in a window, let user click on the object. Returns (x, y)."""
    click_pos = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos.clear()
            click_pos.extend([x, y])

    win = "Click on the object (then press any key)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)
    cv2.setMouseCallback(win, on_click)

    while True:
        vis = image_bgr.copy()
        if click_pos:
            cv2.circle(vis, (click_pos[0], click_pos[1]), 8, (0, 0, 255), -1)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            return None
        if click_pos and key != 255:
            break

    cv2.destroyAllWindows()
    return click_pos[0], click_pos[1]


def main():
    parser = argparse.ArgumentParser(description="SAM2.1 Image Segmentation")
    parser.add_argument("-i", "--image", required=True, help="Input image path")
    parser.add_argument("--output", default="output", help="Output directory (default: output/)")
    parser.add_argument("--model", default="base_plus", choices=["small", "base_plus", "large"],
                        help="Model size (default: base_plus)")
    parser.add_argument("--weights", default=None,
                        help="Path to .safetensors weights (default: weights/sam2.1_hiera_{model}.safetensors)")
    args = parser.parse_args()

    # Load image
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(f"Error: cannot read image: {args.image}")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]
    print(f"Image: {W}x{H}")

    # Let user click
    click = pick_point(image_bgr)
    if click is None:
        print("No click — exiting.")
        return
    print(f"Click at: {click}")

    # Build model
    print(f"Loading model ({args.model})...")
    t0 = time.time()
    sam = build_model(args.model, args.weights)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Predict
    pred = SAM2ImagePredictor(sam)
    t0 = time.time()
    pred.set_image(image_rgb)
    masks, iou, _ = pred.predict(
        point_coords=np.array([click], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
    )
    elapsed = time.time() - t0
    print(f"  Prediction in {elapsed:.2f}s, IoU scores: {iou}")

    # Pick best mask
    best = int(np.argmax(iou))
    sigm = 1.0 / (1.0 + np.exp(-np.clip(masks[best], -30, 30)))
    binary = (sigm > 0.5).astype(np.uint8)

    # Save outputs
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    # Overlay
    green = image_bgr.copy()
    green[binary > 0] = (0, 255, 0)
    overlay = cv2.addWeighted(image_bgr, 0.55, green, 0.45, 0)
    cv2.circle(overlay, click, 6, (0, 0, 255), -1)
    overlay_path = out_dir / f"{stem}_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    # Mask
    mask_path = out_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), binary * 255)

    print(f"  Output: {overlay_path}")
    print(f"  Mask:   {mask_path}")


if __name__ == "__main__":
    main()

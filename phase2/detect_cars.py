"""Stage 1 of the app: find every car in a photo with a pretrained AI model.

No training needed. The model already knows 80 everyday object types
(the COCO dataset), including "car" and "truck". This script:
  1. runs the detector on one photo or a whole folder
  2. draws the boxes it found
  3. saves a crop of each car, ready for the stage-2 classifier you'll train later

Usage:
    python detect_cars.py car.jpg
    python detect_cars.py ../data/raw_photos --save-crops
Outputs go to output/detections/ (boxes drawn) and output/crops/ (one image per car).

The first run downloads the model weights (a few MB) automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

# COCO class IDs the detector uses. Pickups and SUVs are often labeled "truck".
VEHICLE_CLASSES = {2: "car", 7: "truck"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_rgb(path: Path) -> np.ndarray:
    """Same loader as image_basics.py: fix phone rotation, force RGB."""
    with Image.open(path) as img:
        return np.asarray(ImageOps.exif_transpose(img).convert("RGB")).copy()


def find_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def padded_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int, pad: float) -> tuple[int, int, int, int]:
    """Grow the box by `pad` on each side (cars are often cut tight) and keep it inside the image."""
    bw, bh = x2 - x1, y2 - y1
    return (
        max(0, int(x1 - bw * pad)),
        max(0, int(y1 - bh * pad)),
        min(w, int(x2 + bw * pad)),
        min(h, int(y2 + bh * pad)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and crop cars with a pretrained model.")
    parser.add_argument("source", type=Path, help="an image or a folder of images")
    parser.add_argument("--model", default="yolo26n.pt", help="pretrained detector weights")
    parser.add_argument("--conf", type=float, default=0.5, help="minimum confidence (0-1)")
    parser.add_argument("--min-size", type=int, default=150, help="ignore cars smaller than this (px, shortest side)")
    parser.add_argument("--save-crops", action="store_true", help="save one image per detected car")
    args = parser.parse_args()

    images = find_images(args.source)
    if not images:
        raise SystemExit(f"No images found at {args.source}")

    model = YOLO(args.model)
    det_dir, crop_dir = Path("output/detections"), Path("output/crops")
    det_dir.mkdir(parents=True, exist_ok=True)
    if args.save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for path in images:
        img = load_rgb(path)
        h, w = img.shape[:2]

        # One call runs the whole neural network on the photo.
        # Ultralytics expects NumPy images in BGR order (the OpenCV convention), while
        # Pillow gives RGB. img[..., ::-1] reverses the channel axis: RGB -> BGR.
        # Pass RGB by mistake and the model sees blue skin and orange skies.
        bgr = img[..., ::-1]
        result = model.predict(bgr, conf=args.conf, classes=list(VEHICLE_CLASSES), verbose=False)[0]

        boxes = result.boxes
        kept = 0
        for i, (xyxy, conf, cls) in enumerate(zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())):
            x1, y1, x2, y2 = xyxy
            if min(x2 - x1, y2 - y1) < args.min_size:
                continue  # too small/far away to identify the model reliably
            kept += 1
            print(f"{path.name}: {VEHICLE_CLASSES[int(cls)]} {conf:.0%} at "
                  f"(x1={x1:.0f}, y1={y1:.0f}, x2={x2:.0f}, y2={y2:.0f})")
            if args.save_crops:
                px1, py1, px2, py2 = padded_box(x1, y1, x2, y2, w, h, pad=0.05)
                crop = img[py1:py2, px1:px2]  # the same [y, x] slicing from image_basics.py
                Image.fromarray(crop).save(crop_dir / f"{path.stem}_car{i}.jpg", quality=95)

        if kept == 0:
            print(f"{path.name}: no cars found (try a lower --conf)")
        total += kept

        # result.plot() returns the image with boxes drawn, in BGR color order
        Image.fromarray(result.plot()[..., ::-1]).save(det_dir / f"{path.stem}_detected.jpg")

    print(f"\n{total} car(s) in {len(images)} image(s). Boxes drawn: {det_dir}/")
    if args.save_crops:
        print(f"Crops ready for the classifier: {crop_dir}/")


if __name__ == "__main__":
    main()

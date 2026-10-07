"""Ask your trained classifier what car is in a photo.

Shows the top 3 guesses with confidence, and says "not sure" when the best
guess is weak. A classifier ALWAYS picks one of the classes it knows, even for
a car it has never seen, so a confidence threshold is your only defense against
confidently wrong answers.

Usage:
    python predict.py runs/first_classifier/weights/best.pt ../phase2/output/crops
    python predict.py runs/first_classifier/weights/best.pt some_car.jpg --threshold 0.6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(ImageOps.exif_transpose(img).convert("RGB")).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify car photos with a trained model.")
    parser.add_argument("weights", type=Path, help="path to best.pt")
    parser.add_argument("source", type=Path, help="an image or a folder of images (ideally car crops)")
    parser.add_argument("--threshold", type=float, default=0.5, help="below this, answer 'not sure'")
    args = parser.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")
    images = [args.source] if args.source.is_file() else sorted(
        p for p in args.source.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise SystemExit(f"No images found at {args.source}")

    model = YOLO(args.weights)
    for path in images:
        bgr = load_rgb(path)[..., ::-1]  # Ultralytics expects BGR, same as the detector
        probs = model.predict(bgr, verbose=False)[0].probs
        top = list(zip(probs.top5[:3], probs.top5conf.tolist()[:3]))

        best_name, best_conf = model.names[top[0][0]], top[0][1]
        verdict = best_name if best_conf >= args.threshold else f"not sure (best guess: {best_name})"
        print(f"\n{path.name}: {verdict}")
        for cls_id, conf in top:
            bar = "#" * round(conf * 30)
            print(f"   {conf:6.1%}  {bar:<30} {model.names[cls_id]}")


if __name__ == "__main__":
    main()

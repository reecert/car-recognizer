"""Train your first car classifier (stage 2 of the app).

Starts from a classifier pretrained on ImageNet (1.2M photos, 1,000 everyday classes)
and fine-tunes it on your car classes. That's transfer learning: the model already
knows edges, wheels, and glass; it only has to learn what separates a Camry from
a Corolla.

Usage:
    python train.py
    python train.py --data ../data/stanford_subset --epochs 30 --imgsz 320
Results (curves, confusion matrix, best weights) land in runs/<name>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"  # your M4 Pro's GPU
    if torch.cuda.is_available():
        return "0"    # an NVIDIA GPU (e.g. Colab)
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained classifier on car photos.")
    parser.add_argument("--data", type=Path, default=Path("../data/stanford_subset"),
                        help="folder containing train/ and val/ class folders")
    parser.add_argument("--model", default="yolo26n-cls.pt", help="pretrained classifier to start from")
    parser.add_argument("--epochs", type=int, default=30, help="full passes over the training set")
    parser.add_argument("--imgsz", type=int, default=224, help="input size; bigger keeps finer detail")
    parser.add_argument("--batch", type=int, default=32, help="images per training step")
    parser.add_argument("--name", default="first_classifier", help="run name under runs/")
    parser.add_argument("--device", default=None, help="mps, cpu, or a GPU index (auto if omitted)")
    args = parser.parse_args()

    for split in ("train", "val"):
        if not (args.data / split).is_dir():
            raise SystemExit(f"Missing {args.data / split}. Run prepare_stanford.py first.")

    device = args.device or pick_device()
    print(f"Training on: {device}")

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(Path("runs").resolve()),
        name=args.name,
        exist_ok=False,   # each run gets its own folder (first_classifier, first_classifier2, ...)
        patience=10,      # stop early if val accuracy hasn't improved for 10 epochs
        seed=0,           # reproducible shuffling and augmentation
        flipud=0.0,       # never flip upside down: cars aren't upside down
        fliplr=0.5,       # mirror half the images: a mirrored Camry is still a Camry
        plots=True,       # save training curves and the confusion matrix
    )

    # Evaluate the best checkpoint on the validation set
    run_dir = Path(model.trainer.save_dir)
    best = YOLO(run_dir / "weights" / "best.pt")
    metrics = best.val(data=str(args.data.resolve()), imgsz=args.imgsz, device=device, split="val", plots=False)
    print(f"\nTop-1 accuracy: {metrics.top1:.1%}   (first guess correct)")
    print(f"Top-5 accuracy: {metrics.top5:.1%}   (right answer among the top 5 guesses)")
    print(f"\nEverything for this run: {run_dir}/")
    print(f"  results.png           loss and accuracy per epoch")
    print(f"  confusion_matrix*.png which cars get mixed up")
    print(f"  weights/best.pt       your trained model")


if __name__ == "__main__":
    main()

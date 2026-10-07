"""Phase 6: the honest final score, on photos the model never influenced.

During development you pick epochs and settings by looking at val accuracy, so
val accuracy ends up slightly flattering. The test split exists to fix that:
you evaluate on it ONCE, at the end, and that number is the one you report.

Besides overall accuracy, this reports:
  - per-class accuracy        (which cars are weak)
  - the most common mix-ups   (what they get confused with)
  - a confidence table        (how to set the app's "not sure" threshold)

Usage:
    python evaluate.py ../phase4/runs/my_cars/best.pt --data ../data/dataset
    python evaluate.py ../phase4/runs/pytorch_resnet18/best.pt --data ../data/stanford_subset --split val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase4"))
from train_pytorch import build_model, build_transforms, pick_device, plot_confusion  # noqa: E402

THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]


@torch.no_grad()
def predict_all(model, dataset, device, batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Softmax probabilities for every image, plus the true labels."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False)
    probs, labels = [], []
    for x, y in loader:
        probs.append(model(x.to(device)).softmax(1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier on a held-out split.")
    parser.add_argument("checkpoint", type=Path, help="best.pt from phase4/train_pytorch.py")
    parser.add_argument("--data", type=Path, default=Path("../data/dataset"))
    parser.add_argument("--split", default="test", help="test (default, the honest number) or val")
    args = parser.parse_args()

    split_dir = args.data / args.split
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not split_dir.is_dir():
        raise SystemExit(f"Missing {split_dir}. Run phase1/organize.py with --test-fraction > 0.")

    device = pick_device()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    classes = ckpt["classes"]
    model = build_model(ckpt["arch"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    _, val_tf = build_transforms(ckpt["imgsz"])
    ds = datasets.ImageFolder(split_dir, val_tf)
    if ds.classes != classes:
        raise SystemExit(f"Class folders in {split_dir} don't match the checkpoint.\n"
                         f"  checkpoint: {classes}\n  {args.split}: {ds.classes}")

    probs, labels = predict_all(model, ds, device)
    preds = probs.argmax(1)
    conf = probs.max(1)
    k = min(5, len(classes))
    top5 = (np.argsort(-probs, 1)[:, :k] == labels[:, None]).any(1)

    print(f"{args.split.upper()} set: {len(labels)} images, {len(classes)} classes "
          f"| model {ckpt['arch']} (val top-1 during training: {ckpt['val_top1']:.1%})")
    print(f"\nTop-1 accuracy: {(preds == labels).mean():.1%}")
    print(f"Top-5 accuracy: {top5.mean():.1%}")
    if args.split == "test":
        gap = ckpt["val_top1"] - (preds == labels).mean()
        print(f"Val minus test: {gap:+.1%}  (a large positive gap means val was flattering)")

    # Per-class accuracy, weakest first
    print(f"\n{'class':<45}{'images':>7}{'accuracy':>10}")
    rows = []
    for c, name in enumerate(classes):
        m = labels == c
        if m.any():
            rows.append(((preds[m] == c).mean(), name, int(m.sum())))
    for acc, name, n in sorted(rows):
        print(f"{name:<45}{n:>7}{acc:>10.1%}")

    # Most common mix-ups
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(confusion, (labels, preds), 1)
    errors = sorted(((confusion[i, j], classes[i], classes[j])
                     for i in range(len(classes)) for j in range(len(classes))
                     if i != j and confusion[i, j]), reverse=True)
    if errors:
        print("\nMost common mix-ups (true -> predicted):")
        for count, t, p in errors[:5]:
            print(f"  {count:3d}x  {t}  ->  {p}")

    # Confidence threshold table: the app answers only when confident enough
    print("\nIf the app says 'not sure' below a confidence threshold:")
    print(f"{'threshold':>10}{'answered':>10}{'accuracy when answered':>25}")
    for t in THRESHOLDS:
        answered = conf >= t
        acc = (preds[answered] == labels[answered]).mean() if answered.any() else float("nan")
        print(f"{t:>10.1f}{answered.mean():>10.1%}{acc:>25.1%}")
    print("Pick the threshold where accuracy is high enough for you without answering too rarely.")

    out = args.checkpoint.parent / f"confusion_{args.split}.png"
    plot_confusion(confusion, classes, out)
    print(f"\nConfusion matrix: {out}")


if __name__ == "__main__":
    main()

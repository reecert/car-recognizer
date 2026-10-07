"""Phase 5 experiment: how much do pretrained features already know about cars?

Two ways to use a pretrained network:
  - Linear probe  (this script): FREEZE every pretrained layer and train only the
                   final Linear layer. Fast, needs little data, but can only reuse
                   what ImageNet taught the network.
  - Fine-tuning   (Phase 4's train_pytorch.py): train ALL layers, so the features
                   themselves adapt to cars. Usually more accurate, needs more data.

Comparing the two tells you how "car-ready" the ImageNet features are, and how much
fine-tuning adds. That gap is a great interview talking point.

Usage:
    python linear_probe.py
    python linear_probe.py --arch efficientnet_b0 --epochs 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase4"))
from train_pytorch import build_loaders, build_model, evaluate, pick_device, set_seed  # noqa: E402
from inspect_model import head_layer  # noqa: E402  (same folder)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train only the final layer of a frozen pretrained network.")
    parser.add_argument("--data", type=Path, default=Path("../data/stanford_subset"))
    parser.add_argument("--arch", default="resnet18", choices=["resnet18", "efficientnet_b0", "convnext_tiny"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scratch", action="store_true", help="(for testing only) random frozen features")
    args = parser.parse_args()

    for split in ("train", "val"):
        if not (args.data / split).is_dir():
            raise SystemExit(f"Missing {args.data / split}. Run ../phase3/prepare_stanford.py first.")

    set_seed(args.seed)
    device = pick_device()
    train_dl, val_dl, classes = build_loaders(args.data, args.imgsz, args.batch, args.workers)
    model = build_model(args.arch, len(classes), pretrained=not args.scratch).to(device)

    # Freeze everything, then unfreeze only the new head
    for p in model.parameters():
        p.requires_grad = False
    head = head_layer(model, args.arch)
    for p in head.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | {args.arch} | training {trainable:,} of {total:,} weights "
          f"({trainable / total:.2%}); everything else is frozen")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.05)
    best = 0.0

    print(f"\n{'epoch':>5} {'train_loss':>10} {'val_top1':>8} {'val_top5':>8} {'time':>6}")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        # Whole model in eval mode: frozen layers must not update their batch-norm statistics.
        # Only the head learns, and a Linear layer behaves the same in train and eval mode.
        model.eval()
        total_loss, seen = 0.0, 0
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            loss = loss_fn(model(images), labels)
            optimizer.zero_grad()
            loss.backward()          # gradients only flow into the head's weights
            optimizer.step()
            total_loss += loss.item() * len(labels)
            seen += len(labels)

        _, top1, top5, _ = evaluate(model, val_dl, loss_fn, device, len(classes))
        best = max(best, top1)
        print(f"{epoch:>5} {total_loss / seen:>10.3f} {top1:>8.1%} {top5:>8.1%} {time.time() - start:>5.0f}s")

    print(f"\nLinear probe best val top-1: {best:.1%}")
    print("Compare with Phase 4's fine-tuned run of the same architecture: the gap is what fine-tuning adds.")


if __name__ == "__main__":
    main()

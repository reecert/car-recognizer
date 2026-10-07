"""Phase 4: train a car classifier in plain PyTorch, with every step visible.

This does what `yolo classify train` did in Phase 3, but nothing is hidden:
data loading, augmentation, the model, the loss, the optimizer, the training
loop, the validation loop, checkpoints, and the plots are all right here.

Usage:
    python train_pytorch.py                                   (pretrained ResNet-18)
    python train_pytorch.py --arch efficientnet_b0 --imgsz 320
    python train_pytorch.py --scratch --name scratch          (no pretrained weights: the ablation)

Outputs go to runs/<name>/: best.pt, results.csv, curves.png, confusion_matrix.png
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# Pretrained torchvision models expect inputs normalized with ImageNet statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Same seed -> same shuffling, augmentation and starting weights."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_transforms(imgsz: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Augmentation for training; plain, repeatable preprocessing for validation."""
    train_tf = transforms.Compose([
        # Random zoom/crop: the model sees cars at slightly different sizes and positions
        transforms.RandomResizedCrop(imgsz, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),        # a mirrored Camry is still a Camry
        transforms.ColorJitter(0.2, 0.2, 0.2),    # different light and paint shades
        transforms.ToTensor(),                    # (H, W, C) 0..255 -> (C, H, W) 0..1
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    # No randomness at validation time: the same image must always get the same score
    val_tf = transforms.Compose([
        transforms.Resize(int(imgsz * 1.14)),     # shorter side to ~1.14x, then...
        transforms.CenterCrop(imgsz),             # ...the central square
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def build_loaders(data: Path, imgsz: int, batch: int, workers: int):
    train_tf, val_tf = build_transforms(imgsz)
    # ImageFolder: one subfolder per class; the folder name is the label
    train_ds = datasets.ImageFolder(data / "train", train_tf)
    val_ds = datasets.ImageFolder(data / "val", val_tf)
    if train_ds.classes != val_ds.classes:
        raise SystemExit(f"train and val have different class folders:\n{train_ds.classes}\n{val_ds.classes}")

    common = dict(batch_size=batch, num_workers=workers, persistent_workers=workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) > batch, **common)
    val_dl = DataLoader(val_ds, shuffle=False, **common)
    return train_dl, val_dl, train_ds.classes


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    """Load a standard CNN and replace its last layer with one output per car class.

    The pretrained last layer predicts ImageNet's 1,000 classes. We swap it for a
    new, randomly initialized layer with `num_classes` outputs; everything before
    it keeps what it learned from ImageNet.
    """
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    else:
        raise SystemExit(f"unknown --arch {arch}")
    return m


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, loss_fn, optimizer, device) -> tuple[float, float]:
    model.train()  # turns on training behavior (dropout, batch-norm statistics updates)
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)              # 1. forward: one score per class, shape (N, num_classes)
        loss = loss_fn(logits, labels)      # 2. loss: how wrong were the scores?
        optimizer.zero_grad()               # 5. clear last step's gradients
        loss.backward()                     # 3. backward: gradient for every weight
        optimizer.step()                    # 4. step: nudge every weight downhill

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen


@torch.no_grad()  # no gradients needed: faster, less memory
def evaluate(model, loader, loss_fn, device, num_classes: int):
    model.eval()  # turns off training behavior, so results are repeatable
    total_loss, top1, top5, seen = 0.0, 0, 0, 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    k = min(5, num_classes)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += loss_fn(logits, labels).item() * len(labels)

        topk = logits.topk(k, dim=1).indices                    # (N, k) best class indices
        top1 += (topk[:, 0] == labels).sum().item()
        top5 += (topk == labels.unsqueeze(1)).any(1).sum().item()
        seen += len(labels)
        np.add.at(confusion, (labels.cpu().numpy(), topk[:, 0].cpu().numpy()), 1)
    return total_loss / seen, top1 / seen, top5 / seen, confusion


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_curves(history: list[dict], path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train loss")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val loss")
    ax1.set_xlabel("epoch"); ax1.set_title("Loss (lower is better)"); ax1.legend()
    ax2.plot(epochs, [h["train_acc"] for h in history], label="train accuracy")
    ax2.plot(epochs, [h["val_top1"] for h in history], label="val top-1")
    ax2.plot(epochs, [h["val_top5"] for h in history], label="val top-5", linestyle="--")
    ax2.set_xlabel("epoch"); ax2.set_ylim(0, 1); ax2.set_title("Accuracy (higher is better)"); ax2.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_confusion(confusion: np.ndarray, classes: list[str], path: Path) -> None:
    # Normalize each row: "of all true Camrys, what fraction went to each prediction?"
    rows = confusion.sum(1, keepdims=True)
    norm = np.divide(confusion, rows, out=np.zeros(confusion.shape, dtype=float), where=rows > 0)
    size = max(6, 0.6 * len(classes) + 3)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if confusion[i, j]:
                ax.text(j, i, confusion[i, j], ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_title("Confusion matrix (color = row fraction)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a car classifier in plain PyTorch.")
    parser.add_argument("--data", type=Path, default=Path("../data/stanford_subset"))
    parser.add_argument("--arch", default="resnet18", choices=["resnet18", "efficientnet_b0", "convnext_tiny"])
    parser.add_argument("--scratch", action="store_true", help="random starting weights (no pretraining)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None, help="default: 3e-4 pretrained, 1e-3 from scratch")
    parser.add_argument("--patience", type=int, default=8, help="stop after this many epochs without improvement")
    parser.add_argument("--workers", type=int, default=4, help="parallel image-loading processes (0 = none)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="pytorch_resnet18")
    args = parser.parse_args()

    for split in ("train", "val"):
        if not (args.data / split).is_dir():
            raise SystemExit(f"Missing {args.data / split}. Run ../phase3/prepare_stanford.py first.")

    set_seed(args.seed)
    device = pick_device()
    run_dir = Path("runs") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl, classes = build_loaders(args.data, args.imgsz, args.batch, args.workers)
    print(f"Device: {device} | classes: {len(classes)} | "
          f"train images: {len(train_dl.dataset)} | val images: {len(val_dl.dataset)}")

    pretrained = not args.scratch
    model = build_model(args.arch, len(classes), pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.arch} ({n_params / 1e6:.1f}M weights), "
          f"{'pretrained on ImageNet' if pretrained else 'random start (from scratch)'}")

    lr = args.lr or (3e-4 if pretrained else 1e-3)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)  # smoothing: don't reward 100% overconfidence
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)  # lr glides to ~0

    history: list[dict] = []
    best_top1, best_epoch = -1.0, 0
    print(f"\n{'epoch':>5} {'train_loss':>10} {'train_acc':>9} {'val_loss':>8} {'val_top1':>8} {'val_top5':>8} {'time':>6}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_dl, loss_fn, optimizer, device)
        val_loss, val_top1, val_top5, _ = evaluate(model, val_dl, loss_fn, device, len(classes))
        scheduler.step()

        row = dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                   val_loss=val_loss, val_top1=val_top1, val_top5=val_top5, lr=optimizer.param_groups[0]["lr"])
        history.append(row)
        marker = ""
        if val_top1 > best_top1:  # keep the best model, not the last one
            best_top1, best_epoch = val_top1, epoch
            torch.save({"model": model.state_dict(), "arch": args.arch, "classes": classes,
                        "imgsz": args.imgsz, "epoch": epoch, "val_top1": val_top1}, run_dir / "best.pt")
            marker = "  * best"
        print(f"{epoch:>5} {train_loss:>10.3f} {train_acc:>9.1%} {val_loss:>8.3f} "
              f"{val_top1:>8.1%} {val_top5:>8.1%} {time.time() - start:>5.0f}s{marker}")

        if epoch - best_epoch >= args.patience:
            print(f"No improvement for {args.patience} epochs: stopping early.")
            break

    with open(run_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_curves(history, run_dir / "curves.png")

    # Final report on the BEST checkpoint
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    _, top1, top5, confusion = evaluate(model, val_dl, loss_fn, device, len(classes))
    plot_confusion(confusion, classes, run_dir / "confusion_matrix.png")

    print(f"\nBest epoch: {best_epoch}  |  val top-1 {top1:.1%}  |  val top-5 {top5:.1%}")
    errors = [(confusion[i, j], classes[i], classes[j])
              for i in range(len(classes)) for j in range(len(classes)) if i != j and confusion[i, j]]
    if errors:
        print("Most common mix-ups (true -> predicted):")
        for count, true_cls, pred_cls in sorted(errors, reverse=True)[:5]:
            print(f"  {count:3d}x  {true_cls}  ->  {pred_cls}")
    print(f"\nSaved to {run_dir}/: best.pt, results.csv, curves.png, confusion_matrix.png")


if __name__ == "__main__":  # required on macOS: DataLoader workers re-import this file
    main()

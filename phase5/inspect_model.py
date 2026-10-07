"""Phase 5: look inside your trained car classifier.

Three views of what the network learned:
  1. First-layer filters   - the tiny patterns it looks for first (edges, color blobs)
  2. Grad-CAM heatmaps     - WHERE in each photo it looked to make its decision
  3. Feature space map     - how it arranges cars internally; overlapping clusters = confusion

Usage:
    python inspect_model.py ../phase4/runs/pytorch_resnet18/best.pt
    python inspect_model.py ../phase4/runs/scratch/best.pt        (compare with from-scratch!)
Outputs go to an inspect/ folder next to the checkpoint.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

# Reuse the exact model and preprocessing from Phase 4, so nothing can drift apart
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase4"))
from train_pytorch import IMAGENET_MEAN, IMAGENET_STD, build_model, build_transforms, pick_device  # noqa: E402


# ---------------------------------------------------------------------------
# Model plumbing: which layer is which, per architecture
# ---------------------------------------------------------------------------

def last_conv_block(model: nn.Module, arch: str) -> nn.Module:
    """The final convolutional stage: its feature maps still have spatial layout (where things are)."""
    return model.layer4 if arch == "resnet18" else model.features[-1]


def head_layer(model: nn.Module, arch: str) -> nn.Linear:
    """The final Linear layer that turns features into one score per class."""
    if arch == "resnet18":
        return model.fc
    if arch == "efficientnet_b0":
        return model.classifier[1]
    return model.classifier[2]  # convnext_tiny


def first_conv(model: nn.Module) -> nn.Conv2d:
    return next(m for m in model.modules() if isinstance(m, nn.Conv2d) and m.in_channels == 3)


def denormalize(t: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization: (C, H, W) tensor -> (H, W, C) 0..1 array for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# 1. First-layer filters
# ---------------------------------------------------------------------------

def plot_filters(model: nn.Module, path: Path) -> None:
    w = first_conv(model).weight.detach().cpu()           # (out_channels, 3, k, k)
    n = min(len(w), 64)
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    for i, ax in enumerate(np.atleast_1d(axes).flat):
        ax.axis("off")
        if i < n:
            f = w[i]
            f = (f - f.min()) / (f.max() - f.min() + 1e-8)   # stretch each filter to 0..1 to see it
            ax.imshow(f.permute(1, 2, 0).numpy(), interpolation="nearest")
    fig.suptitle(f"First-layer filters ({tuple(w.shape[2:])} pixels each)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Grad-CAM
# ---------------------------------------------------------------------------

def grad_cam(model: nn.Module, block: nn.Module, x: torch.Tensor, class_idx: int | None = None):
    """Heatmap of which regions pushed the score of `class_idx` up.

    Recipe: take the last conv block's feature maps (each one "lights up" for some
    pattern), weight each map by how much the class score depends on it (the average
    gradient), add them up, and keep only the positive evidence.
    """
    store: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        store["acts"] = output
        output.register_hook(lambda grad: store.__setitem__("grads", grad))

    handle = block.register_forward_hook(forward_hook)
    try:
        model.zero_grad()
        logits = model(x)                                  # x: (1, 3, H, W)
        probs = logits.softmax(1)[0]
        idx = int(probs.argmax()) if class_idx is None else class_idx
        logits[0, idx].backward()
    finally:
        handle.remove()

    acts, grads = store["acts"][0], store["grads"][0]      # (C, h, w) each
    weights = grads.mean(dim=(1, 2), keepdim=True)         # importance of each feature map
    cam = F.relu((weights * acts).sum(0))                  # (h, w), positive evidence only
    cam = F.interpolate(cam[None, None], size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy(), idx, probs[idx].item()


def plot_grad_cams(model, arch, dataset, classes, device, n: int, seed: int, path: Path) -> None:
    rng = random.Random(seed)
    picks = rng.sample(range(len(dataset)), min(n, len(dataset)))
    block = last_conv_block(model, arch)

    fig, axes = plt.subplots(2, len(picks), figsize=(3 * len(picks), 6.4), squeeze=False)
    for col, i in enumerate(picks):
        x, label = dataset[i]
        cam, pred, conf = grad_cam(model, block, x.unsqueeze(0).to(device))
        img = denormalize(x)
        ok = pred == label
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"true: {classes[label]}", fontsize=7)
        axes[1, col].imshow(img)
        axes[1, col].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, col].set_title(f"{'OK' if ok else 'WRONG'} pred: {classes[pred]} {conf:.0%}",
                               fontsize=7, color="green" if ok else "red")
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    fig.suptitle("Grad-CAM: red = regions that drove the prediction")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.25)  # room for the prediction titles between rows
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Feature space
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model, arch, dataset, device, batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """The vector the head layer receives: the model's internal 'description' of each car."""
    feats: list[torch.Tensor] = []
    handle = head_layer(model, arch).register_forward_hook(lambda _m, inp, _out: feats.append(inp[0].cpu()))
    labels: list[int] = []
    try:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False)
        for x, y in loader:
            model(x.to(device))
            labels += y.tolist()
    finally:
        handle.remove()
    return torch.cat(feats).numpy(), np.array(labels)


def project_2d(feats: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    """t-SNE if scikit-learn is installed (best clusters), otherwise PCA with plain NumPy."""
    try:
        from sklearn.manifold import TSNE
        perplexity = max(2, min(30, (len(feats) - 1) // 3))
        return TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca").fit_transform(feats), "t-SNE"
    except ImportError:
        centered = feats - feats.mean(0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ vt[:2].T, "PCA (pip install scikit-learn for t-SNE)"


def plot_feature_space(feats, labels, classes, seed: int, path: Path) -> None:
    pts, method = project_2d(feats, seed)
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("tab10" if len(classes) <= 10 else "tab20")
    for c in range(len(classes)):
        m = labels == c
        ax.scatter(pts[m, 0], pts[m, 1], s=14, color=cmap(c % cmap.N), label=classes[c], alpha=0.8)
    ax.legend(fontsize=7, markerscale=1.5, loc="best")
    ax.set_title(f"Feature space ({method}): each dot is one validation photo")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def closest_class_pairs(feats, labels, classes, top: int = 5) -> list[tuple[float, str, str]]:
    """Cosine similarity between class centroids: the pairs the model finds most alike."""
    present = [c for c in range(len(classes)) if (labels == c).any()]
    cents = np.stack([feats[labels == c].mean(0) for c in present])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-8
    sim = cents @ cents.T
    pairs = [(float(sim[i, j]), classes[present[i]], classes[present[j]])
             for i in range(len(present)) for j in range(i + 1, len(present))]
    return sorted(pairs, reverse=True)[:top]


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize what a trained classifier learned.")
    parser.add_argument("checkpoint", type=Path, help="best.pt from phase4")
    parser.add_argument("--data", type=Path, default=Path("../data/stanford_subset"))
    parser.add_argument("--n-cam", type=int, default=8, help="how many photos get Grad-CAM heatmaps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}\nTrain one first: cd ../phase4 && python train_pytorch.py")
    if not (args.data / "val").is_dir():
        raise SystemExit(f"Missing {args.data / 'val'}. Run ../phase3/prepare_stanford.py first.")

    device = pick_device()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    arch, classes, imgsz = ckpt["arch"], ckpt["classes"], ckpt["imgsz"]

    model = build_model(arch, len(classes), pretrained=False)   # architecture only...
    model.load_state_dict(ckpt["model"])                          # ...then YOUR trained weights
    model.to(device).eval()

    _, val_tf = build_transforms(imgsz)
    val_ds = datasets.ImageFolder(args.data / "val", val_tf)
    if val_ds.classes != classes:
        raise SystemExit("The val folder's classes don't match the checkpoint's classes.")

    out = args.checkpoint.parent / "inspect"
    out.mkdir(exist_ok=True)
    print(f"Model: {arch} | trained to epoch {ckpt['epoch']} | val top-1 {ckpt['val_top1']:.1%} | device {device}")

    plot_filters(model, out / "filters.png")
    print(f"1. first-layer filters      -> {out / 'filters.png'}")

    plot_grad_cams(model, arch, val_ds, classes, device, args.n_cam, args.seed, out / "grad_cam.png")
    print(f"2. Grad-CAM heatmaps        -> {out / 'grad_cam.png'}")

    feats, labels = extract_features(model, arch, val_ds, device)
    plot_feature_space(feats, labels, classes, args.seed, out / "feature_space.png")
    print(f"3. feature space map        -> {out / 'feature_space.png'}  ({feats.shape[1]} numbers per photo)")

    print("\nClass pairs the model sees as most alike (cosine similarity of their average features):")
    for sim, a, b in closest_class_pairs(feats, labels, classes):
        print(f"  {sim:.2f}  {a}  <->  {b}")


if __name__ == "__main__":
    main()

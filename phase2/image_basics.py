"""Phase 2: images are just NumPy arrays.

Walks through everything a vision model does to a photo before it sees it:
load -> fix rotation -> inspect -> crop -> resize -> flip -> normalize -> reorder for PyTorch.

Usage:
    python image_basics.py path/to/car.jpg
Saves a picture of every step to output/overview.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

# Mean and standard deviation of ImageNet photos, per RGB channel.
# Pretrained models expect inputs normalized with exactly these numbers.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def load_rgb(path: Path) -> np.ndarray:
    """Load a photo as an (H, W, 3) uint8 array, rotated the way it looked on the phone.

    Phones save photos sideways plus an EXIF "orientation" tag that says how to
    rotate them. Most viewers obey it, but PIL does not unless you call
    exif_transpose. Skip this and some of your training photos are sideways.
    """
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")  # PNGs may have a 4th alpha channel; grayscale has 1
        return np.asarray(img).copy()


def crop(arr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Crop with a box in (x1, y1, x2, y2) form — the form detectors output.

    Arrays are indexed [row, column] = [y, x], so y comes FIRST in the slice.
    Mixing this up is the most common image bug there is.
    """
    return arr[y1:y2, x1:x2]


def resize_keep_aspect(arr: np.ndarray, size: int) -> np.ndarray:
    """Fit inside size x size without stretching, then pad to a square ("letterbox").

    Plain resizing to a square would squash a long pickup truck into a short one.
    """
    img = Image.fromarray(arr)
    img.thumbnail((size, size), Image.Resampling.BILINEAR)  # shrinks, keeps aspect ratio
    canvas = Image.new("RGB", (size, size), (114, 114, 114))  # neutral gray padding
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return np.asarray(canvas).copy()


def normalize(arr: np.ndarray) -> np.ndarray:
    """uint8 0..255 -> float32, then ImageNet mean/std per channel.

    Broadcasting: arr is (H, W, 3) and IMAGENET_MEAN is (3,). NumPy lines up the
    LAST dimensions, so the 3 means are applied to the 3 channels of every pixel.
    """
    x = arr.astype(np.float32) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def to_model_input(arr: np.ndarray) -> np.ndarray:
    """(H, W, C) -> (1, C, H, W): PyTorch wants channels first, plus a batch dimension."""
    chw = arr.transpose(2, 0, 1)  # move axis 2 (channels) to the front
    return chw[np.newaxis, ...]   # add the batch axis


def main(path: Path) -> None:
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    section("1. Loading")
    img = load_rgb(path)
    h, w, c = img.shape
    print(f"shape   {img.shape}  -> {h} rows (height), {w} columns (width), {c} channels (R, G, B)")
    print(f"dtype   {img.dtype}  -> each number is 0..255 (1 byte)")
    print(f"size    {img.size:,} numbers = {img.nbytes / 1e6:.1f} MB in memory")
    print(f"range   min {img.min()}, max {img.max()}")

    section("2. Indexing")
    cy, cx = h // 2, w // 2
    print(f"center pixel img[{cy}, {cx}] = {img[cy, cx]}  (R, G, B)")
    print(f"its red value img[{cy}, {cx}, 0] = {img[cy, cx, 0]}")
    print("average color per channel:", img.mean(axis=(0, 1)).round(1), "(axis=(0,1) averages over rows and columns)")

    section("3. Cropping (slicing)")
    # Stand-in for a detector box: the middle 60% of the photo
    x1, y1, x2, y2 = int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)
    car = crop(img, x1, y1, x2, y2)
    print(f"box (x1={x1}, y1={y1}, x2={x2}, y2={y2}) -> img[{y1}:{y2}, {x1}:{x2}] -> shape {car.shape}")
    print("a crop is a VIEW of the same memory, not a copy:", np.shares_memory(car, img))

    section("4. Resizing for the model")
    square = resize_keep_aspect(car, 320)
    print(f"{car.shape} -> {square.shape} (letterboxed, no stretching)")

    section("5. Flipping (augmentation)")
    flipped = square[:, ::-1]  # reverse the column order = mirror left-right
    print("horizontal flip = square[:, ::-1]; a mirrored car is still the same car model")
    print("vertical flip   = square[::-1, :]; upside-down cars don't happen, so don't use it")

    section("6. Normalizing")
    norm = normalize(square)
    print(f"dtype {norm.dtype}, range {norm.min():.2f} .. {norm.max():.2f}")
    print("per-channel mean after normalizing:", norm.mean(axis=(0, 1)).round(2), "(near 0 for typical photos)")

    section("7. Reordering for PyTorch")
    batch = to_model_input(norm)
    print(f"(H, W, C) {norm.shape} -> (N, C, H, W) {batch.shape}")
    batch_of_two = np.stack([to_model_input(norm)[0], to_model_input(normalize(flipped))[0]])
    print(f"stacking 2 images -> {batch_of_two.shape}: this is literally what a DataLoader hands the model")

    section("8. Saving a visual overview")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    panels = [
        (img, f"Original {img.shape[1]}x{img.shape[0]}"),
        (car, f"Crop img[{y1}:{y2}, {x1}:{x2}]"),
        (square, "Letterboxed to 320x320"),
        (flipped, "Horizontal flip [:, ::-1]"),
    ]
    for ax, (pic, title) in zip(axes.flat, panels):
        ax.imshow(pic)
        ax.set_title(title)
        ax.axis("off")
    # Draw the crop box on the original so you can see where it came from
    axes.flat[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="yellow", lw=2))

    hist_ax = axes.flat[4]
    for i, (name, color) in enumerate([("R", "tab:red"), ("G", "tab:green"), ("B", "tab:blue")]):
        hist_ax.hist(img[..., i].ravel(), bins=64, color=color, alpha=0.5, label=name)
    hist_ax.set_title("Pixel values per channel (0-255)")
    hist_ax.legend()

    # Show what the model "sees" after normalization, rescaled to 0..1 just for display
    view = (norm - norm.min()) / (norm.max() - norm.min())
    axes.flat[5].imshow(view)
    axes.flat[5].set_title("After normalization (rescaled to display)")
    axes.flat[5].axis("off")

    fig.tight_layout()
    out_path = out_dir / "overview.png"
    fig.savefig(out_path, dpi=110)
    print(f"saved {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python image_basics.py path/to/car.jpg")
    photo = Path(sys.argv[1])
    if not photo.is_file():
        raise SystemExit(f"File not found: {photo}")
    main(photo)

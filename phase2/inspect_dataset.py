"""Health check for an image dataset before training on it.

Finds the problems that silently ruin training runs:
  - corrupt or empty files
  - images too small to show detail
  - non-RGB images (grayscale, transparency)
  - exact duplicate photos (a duplicate split across train and val is leakage)
  - class imbalance

Usage:
    python inspect_dataset.py ../phase1/dataset/all
    python inspect_dataset.py ../phase1/dataset            (inspects train/ and val/ separately)
Writes a per-image table to output/dataset_report.csv.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MIN_SIDE = 224  # smaller than this and fine details like grilles and badges are lost


def file_hash(path: Path) -> str:
    """Fingerprint of the file's bytes; identical files get identical hashes."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def inspect_image(path: Path, root: Path) -> dict:
    parts = path.relative_to(root).parts
    # dataset/train/toyota_camry/x.jpg -> split=train, class=toyota_camry
    # dataset/all/toyota_camry/x.jpg or a plain class folder -> split=None
    class_name = parts[-2] if len(parts) >= 2 else "(root)"
    split = parts[-3] if len(parts) >= 3 and parts[-3] in {"train", "val", "test"} else None

    row = {
        "path": str(path.relative_to(root)),
        "split": split,
        "class": class_name,
        "bytes": path.stat().st_size,
        "width": None,
        "height": None,
        "mode": None,
        "ok": False,
        "problem": None,
        "hash": file_hash(path),
    }

    try:
        with Image.open(path) as img:
            img.verify()  # checks the file structure without decoding every pixel
        # verify() leaves the image unusable, so reopen to read its details
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)  # report the size as it will actually be used
            row.update(width=img.width, height=img.height, mode=img.mode, ok=True)
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        row["problem"] = "empty file" if row["bytes"] == 0 else f"corrupt: {type(e).__name__}"
        return row

    if min(row["width"], row["height"]) < MIN_SIDE:
        row["problem"] = f"small (shortest side < {MIN_SIDE}px)"
    elif row["mode"] != "RGB":
        row["problem"] = f"mode {row['mode']} (will be converted to RGB)"
    return row


def main(root: Path) -> None:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise SystemExit(f"No images found under {root}")

    df = pd.DataFrame([inspect_image(p, root) for p in files])

    print(f"Inspected {len(df)} image(s) under {root}\n")

    # --- Class balance ---------------------------------------------------
    group_cols = ["split", "class"] if df["split"].notna().any() else ["class"]
    counts = df.groupby(group_cols).size().rename("images")
    print("Images per class:")
    print(counts.to_string(), "\n")
    per_class = df.groupby("class").size()
    if len(per_class) > 1 and per_class.max() > 3 * per_class.min():
        print(f"[warn] imbalance: largest class has {per_class.max()} images, smallest {per_class.min()}\n")

    # --- Sizes -------------------------------------------------------------
    good = df[df["ok"]]
    if len(good):
        print("Image sizes (readable images):")
        print(good[["width", "height"]].describe().loc[["min", "50%", "max"]].rename(index={"50%": "median"}).to_string(), "\n")

    # --- Problems ----------------------------------------------------------
    problems = df[df["problem"].notna()]
    if len(problems):
        print(f"[warn] {len(problems)} image(s) with problems:")
        print(problems[["path", "problem"]].to_string(index=False), "\n")
    else:
        print("No corrupt, small or non-RGB images.\n")

    # --- Duplicates (empty files all share one hash, so skip them here) -------
    nonempty = df[df["bytes"] > 0]
    dupes = nonempty[nonempty.duplicated("hash", keep=False)].sort_values("hash")
    if len(dupes):
        print(f"[warn] {len(dupes)} file(s) are exact duplicates of another file:")
        print(dupes[["path", "hash"]].to_string(index=False), "\n")
        if "split" in dupes and dupes.groupby("hash")["split"].nunique().gt(1).any():
            print("[ERROR] some duplicates sit in BOTH train and val: that is data leakage.\n")
    else:
        print("No exact duplicates.\n")

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "dataset_report.csv", index=False)
    print(f"Full per-image table: {out_dir / 'dataset_report.csv'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python inspect_dataset.py path/to/dataset")
    main(Path(sys.argv[1]))

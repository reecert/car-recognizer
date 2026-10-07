"""Download Stanford Cars and export a small subset as train/val class folders.

Stanford Cars: 196 car classes (make, model, body, year), ~16,000 photos, cars up to
model year 2012. It's for learning and research only (not a commercial product), and
it's too old to recognize recent cars. That's fine for Phase 3: the goal is to learn
how training works while you collect your own photos.

Output layout (what Ultralytics and PyTorch's ImageFolder both expect):
    ../data/stanford_subset/train/toyota_camry_sedan_2012/00001.jpg
    ../data/stanford_subset/val/toyota_camry_sedan_2012/00042.jpg

Usage:
    python prepare_stanford.py --list                     (print all 196 class names)
    python prepare_stanford.py                            (default 10 classes)
    python prepare_stanford.py --classes "Toyota Camry" "Honda Accord Sedan" "Jeep Wrangler"

The first run downloads the dataset (a couple of GB) into ~/.cache/huggingface.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "tanganke/stanford_cars"

# Common US cars, including some look-alike pairs on purpose (Camry/Corolla, F-150/Silverado)
DEFAULT_CLASSES = [
    "Toyota Camry Sedan",
    "Toyota Corolla Sedan",
    "Honda Accord Sedan",
    "Honda Odyssey Minivan",
    "Ford F-150",
    "Chevrolet Silverado 1500",
    "Jeep Wrangler",
    "Tesla Model S",
    "Ford Mustang",
    "BMW 3 Series Sedan",
]


def folder_name(class_name: str) -> str:
    """'Ford F-150 Regular Cab 2012' -> 'ford_f_150_regular_cab_2012'"""
    return re.sub(r"[^a-z0-9]+", "_", class_name.lower()).strip("_")


def year_of(class_name: str) -> int:
    m = re.search(r"(19|20)\d{2}$", class_name)
    return int(m.group()) if m else 0


def pick_classes(all_names: list[str], keywords: list[str]) -> dict[int, str]:
    """For each keyword, choose the matching class with the newest model year."""
    chosen: dict[int, str] = {}
    for kw in keywords:
        matches = [(i, n) for i, n in enumerate(all_names) if kw.lower() in n.lower()]
        if not matches:
            print(f"[warn] no class matches '{kw}' (use --list to see all names)")
            continue
        idx, name = max(matches, key=lambda m: year_of(m[1]))
        if idx in chosen:
            print(f"[warn] '{kw}' matched '{name}', which is already chosen")
            continue
        if len(matches) > 1:
            others = ", ".join(n for i, n in matches if i != idx)
            print(f"'{kw}' -> {name}   (also matched: {others})")
        else:
            print(f"'{kw}' -> {name}")
        chosen[idx] = name
    return chosen


def export_split(ds, chosen: dict[int, str], out_dir: Path) -> dict[str, int]:
    """Save every image of the chosen classes as a JPEG in out_dir/{class}/."""
    if out_dir.exists():
        shutil.rmtree(out_dir)  # rebuild from scratch so old classes never linger
    subset = ds.filter(lambda label: label in chosen, input_columns="label")  # label only: fast
    counts: dict[str, int] = {}
    for i, example in enumerate(subset):
        cls = folder_name(chosen[example["label"]])
        (out_dir / cls).mkdir(parents=True, exist_ok=True)
        example["image"].convert("RGB").save(out_dir / cls / f"{i:05d}.jpg", quality=95)
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Stanford Cars subset as class folders.")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="keywords, one per class")
    parser.add_argument("--out", type=Path, default=Path("../data/stanford_subset"))
    parser.add_argument("--list", action="store_true", help="print all class names and exit")
    args = parser.parse_args()

    print(f"Loading {DATASET_ID} (first run downloads it) ...")
    train = load_dataset(DATASET_ID, split="train")
    names: list[str] = train.features["label"].names

    if args.list:
        for i, n in enumerate(names):
            print(f"{i:3d}  {n}")
        return

    chosen = pick_classes(names, args.classes)
    if len(chosen) < 2:
        raise SystemExit("Need at least 2 classes to train a classifier.")

    # Stanford's official "test" split becomes our validation set
    test = load_dataset(DATASET_ID, split="test")

    print("\nExporting images ...")
    counts = {
        "train": export_split(train, chosen, args.out / "train"),
        "val": export_split(test, chosen, args.out / "val"),
    }

    print(f"\n{'class':<45}{'train':>7}{'val':>7}")
    for cls in sorted(counts["train"]):
        print(f"{cls:<45}{counts['train'][cls]:>7}{counts['val'].get(cls, 0):>7}")
    print(f"{'TOTAL':<45}{sum(counts['train'].values()):>7}{sum(counts['val'].values()):>7}")
    print(f"\nDataset ready: {args.out}/")


if __name__ == "__main__":
    main()

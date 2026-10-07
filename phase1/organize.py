"""Organize raw car photos into a class-per-folder dataset with a leakage-free split.

Pipeline:
    raw_photos/                      (your camera photos, any order)
      -> dataset/all/{make}_{model}/ (copied and sorted by class)
      -> dataset/train/... and dataset/val/... (split by PHYSICAL car)

Expected filename format:
    {make}_{model}_{carID}_{view}.{ext}
    e.g. toyota_camry_car07_front.jpg

Why split by car and not by image? Photos of the same physical car look
nearly identical. If car07's front photo is in train and its side photo is
in val, the model can "recognize" that exact car instead of the car model,
and validation accuracy comes out misleadingly high. Keeping every photo of
a car in one split prevents that leakage.

Usage:
    python organize.py
    python organize.py --raw raw_photos --out dataset --val-fraction 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# make: letters/digits; model: letters/digits/hyphens (f-150); car id: "car" + digits; view: letters/digits
FILENAME_PATTERN = re.compile(
    r"^(?P<make>[a-z0-9]+)_(?P<model>[a-z0-9-]+)_(?P<car_id>car\d+)_(?P<view>[a-z0-9]+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Photo:
    path: Path
    make: str
    model: str
    car_id: str
    view: str

    @property
    def class_name(self) -> str:
        return f"{self.make}_{self.model}"


# ---------------------------------------------------------------------------
# 1. Scan and parse
# ---------------------------------------------------------------------------

def parse_photo(path: Path) -> tuple[Photo | None, str | None]:
    """Return (Photo, None) for a valid file, or (None, reason) for one to skip."""
    if path.name.startswith("."):
        return None, "hidden system file"
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None, f"not an image ({path.suffix or 'no extension'})"
    match = FILENAME_PATTERN.match(path.stem)
    if match is None:
        return None, "name doesn't match {make}_{model}_{carID}_{view}"
    # Lowercase everything so "Toyota_Camry" and "toyota_camry" land in one class
    return Photo(
        path=path,
        make=match["make"].lower(),
        model=match["model"].lower(),
        car_id=match["car_id"].lower(),
        view=match["view"].lower(),
    ), None


def scan(raw_dir: Path) -> list[Photo]:
    """Find valid photos in raw_dir, warning about everything skipped."""
    if not raw_dir.is_dir():
        raise SystemExit(f"Raw photo folder not found: {raw_dir}")

    photos: list[Photo] = []
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file():
            continue
        photo, reason = parse_photo(path)
        if photo is None:
            print(f"  [skip] {path.name}: {reason}")
        else:
            photos.append(photo)
    return photos


def check_car_ids(photos: list[Photo]) -> None:
    """A carID must belong to exactly one class; otherwise the split rule breaks."""
    classes_per_car: dict[str, set[str]] = defaultdict(set)
    for p in photos:
        classes_per_car[p.car_id].add(p.class_name)
    conflicts = {car: cls for car, cls in classes_per_car.items() if len(cls) > 1}
    if conflicts:
        details = ", ".join(f"{car} -> {sorted(cls)}" for car, cls in conflicts.items())
        raise SystemExit(f"Same carID used for different car models (rename the files): {details}")


# ---------------------------------------------------------------------------
# 2. Organize into dataset/all/{class}/
# ---------------------------------------------------------------------------

def copy_if_needed(src: Path, dst: Path) -> bool:
    """Copy src to dst unless an identical-size copy already exists. Returns True if copied.

    Skipping existing copies makes reruns safe: no duplicates, no crash.
    """
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)  # copy2 keeps timestamps; originals are never moved or changed
    return True


def organize(photos: list[Photo], all_dir: Path) -> None:
    copied = sum(copy_if_needed(p.path, all_dir / p.class_name / p.path.name) for p in photos)
    print(f"  copied {copied} new file(s), {len(photos) - copied} already present")


# ---------------------------------------------------------------------------
# 3. Leakage-free split by physical car
# ---------------------------------------------------------------------------

def split_by_car(photos: list[Photo], val_fraction: float, seed: int) -> dict[str, list[Photo]]:
    """Split per class by carID: all photos of a car go to the same split.

    - roughly val_fraction of each class's cars go to val
    - every class with 2+ cars gets at least 1 val car
    - a class with only 1 car keeps it in train (val needs other cars to be meaningful)
    - the same seed always produces the same split
    """
    rng = random.Random(seed)  # own generator, so nothing else can disturb the result

    cars_by_class: dict[str, set[str]] = defaultdict(set)
    for p in photos:
        cars_by_class[p.class_name].add(p.car_id)

    val_cars: set[str] = set()
    for class_name in sorted(cars_by_class):  # sorted -> deterministic order
        car_ids = sorted(cars_by_class[class_name])  # sorted before shuffling -> reproducible
        if len(car_ids) < 2:
            print(f"  [warn] {class_name} has only 1 car; it stays in train (no val example)")
            continue
        rng.shuffle(car_ids)
        n_val = max(1, round(len(car_ids) * val_fraction))
        n_val = min(n_val, len(car_ids) - 1)  # always leave at least 1 car for training
        val_cars.update(car_ids[:n_val])

    splits: dict[str, list[Photo]] = {"train": [], "val": []}
    for p in photos:
        splits["val" if p.car_id in val_cars else "train"].append(p)
    return splits


def write_splits(splits: dict[str, list[Photo]], out_dir: Path, all_dir: Path) -> None:
    """Rebuild dataset/train and dataset/val from dataset/all.

    The split folders are deleted and rebuilt each run, so changing the seed or
    val fraction never leaves stale files behind. dataset/all is kept.
    """
    for split_name, split_photos in splits.items():
        split_dir = out_dir / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        for p in split_photos:
            src = all_dir / p.class_name / p.path.name
            dst = split_dir / p.class_name / p.path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# 4. Report and leakage check
# ---------------------------------------------------------------------------

def assert_no_leakage(splits: dict[str, list[Photo]]) -> None:
    train_cars = {p.car_id for p in splits["train"]}
    val_cars = {p.car_id for p in splits["val"]}
    overlap = train_cars & val_cars
    assert not overlap, f"Data leakage: these cars are in both train and val: {sorted(overlap)}"


def report(splits: dict[str, list[Photo]]) -> None:
    classes = sorted({p.class_name for photos in splits.values() for p in photos})
    header = f"{'class':<16}{'train imgs':>11}{'train cars':>12}{'val imgs':>10}{'val cars':>10}"
    print(header)
    print("-" * len(header))

    totals = [0, 0, 0, 0]
    for c in classes:
        row = []
        for split_name in ("train", "val"):
            in_class = [p for p in splits[split_name] if p.class_name == c]
            row += [len(in_class), len({p.car_id for p in in_class})]
        totals = [t + r for t, r in zip(totals, row)]
        print(f"{c:<16}{row[0]:>11}{row[1]:>12}{row[2]:>10}{row[3]:>10}")

    print("-" * len(header))
    print(f"{'TOTAL':<16}{totals[0]:>11}{totals[1]:>12}{totals[2]:>10}{totals[3]:>10}")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Organize car photos and split by physical car.")
    parser.add_argument("--raw", type=Path, default=Path("raw_photos"), help="folder of raw photos")
    parser.add_argument("--out", type=Path, default=Path("dataset"), help="output dataset folder")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="share of cars per class for val")
    parser.add_argument("--seed", type=int, default=42, help="random seed for a reproducible split")
    args = parser.parse_args()

    if not 0 < args.val_fraction < 1:
        raise SystemExit("--val-fraction must be between 0 and 1")

    all_dir = args.out / "all"

    print(f"Scanning {args.raw}/ ...")
    photos = scan(args.raw)
    if not photos:
        raise SystemExit("No valid photos found.")
    check_car_ids(photos)
    print(f"  found {len(photos)} valid photo(s)")

    print(f"\nOrganizing into {all_dir}/ ...")
    organize(photos, all_dir)

    print(f"\nSplitting by car (val fraction {args.val_fraction}, seed {args.seed}) ...")
    splits = split_by_car(photos, args.val_fraction, args.seed)
    write_splits(splits, args.out, all_dir)

    print()
    report(splits)
    assert_no_leakage(splits)
    print("\nLeakage check passed: no car appears in both train and val.")


if __name__ == "__main__":
    main()

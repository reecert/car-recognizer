"""Organize raw car photos into a class-per-folder dataset with a leakage-free split.

Pipeline:
    raw_photos/                      (your camera photos, any order)
      -> dataset/all/{make}_{model}/ (copied and sorted by class)
      -> dataset/train/, val/, test/  (split by PHYSICAL car)

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
    python organize.py --raw ../data/raw_photos --out ../data/dataset
    python organize.py --raw ../data/raw_photos --out ../data/dataset --val-fraction 0.15 --test-fraction 0.15
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

def split_counts(n_cars: int, val_fraction: float, test_fraction: float) -> tuple[int, int]:
    """How many of a class's cars go to val and test. Train always keeps at least 1."""
    if n_cars < 2:
        return 0, 0                                   # 1 car: train only
    if n_cars == 2 or test_fraction == 0:
        return 1 if n_cars == 2 else max(1, round(n_cars * val_fraction)), 0
    n_val = max(1, round(n_cars * val_fraction))
    n_test = max(1, round(n_cars * test_fraction))
    while n_val + n_test > n_cars - 1:                # always leave at least 1 car for training
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    return n_val, n_test


def split_by_car(photos: list[Photo], val_fraction: float, test_fraction: float,
                 seed: int) -> dict[str, list[Photo]]:
    """Split per class by carID: all photos of a car go to the same split.

    - train: the model learns from these
    - val:   used DURING development to pick epochs and settings
    - test:  touched ONCE at the end for the honest final number; never used for decisions
    - a class with 3+ cars gets at least 1 val car and 1 test car
    - a class with 2 cars gets 1 train + 1 val (no test example: warned)
    - a class with 1 car stays in train (warned)
    - the same seed always produces the same split
    """
    rng = random.Random(seed)  # own generator, so nothing else can disturb the result

    cars_by_class: dict[str, set[str]] = defaultdict(set)
    for p in photos:
        cars_by_class[p.class_name].add(p.car_id)

    split_of_car: dict[str, str] = {}
    for class_name in sorted(cars_by_class):  # sorted -> deterministic order
        car_ids = sorted(cars_by_class[class_name])  # sorted before shuffling -> reproducible
        rng.shuffle(car_ids)
        n_val, n_test = split_counts(len(car_ids), val_fraction, test_fraction)
        if len(car_ids) == 1:
            print(f"  [warn] {class_name} has only 1 car: train only (no val or test example)")
        elif n_test == 0 and test_fraction > 0:
            print(f"  [warn] {class_name} has only {len(car_ids)} cars: no test example (need 3+)")
        for i, car in enumerate(car_ids):
            split_of_car[car] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")

    splits: dict[str, list[Photo]] = {"train": [], "val": [], "test": []}
    for p in photos:
        splits[split_of_car[p.car_id]].append(p)
    return splits


def write_splits(splits: dict[str, list[Photo]], out_dir: Path, all_dir: Path) -> None:
    """Rebuild dataset/train, val and test from dataset/all.

    The split folders are deleted and rebuilt each run, so changing the seed or
    fractions never leaves stale files behind. dataset/all is kept.
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
    cars = {name: {p.car_id for p in photos} for name, photos in splits.items()}
    names = list(cars)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = cars[a] & cars[b]
            assert not overlap, f"Data leakage: cars in both {a} and {b}: {sorted(overlap)}"


def report(splits: dict[str, list[Photo]]) -> None:
    classes = sorted({p.class_name for photos in splits.values() for p in photos})
    names = ["train", "val", "test"]
    header = f"{'class':<20}" + "".join(f"{n + ' imgs':>11}{n + ' cars':>11}" for n in names)
    print(header)
    print("-" * len(header))

    totals = [0] * (2 * len(names))
    for c in classes:
        row = []
        for split_name in names:
            in_class = [p for p in splits[split_name] if p.class_name == c]
            row += [len(in_class), len({p.car_id for p in in_class})]
        totals = [t + r for t, r in zip(totals, row)]
        print(f"{c:<20}" + "".join(f"{v:>11}" for v in row))

    print("-" * len(header))
    print(f"{'TOTAL':<20}" + "".join(f"{v:>11}" for v in totals))


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Organize car photos and split by physical car.")
    parser.add_argument("--raw", type=Path, default=Path("raw_photos"), help="folder of raw photos")
    parser.add_argument("--out", type=Path, default=Path("dataset"), help="output dataset folder")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="share of cars per class for val")
    parser.add_argument("--test-fraction", type=float, default=0.15,
                        help="share of cars per class for test (0 = no test split)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for a reproducible split")
    args = parser.parse_args()

    if not 0 < args.val_fraction < 1 or not 0 <= args.test_fraction < 1:
        raise SystemExit("--val-fraction must be in (0, 1) and --test-fraction in [0, 1)")
    if args.val_fraction + args.test_fraction >= 0.8:
        raise SystemExit("val + test fractions leave too little for training")

    all_dir = args.out / "all"

    print(f"Scanning {args.raw}/ ...")
    photos = scan(args.raw)
    if not photos:
        raise SystemExit("No valid photos found.")
    check_car_ids(photos)
    print(f"  found {len(photos)} valid photo(s)")

    print(f"\nOrganizing into {all_dir}/ ...")
    organize(photos, all_dir)

    print(f"\nSplitting by car (val {args.val_fraction}, test {args.test_fraction}, seed {args.seed}) ...")
    splits = split_by_car(photos, args.val_fraction, args.test_fraction, args.seed)
    write_splits(splits, args.out, all_dir)

    print()
    report(splits)
    assert_no_leakage(splits)
    print("\nLeakage check passed: no car appears in more than one split.")

    # Training needs every class present in val (and test, if used)
    all_classes = {p.class_name for p in photos}
    needed = ["val"] + (["test"] if args.test_fraction > 0 else [])
    missing = {s: sorted(all_classes - {p.class_name for p in splits[s]}) for s in needed}
    missing = {s: m for s, m in missing.items() if m}
    if missing:
        print("\n[not ready to train] some classes have no photos in:")
        for s, m in missing.items():
            print(f"  {s}: {', '.join(m)}")
        print("Collect at least 3 different cars per class (2 if you use --test-fraction 0).")
    else:
        print("Every class is present in every split: ready to train.")


if __name__ == "__main__":
    main()

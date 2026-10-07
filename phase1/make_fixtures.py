"""Create a fake raw_photos/ folder of empty files to test the organizer.

The names mimic how real photos will be named:
    {make}_{model}_{carID}_{view}.{ext}
plus a few junk files that real folders always contain.
"""

from pathlib import Path

FIXTURE_NAMES = [
    "toyota_camry_car01_front.jpg",
    "toyota_camry_car01_side.jpg",
    "toyota_camry_car02_front.jpg",
    "toyota_camry_car02_rear.jpg",
    "toyota_camry_car03_side.jpg",
    "honda_civic_car04_front.jpg",
    "honda_civic_car04_rear.JPG",   # uppercase extension
    "honda_civic_car05_side.jpg",
    "honda_civic_car06_front.png",  # different image format
    "ford_f-150_car07_front.jpg",   # hyphen in the model name
    "ford_f-150_car07_side.jpg",
    "ford_f-150_car08_rear.jpg",
    "tesla_model3_car09_front.jpg",
    "tesla_model3_car10_side.jpg",
    "tesla_model3_car10_rear.jpg",
    "IMG_4021.jpg",                 # junk: doesn't follow the naming pattern
    "notes.txt",                    # junk: not an image
    ".DS_Store",                    # junk: macOS hidden file
]


def make_fixtures(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in FIXTURE_NAMES:
        (raw_dir / name).touch()
    print(f"Created {len(FIXTURE_NAMES)} files in {raw_dir}/")


if __name__ == "__main__":
    make_fixtures(Path("raw_photos"))

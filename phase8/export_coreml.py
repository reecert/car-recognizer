"""Phase 8: convert your trained PyTorch classifier into a CoreML model for the iPhone.

The exported model takes a plain camera image and returns a probability per car,
so the iPhone app does no math of its own:
  - ImageNet normalization is baked INTO the model (the app passes raw pixels)
  - softmax is baked in (outputs are probabilities, 0..1)
  - class names are embedded (Apple's Vision framework returns labels directly)

Writes to ios_assets/:
  CarClassifier.mlpackage   the model, ready to drop into Xcode
  specs.json                specs for the classes this model knows (if --specs given)

Usage:
    python export_coreml.py ../phase4/runs/pytorch_resnet18/best.pt --specs specs_stanford.json
    python export_coreml.py ../phase4/runs/pytorch_resnet18/best.pt --int8      (half the size)

On a Mac it also checks that CoreML and PyTorch agree on real validation photos.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase4"))
from train_pytorch import IMAGENET_MEAN, IMAGENET_STD, build_model  # noqa: E402

import coremltools as ct  # noqa: E402  (imported after torch on purpose)


class DeployableClassifier(nn.Module):
    """Wraps the trained network so the phone can feed it raw pixels.

    Input:  (1, 3, H, W) RGB in 0..1 (CoreML scales 0..255 down by 1/255 for us)
    Output: (1, num_classes) probabilities
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model((x - self.mean) / self.std), dim=1)


def center_square(img, size: int):
    """The same crop the app uses: the largest centered square, resized to size x size."""
    from PIL import Image
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((size, size), Image.Resampling.BILINEAR)


def parity_check(wrapper: nn.Module, mlmodel, val_dir: Path, classes: list[str], imgsz: int, n: int) -> None:
    """Run the same photos through PyTorch and CoreML; they should agree."""
    from PIL import Image, ImageOps
    files = sorted(p for p in val_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    files = files[:: max(1, len(files) // n)][:n]  # spread the sample across all classes
    if not files:
        print("  (no validation photos found, skipping parity check)")
        return
    agree, diffs = 0, []
    for f in files:
        with Image.open(f) as im:
            img = center_square(ImageOps.exif_transpose(im).convert("RGB"), imgsz)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
        with torch.no_grad():
            p_torch = wrapper(x)[0].numpy()
        out = mlmodel.predict({"image": img})
        prob_dict = next(v for v in out.values() if isinstance(v, dict))
        p_coreml = np.array([prob_dict[c] for c in classes])
        agree += int(p_torch.argmax() == p_coreml.argmax())
        diffs.append(np.abs(p_torch - p_coreml).max())
    print(f"  top-1 agreement: {agree}/{len(files)} photos")
    print(f"  largest probability difference: {max(diffs):.4f} (FP16/INT8 rounding; small is fine)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained classifier to CoreML.")
    parser.add_argument("checkpoint", type=Path, help="best.pt from phase4/train_pytorch.py")
    parser.add_argument("--out", type=Path, default=Path("ios_assets"))
    parser.add_argument("--specs", type=Path, default=None, help="JSON of specs keyed by class name")
    parser.add_argument("--int8", action="store_true", help="8-bit weights: ~half the size, tiny accuracy cost")
    parser.add_argument("--val", type=Path, default=Path("../data/stanford_subset/val"),
                        help="photos for the PyTorch-vs-CoreML check (Mac only)")
    parser.add_argument("--check", type=int, default=40, help="how many photos to compare")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    classes, imgsz = ckpt["classes"], ckpt["imgsz"]
    model = build_model(ckpt["arch"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["model"])
    wrapper = DeployableClassifier(model).eval()
    print(f"Loaded {ckpt['arch']} | {len(classes)} classes | input {imgsz}x{imgsz} | val top-1 {ckpt['val_top1']:.1%}")

    # 1. Trace: record the network's operations by running one example through it
    traced = torch.jit.trace(wrapper, torch.rand(1, 3, imgsz, imgsz))

    # 2. Convert to CoreML
    print("Converting to CoreML ...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, imgsz, imgsz),
                             scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
        classifier_config=ct.ClassifierConfig(classes),
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,     # half precision: what the Neural Engine runs
        minimum_deployment_target=ct.target.iOS16,
    )

    if args.int8:
        from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights
        config = OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric"))
        mlmodel = linear_quantize_weights(mlmodel, config=config)
        print("Applied 8-bit weight quantization")

    mlmodel.short_description = "Car make/model classifier"
    mlmodel.user_defined_metadata["arch"] = ckpt["arch"]
    mlmodel.user_defined_metadata["val_top1"] = f"{ckpt['val_top1']:.4f}"
    mlmodel.user_defined_metadata["precision"] = "int8 weights" if args.int8 else "fp16"

    args.out.mkdir(parents=True, exist_ok=True)
    pkg = args.out / "CarClassifier.mlpackage"
    if pkg.exists():
        shutil.rmtree(pkg)
    mlmodel.save(str(pkg))
    size_mb = sum(f.stat().st_size for f in pkg.rglob("*") if f.is_file()) / 1e6
    print(f"Saved {pkg} ({size_mb:.1f} MB)")

    # 3. Specs for the app: only the classes this model knows
    if args.specs:
        all_specs = json.loads(args.specs.read_text(encoding="utf-8"))
        app_specs = {c: all_specs[c] for c in classes if c in all_specs}
        missing = [c for c in classes if c not in all_specs]
        (args.out / "specs.json").write_text(json.dumps(app_specs, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {args.out / 'specs.json'} ({len(app_specs)} of {len(classes)} classes)")
        if missing:
            print(f"  [warn] no specs for: {', '.join(missing)} (the app will show the name only)")

    # 4. Parity check (CoreML can only run predictions on macOS)
    if platform.system() == "Darwin":
        print("\nChecking CoreML against PyTorch on validation photos ...")
        if args.val.is_dir():
            parity_check(wrapper, mlmodel, args.val, classes, imgsz, args.check)
        else:
            print(f"  (skipped: {args.val} not found)")
    else:
        print("\n(Parity check skipped: CoreML predictions only run on macOS.)")


if __name__ == "__main__":
    main()

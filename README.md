# car-recognizer

Point an iPhone at a car and get its make, model and specs. Built in phases, one folder each.
Run every script **from inside its phase folder** (paths like `../data/...` are relative to it).

| Phase | Folder | What it does |
|---|---|---|
| 1 | `phase1/` | `organize.py`: sort raw photos into a leakage-free train/val/test split (by physical car). `make_fixtures.py` creates test files. |
| 2 | `phase2/` | `detect_cars.py`: pretrained YOLO car detector and crops. `inspect_dataset.py`: dataset report. |
| 3 | `phase3/` | `prepare_stanford.py`: export a 10-class Stanford Cars subset to `data/`. `train.py` / `predict.py`: Ultralytics classifier. |
| 4 | `phase4/` | `train_pytorch.py`: the same in plain PyTorch, with ablations (`--scratch`, `--arch`). |
| 5 | `phase5/` | `inspect_model.py` (filters, Grad-CAM, feature space), `linear_probe.py`. |
| 6 | `phase6/` | `evaluate.py`: the held-out test score, per-class accuracy, confidence thresholds. |
| 8 | `phase8/` | `export_coreml.py` and the SwiftUI app sources (`*.swift`). |

## Setup

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd phase3 && python prepare_stanford.py       # downloads Stanford Cars, writes ../data/stanford_subset
```

## Typical run

```sh
cd phase4 && python train_pytorch.py                                   # -> runs/pytorch_resnet18/best.pt
cd ../phase6 && python evaluate.py ../phase4/runs/pytorch_resnet18/best.pt --data ../data/stanford_subset --split val
cd ../phase8 && python export_coreml.py ../phase4/runs/pytorch_resnet18/best.pt --specs specs_stanford.json
```

Then add `phase8/*.swift` plus `ios_assets/CarClassifier.mlpackage` and `ios_assets/specs.json` to an Xcode iOS app target.

Not in git (see `.gitignore`): datasets in `data/`, training runs, `*.pt` weights, the CoreML export.

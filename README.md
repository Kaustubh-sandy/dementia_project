# Dementia MRI: CNN-ViT feature fusion (OASIS-1)

Predicts the recorded CDR group (0 vs >0) from frozen ResNet-18 / ViT-B/16 slice features, compares CNN, ViT and fusion
at 4/8/16 axial slices, measures compute cost, and ships a local Streamlit demo. Academic prototype; not a diagnosis.
Design notes: [docs/DECISIONS.md](docs/DECISIONS.md), [docs/IMPLEMENTATION_AND_TRAINING.md](docs/IMPLEMENTATION_AND_TRAINING.md).

## 1. Setup (run from this folder, PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build from pytorch.org
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip freeze > requirements-lock.txt
python -m pytest            # synthetic tests
```

## 2. Data
Paths are in `configs/base.yaml` and already point at the discs and spreadsheet one level above this folder:
`../oasis_cross-sectional_disc*/` and `../oasis_cross-sectional-5708aa0a98d82080.xlsx` (xlsx or csv both work).
Extract ALL discs first. Only discs 1-6 are extracted now (94 of 198 eligible participants).

## 3. Run (baseline: CNN-16 + age-only)
```powershell
dementia prepare                           # inventory, cohort, 80/20 split, CV folds -> outputs/manifests
dementia inspect                           # optional: QC montage in outputs/qc (orientation/slices are preset, see config)
dementia extract --encoder cnn --limit 5   # pilot; prints seconds per participant
dementia extract --encoder cnn             # all development participants
dementia train                             # CV on development only; locks the run
dementia extract --encoder cnn --split test
dementia evaluate                          # ONCE; metrics, CIs, ROC -> outputs/results
dementia benchmark --subjects 2 --repeats 2
dementia predict --image "<path to *_t88_masked_gfc.hdr>"
streamlit run app/app.py
```

## 4. Full comparison (9 configs + age baseline), only if the ViT pilot time is acceptable
```powershell
dementia --config configs/full.yaml prepare
dementia --config configs/full.yaml extract --encoder cnn
dementia --config configs/full.yaml extract --encoder vit --limit 5    # time it first
dementia --config configs/full.yaml extract --encoder vit
dementia --config configs/full.yaml train --full
dementia --config configs/full.yaml extract --encoder cnn --split test
dementia --config configs/full.yaml extract --encoder vit --split test
dementia --config configs/full.yaml evaluate
dementia --config configs/full.yaml benchmark
$env:DEMENTIA_CONFIG = "configs/full.yaml"; streamlit run app/app.py
```
Use a new `output_dir` per protocol (full.yaml uses `outputs_full`). `train` locks a run; `evaluate` runs once.
If you add discs after `prepare`, rerun `dementia prepare --force` BEFORE `train` (feature caches are reused if still valid).

## Data policy
OASIS scans, clinical tables and all participant-level outputs are git-ignored. Never commit them.

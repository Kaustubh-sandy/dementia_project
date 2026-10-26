# Dementia MRI: CNN-ViT feature fusion (OASIS-1)

Predicts the recorded CDR group (CDR 0 vs CDR > 0) from a brain MRI. Frozen ImageNet ResNet-18 and ViT-B/16
turn axial slices into features; only a logistic regression is trained. Academic prototype, not a diagnosis.
Start with [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for the project flow and the outputs you get.

```
dementia_project/
├── config.yaml          paths, cohort rules, slices, suites
├── run.py               pipeline entry point (one step per command)
├── app.py               Streamlit demo
├── dementia/
│   ├── data.py          scans + clinical labels -> cohort -> 80/20 split + CV folds
│   ├── preprocess.py    MRI -> normalised axial slices -> 224x224 input; audit
│   ├── features.py      frozen encoders, cached features, pooling/fusion; doctor
│   ├── train.py         scaler + logistic regression, C tuned by CV, selection, lock
│   ├── evaluate.py      held-out metrics, bootstrap CIs, ROC, confusion matrices
│   └── predict.py       single-scan prediction, live-vs-cached check, timing
├── tests/               synthetic checks
├── docs/                guides, IMPLEMENTATION_AND_TRAINING.md, HANDOFF_CHECKLIST.md
└── outputs/             generated, git-ignored
```

## Data (already wired up in config.yaml)
All dataset files live in `C:\AUTISM-Project\DATASET\` (one level above this folder):
- Scans: `../DATASET/oasis_cross-sectional_disc*/discN/OAS1_xxxx_MR1/PROCESSED/MPRAGE/T88_111/*_t88_masked_gfc.hdr` + `.img`
- Labels: `../DATASET/oasis_cross-sectional-5708aa0a98d82080.xlsx` (ID, Age, CDR)
- New discs: extract them into `DATASET/` as `oasis_cross-sectional_discN/`; they are picked up automatically.
- Cohort: age >= 60, CDR in {0, 0.5, 1, 2}, earliest session per person; label = CDR > 0.
- Discs 1-6 give 94 participants (48 CDR 0, 46 CDR > 0). All discs give about 198. Extract every disc before the real run.
- Orientation and slices are preset and checked: axial = array axis 2, 16 slices at indices 41..131.

## What is trained
| Part | Trained? |
|---|---|
| ResNet-18 (512-d/slice), ViT-B/16 (768-d/slice) | No, frozen ImageNet weights |
| Mean of a participant's slice features; fusion = concat (1280-d) | No, fixed |
| StandardScaler + class-balanced L2 logistic regression | Yes; C in {0.001, 0.01, 0.1, 1} chosen by 5-fold CV ROC-AUC on development only, then refit on all development |
| Age-only logistic regression | Yes, confound baseline |

Suites (`suite:` in config.yaml or `--suite`): **review** = CNN-16, ViT-16, fusion-16 (+ age), **full** = all three at 4/8/16 slices (+ age),
**cnn** = CNN-16 (+ age). The app uses the model with the best development CV AUC.

## Setup (once)
```powershell
cd C:\AUTISM-Project\dementia_project
.\.venv\Scripts\Activate.ps1
# GPU (RTX 3050 Ti): copy the Windows + Pip + CUDA command from https://pytorch.org/get-started/locally/
# CPU only:          python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest -q
```

## Run the pipeline (in this order)
```powershell
python run.py doctor                 # versions, CUDA/GPU, real ResNet+ViT forward pass (downloads weights once)
python run.py prepare                # outputs/participants.csv, exclusions.csv, prepare_summary.json
python run.py audit                  # every dev scan loads; montages in outputs/qc - look at them
python run.py extract --limit 2      # pilot: prints seconds per participant
python run.py extract                # dev features for CNN and ViT (resumable)
python run.py train                  # review suite; outputs/results/dev_cv_results.csv; locks the models
python run.py audit --split test
python run.py extract --split test   # only allowed after train
python run.py evaluate               # ONCE: test_metrics.csv, test_roc.png, confusion_*.png, paired_auc_differences.csv
python run.py verify                 # live prediction must equal cached-feature prediction
python run.py benchmark              # outputs/results/timings.csv
streamlit run app.py                 # demo at http://localhost:8501
```
Single scan: `python run.py predict --image "<path to *_t88_masked_gfc.hdr>"`.
Full slice-budget study: `python run.py train --suite full` instead of `train` (same features, decide before evaluating).

## Rules the code enforces
- Split by participant; the scaler is fitted inside each CV training fold.
- `prepare` can be rerun (e.g. after extracting more discs) until `train` locks. A new protocol needs a new `output_dir`.
- After `train`, editing participants.csv, slices, geometry or threshold is detected and refused.
- Test features only after the lock; `evaluate` runs once.
- CUDA out of memory: set `batch_size: 1` in config.yaml.

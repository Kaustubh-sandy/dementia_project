# Dementia MRI: CNN-ViT feature fusion (OASIS-1)

Predicts the recorded CDR group (CDR 0 vs CDR > 0) from a brain MRI. Frozen ImageNet ResNet-18 and ViT-B/16
turn 4/8/16 axial slices into features; only a logistic regression is trained. Academic prototype, not a diagnosis.

```
dementia_project/
├── config.yaml          all settings (paths, cohort, slices, experiments)
├── run.py               pipeline entry point
├── app.py               Streamlit demo
├── dementia/
│   ├── data.py          scans + clinical labels -> cohort -> 80/20 split + CV folds
│   ├── preprocess.py    MRI -> normalised axial slices -> 224x224 model input
│   ├── features.py      frozen encoders, cached features, mean pooling, fusion
│   ├── train.py         scaler + logistic regression, C tuned by CV, model selection
│   ├── evaluate.py      held-out metrics, bootstrap CIs, ROC
│   └── predict.py       single-scan prediction + timing benchmark
├── tests/               synthetic checks
└── outputs/             generated (git-ignored)
```

## What is trained
| Part | Trained? |
|---|---|
| ResNet-18 (512-d per slice), ViT-B/16 (768-d per slice) | No, frozen |
| Mean of slice features per participant; fusion = concat (1280-d) | No, fixed |
| StandardScaler + class-balanced L2 logistic regression | Yes, C chosen by 5-fold CV ROC-AUC on dev only |
| Age-only logistic regression | Yes, baseline |

Models: {cnn, vit, fusion} x {4, 8, 16 slices} = 9, plus age_only. Set in `config.yaml`.

## Run
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # if not installed
python -m pip install -r requirements.txt
python -m pytest -q

python run.py prepare                    # outputs/participants.csv, exclusions.csv, prepare_summary.json
python run.py qc                         # outputs/qc/*.png - check slices look right
python run.py extract --limit 5          # pilot: prints seconds per participant
python run.py extract                    # all dev participants, CNN + ViT (resumable)
python run.py train                      # outputs/results/dev_cv_results.csv; locks models
python run.py extract --split test       # test features, only allowed after train
python run.py evaluate                   # ONCE: test_metrics.csv, test_roc.png, paired_auc_differences.csv
python run.py benchmark                  # outputs/results/timings.csv
python run.py predict --image "..\oasis_cross-sectional_disc1\disc1\OAS1_0001_MR1\PROCESSED\MPRAGE\T88_111\OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc.hdr"
streamlit run app.py
```
Or everything in one go: `python run.py all`.

Quick CNN-only run: set `representations: [cnn]` and `budgets: [16]` in `config.yaml`.

## Rules
- Split is by participant; the test set is used once, after `train` locks the models.
- `prepare` can be rerun (e.g. after extracting more discs) until `train` runs. To start over, change `output_dir`.
- Extract ALL OASIS discs before the real run; discs 1-6 give only 94 of the 198 eligible participants.

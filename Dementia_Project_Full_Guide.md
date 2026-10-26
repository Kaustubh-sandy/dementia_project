# MRI-Based Dementia Classification: CNN–ViT Feature Fusion
### Full project summary, dataset download list, folder structure and step-by-step flow

Course: BEEE497J, Review I | Guide: Dr. M Monica Subashini
Review date: 28 Oct 2026 | Start date: 26 Oct 2026

---

## 1. Project at a glance

| Item | Decision |
|---|---|
| Title | MRI-Based Dementia Classification Using CNN–Vision Transformer Feature Fusion: A Study of Accuracy and Computational Efficiency |
| Research question | When does combining CNN and ViT features improve participant-level CDR-group classification enough to justify the extra computation, and how many MRI slices are actually needed? |
| Dataset | OASIS-1 cross-sectional (structural T1 MRI + clinical CSV) |
| Target | CDR = 0 → class 0 (non-demented). CDR > 0 (0.5, 1, 2) → class 1. This is a CDR group, **not** a confirmed Alzheimer's diagnosis. |
| Cohort | Age ≥ 60, valid CDR, usable scan, **one session per participant** (earliest) |
| Encoders | ResNet-18 (CNN, 512-d) and ViT-B/16 (768-d), ImageNet-pretrained, **frozen** (never trained) |
| What IS trained | Only a StandardScaler + class-balanced L2 logistic regression on pooled features |
| Slice budgets | 4 / 8 / 16 axial slices (nested subsets of one fixed 16-slice list) |
| Representations | CNN only (512-d), ViT only (768-d), Fusion = concat (1280-d) |
| Experiments | 3 representations × 3 slice budgets = **9 configurations** + **age-only baseline** |
| Split | Stratified 80 % development / 20 % held-out test, at participant level, fixed by seed |
| Primary metric | Participant-level ROC-AUC (plus balanced accuracy, sensitivity, specificity, F1, confusion matrix, bootstrap 95 % CIs) |
| Cost metrics | Extraction time per participant, fit time, peak GPU/CPU memory |
| Honest framing | Fusion does not have to win. A well-controlled negative result still answers the question. |

---

## 2. Which OASIS datasets to download

From the OASIS-1 download page (your screenshot), there are four blocks. You need only **two things**.

| Page section | Download? | Why |
|---|---|---|
| **CSV File with Demographic and Clinical Data** (`oasis_cross-sectional.csv`) | **YES, required** | Contains ID, Age, CDR (labels), plus MMSE, eTIV, nWBV, ASF etc. Labels and age baseline come from here. |
| **OASIS-1: Raw Data Download** (the numbered disc archives, Disc 1 … Disc N) | **YES, required** | Despite the name, each session archive contains `RAW/`, `PROCESSED/` and `FSL_SEG/`. The image we need, `PROCESSED/T88_111/*_t88_masked_gfc`, is inside these. |
| Cross-sectional Fact Sheet (PDF) | Optional (you already have it) | Documents image products, orientation and demographics. Keep it for the report. |
| Reliability Data Spreadsheet | No | Only for scan–rescan reliability (the 20 repeat sessions). Not needed. |
| OASIS-1: FreeSurfer Download | No | Segmentation/volume outputs. We do not use them. |

**Download all the numbered discs**, not just some. Subject IDs are spread across discs, so skipping discs silently drops participants. (Check the exact disc count shown on the page when you expand "Raw Data Download".)

### The exact image file we use
Per session folder:
```
OAS1_xxxx_MRy/
├── OAS1_xxxx_MRy.xml / .txt
├── RAW/                      <- NOT used (individual scan repeats)
├── PROCESSED/
│   ├── SUBJ_111/             <- NOT used (native-space average)
│   └── T88_111/
│       ├── *_t88_gfc.hdr/.img           <- NOT used (not brain-masked)
│       └── *_t88_masked_gfc.hdr/.img    <- USE THIS (brain-masked, atlas-aligned, 176×208×176, 1 mm)
└── FSL_SEG/                  <- NOT used (tissue labels, not intensity)
```
Use **`*_t88_masked_gfc.hdr` + `.img`** (Analyze 7.5 format, 16-bit big-endian). Both files must stay together in the same folder.

### Disk space and access
- Whole dataset: about 15.8 GB compressed, about 50 GB uncompressed. Plan for roughly 70 GB free during extraction.
- You only need the masked `.hdr/.img` pair for eligible participants (each roughly 13 MB). After extraction you can copy just those into a slim folder and delete the rest.
- OASIS requires registration / acceptance of the data-use agreement. Your deck lists the access request as *in progress*, so finish this first. Everything below depends on it.

### What you should expect in the cohort (from the Fact Sheet, Table 2)
- Total 416 subjects, aged 18–96. The CSV has about 436 rows because 20 non-demented subjects have a second reliability session (MR2). The "earliest session per participant" rule handles this.
- Subjects **aged ≥ 60**: about 198 (98 non-demented, 100 demented: CDR 0.5 = 70, CDR 1 = 28, CDR 2 = 2).
- Under-60s have no CDR dementia cases and are excluded by the age rule.
- After the 80/20 split: roughly 158 development / 40 test participants. Final counts depend on missing values and QC, so **report counts from your own `preparation.json`**, not these estimates.
- A test set of about 40 means confidence intervals will be wide. Say so.

---

## 3. Folder structure

### 3.1 Data (outside the code folder)
```
D:/OASIS1/
├── oasis_cross-sectional.csv          <- clinical CSV (ID, Age, CDR, ...)
├── OAS1_0001_MR1/
│   ├── RAW/
│   ├── PROCESSED/
│   │   ├── SUBJ_111/
│   │   └── T88_111/
│   │       ├── OAS1_0001_MR1_..._t88_masked_gfc.hdr   <- used
│   │       └── OAS1_0001_MR1_..._t88_masked_gfc.img   <- used
│   └── FSL_SEG/
├── OAS1_0002_MR1/
└── ...
```
The original data stays on your PC. Nothing is uploaded.

### 3.2 Project folder (from the starter package)
```
D:/dementia_starter/
├── config.yaml              # paths, geometry, slice indices, experiment settings
├── pipeline.py              # prepare / inspect / extract / train / evaluate / benchmark / predict
├── app.py                   # local Streamlit demo (same prediction code)
├── requirements.txt         # dependencies (install torch/torchvision separately)
├── requirements-lock.txt    # created by you after install (pip freeze)
├── README.md
├── VALIDATION.md            # what was tested before delivery
├── tests/
│   └── test_pipeline.py     # synthetic checks only, never report as accuracy
├── .venv/                   # your virtual environment
└── outputs/                 # everything generated
    ├── manifests/
    │   ├── inventory.csv        # all MRI files found
    │   ├── participants.csv     # final cohort, labels, split, CV folds
    │   ├── exclusions.csv       # who was dropped and why
    │   └── preparation.json     # counts summary
    ├── qc/
    │   ├── <id>_planes.png      # orientation preview
    │   └── <id>_geometry.json   # shape, zooms, effective_affine
    ├── features/
    │   ├── cnn/<participant>.npz    # (n_slices, 512)
    │   └── vit/<participant>.npz    # (n_slices, 768)
    ├── models/
    │   ├── *.joblib             # fitted classifier bundles
    │   ├── selected.json        # model used by the app
    │   ├── experiments.json
    │   └── lock.json            # locks geometry/slices/config before test
    └── results/
        ├── development_comparison.csv
        ├── test_metrics.csv
        ├── test_predictions.csv
        ├── test_roc.png
        ├── paired_auc_differences.csv   # full mode only
        ├── evaluation_notes.json
        ├── encoder_timings.csv
        ├── end_to_end_timings.csv
        └── benchmark_environment.json
```
Note: your deck's module plan names `extract_features.py`, `train_evaluate.py`, `app.py`. The package combines these into `pipeline.py` subcommands plus `app.py`. They do the same jobs. Mention that on the slide if asked.

---

## 4. End-to-end flow (what happens to the data)

```
OASIS-1 session (.hdr/.img, T88 masked, 176×208×176)
   │  match ID → CSV (Age, CDR)
   ▼
Cohort filter: age ≥ 60, valid CDR, earliest session per participant
   │  label: CDR=0 → 0 ; CDR>0 → 1
   ▼
Stratified 80/20 participant split (seed fixed) + CV folds inside development
   │
   ▼
Verify orientation/geometry (manual, once) → choose 16 axial slice indices
   │  nested subsets: 4 ⊂ 8 ⊂ 16
   ▼
Preprocess each slice: clip 1–99 percentile in brain → scale [0,1] →
pad to square → resize 224×224 → repeat to 3 channels → ImageNet normalise
   │
   ├──► ResNet-18 (frozen)  → 512-d per slice ─┐
   └──► ViT-B/16 (frozen)   → 768-d per slice ─┤  cached as .npz
                                               ▼
Mean-pool over the participant's slices → one row per participant
   CNN: N×512 | ViT: N×768 | Fusion: concat N×1280
   ▼
StandardScaler (fit inside each training fold only) + L2 logistic regression
   C ∈ {0.001, 0.01, 0.1, 1}, chosen by CV ROC-AUC on development only
   ▼
Select best model by development CV AUC → LOCK config → extract test features
   ▼
Evaluate ONCE on held-out test → metrics + bootstrap CIs + ROC
   ▼
Streamlit demo: pick local sample → slices, predicted CDR group, score, time
```

Key principle: **one participant = one example.** Slices are never treated as separate patients (that would leak data and inflate results).

---

## 5. Step-by-step commands

Run everything from the project folder. Throughout, `python` means the virtual environment's Python.

### Step 0: Get the data
1. Complete OASIS registration / data-use agreement.
2. Download `oasis_cross-sectional.csv` and all numbered raw-data discs.
3. Extract every archive, including nested session zips, keeping the folder structure. Keep `.hdr` and `.img` together.
4. If the clinical file is only XLS, export it to CSV, keeping ID, Age, CDR and leaving missing values blank.

*Optional space saver (untested helper, adjust to your actual archive layout):* after extraction, copy only the needed pair per session.
```python
# trim_oasis.py : copies only *_t88_masked_gfc.hdr/.img into a slim folder
import shutil, pathlib
src = pathlib.Path("D:/OASIS1_full")
dst = pathlib.Path("D:/OASIS1")
for f in src.glob("**/PROCESSED/T88_111/*_t88_masked_gfc.*"):
    if f.suffix in (".hdr", ".img"):
        target = dst / f.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
```
Copy the CSV into the slim folder too.

### Step 1: Environment (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip freeze > requirements-lock.txt
.\.venv\Scripts\python.exe -c "import torch, torchvision, nibabel, sklearn; print('Imports OK'); print('CUDA:', torch.cuda.is_available())"
```
`CUDA: False` is expected for the CPU build. Use Python 3.11 if possible. For an NVIDIA GPU, pick the matching torch/torchvision command from pytorch.org. Do not mix versions.

### Step 2: Edit `config.yaml`
```yaml
data_root: 'D:/OASIS1'
metadata_csv: 'D:/OASIS1/oasis_cross-sectional.csv'
output_dir: 'outputs'
device: cpu
batch_size: 2        # use 1 if RAM is limited
image_glob: '**/PROCESSED/T88_111/*_t88_masked_gfc.hdr'
```
Default metadata columns: `ID`, `Age`, `CDR`. IDs must look like `OAS1_0001_MR1`.

### Step 3: Build the participant table and split
```powershell
python pipeline.py prepare
```
Done when you see class counts for development and test, and a manual spot-check shows the right MRI path matches the right CDR. This command will not overwrite an existing manifest. If you find a real metadata error, fix it and use a new `output_dir`.

### Step 4: Verify orientation and pick slices (manual, label-blind)
```powershell
python pipeline.py inspect
python pipeline.py inspect --subject OAS1_XXXX
```
Open `outputs/qc/..._planes.png` and `..._geometry.json` for several **development** participants. Then set in `config.yaml`: `axial_axis`, `expected_shape`, `expected_zooms`, `expected_affine` (copy the full 4×4 `effective_affine`), `notes`, and finally `verified: true`.

Generate 16 slice positions (example range only, confirm with your previews):
```powershell
python -c "import numpy as np; print(np.linspace(40,130,16,dtype=int).tolist())"
```
Paste into `slice_indices`. The 4- and 8-slice subsets are positions inside this list. Never choose slices by looking at test results.

### Step 5: Five-participant CNN pilot (measures your real speed)
```powershell
python pipeline.py extract --encoder cnn --limit 5
```
First run downloads ResNet-18 weights (internet needed). Done when 5 files exist in `outputs/features/cnn/` and the log reports shape `(16, 512)`. Estimate: remaining participants × measured median seconds per participant. Do not guess hours.

### Step 6: Choose your path BEFORE final evaluation

**Path A, deadline-first baseline (recommended for 28 Oct):**
```powershell
python pipeline.py extract --encoder cnn
python pipeline.py train
python pipeline.py extract --encoder cnn --split test
python pipeline.py evaluate
```

**Path B, full comparison (only if the ViT pilot fits your time):**
```powershell
python pipeline.py extract --encoder vit --limit 5      # time it first
python pipeline.py extract --encoder cnn
python pipeline.py extract --encoder vit
python pipeline.py train --full
python pipeline.py extract --encoder cnn --split test
python pipeline.py extract --encoder vit --split test
python pipeline.py evaluate
```
Use one training path per output directory. To go from baseline to full later, create `config_full.yaml` with a new `output_dir` and pass `--config config_full.yaml` to every subcommand. Verify the participant assignments stayed the same.

### Step 7: Benchmark real cost
```powershell
python pipeline.py benchmark --subjects 2 --repeats 2
```
Produces `encoder_timings.csv`, `end_to_end_timings.csv`, `benchmark_environment.json`. This feeds the "accuracy vs computation" part of your research question.

### Step 8: Predict one scan directly
```powershell
python pipeline.py predict --image "D:/OASIS1/ACTUAL/PATH/scan_t88_masked_gfc.hdr"
```

### Step 9: Demo app
```powershell
python -m streamlit run app.py
```
Pick a local sample → Analyse MRI → shows slices, predicted CDR group, score, elapsed time. Before final evaluation, held-out participants are hidden from the selector. The score is **not** a clinically validated probability.

---

## 6. What is trained vs frozen

| Component | Trained? | Notes |
|---|---|---|
| ResNet-18 | No (frozen, eval mode) | Used only as a feature extractor |
| ViT-B/16 | No (frozen, eval mode) | Most expensive stage on CPU |
| StandardScaler | Yes, inside each training fold | Prevents leakage |
| Logistic regression | Yes | L2, class-balanced, C tuned over {0.001, 0.01, 0.1, 1} |
| Age-only baseline | Yes | Tests how much signal is just age. Not the app's model. |

Pooling: `X_cnn = mean over slices (N×512)`, `X_vit = mean over slices (N×768)`, `X_fusion = concat(X_cnn, X_vit) (N×1280)`.

---

## 7. The nine configurations + baseline

| Representation | 4 slices | 8 slices | 16 slices |
|---|---|---|---|
| CNN only (512-d) | Config 1 | Config 2 | Config 3 |
| ViT only (768-d) | Config 4 | Config 5 | Config 6 |
| Fusion (1280-d) | Config 7 | Config 8 | Config 9 |

Plus: age-only logistic regression on the same folds. Model selection for the app = highest development CV AUC (ties prefer simpler representation and fewer slices). That CV score is **not** an unbiased final estimate. Only the held-out test is.

---

## 8. Rules that protect your results (do not break these)

1. Split by **participant**, never by slice.
2. One session per participant (earliest eligible).
3. Fit scaler/classifier only on training folds.
4. Choose slices, geometry and hyperparameters using **development data only**.
5. Do **not** use CDR, MMSE or other clinical scores as image-model inputs. CDR is the label, and MMSE is closely tied to it. Age is used only in the age-only baseline.
6. Touch the test set once: extract test features only after `train` has locked the config, then run `evaluate` once.
7. Do not reshuffle the split or remove hard test cases to improve accuracy. Poor performance is a finding to report.
8. Do not update packages mid-study. Keep `requirements-lock.txt`.
9. Report numbers only from your own output CSVs.

---

## 9. Evaluation outputs and what they mean

- `test_metrics.csv`: ROC-AUC, balanced accuracy, sensitivity, specificity, F1, confusion counts, with stratified participant-bootstrap 95 % CIs.
- `paired_auc_differences.csv`: fusion vs each encoder (full mode only), paired resampling.
- `test_roc.png`: ROC curves.
- CIs describe uncertainty for this test set and fitted models. They do not prove generalisation to other scanners or hospitals. Comparisons are descriptive (no multiple-testing correction).
- Similar point estimates do **not** prove two models are equivalent.

---

## 10. Schedule (26 → 28 Oct 2026)

| Date | Priority | Evidence to save |
|---|---|---|
| 26 Oct | Access/download data, install env, `prepare`, `inspect`, verify geometry, choose slices, CNN pilot | Participant counts, QC montage, pilot timing |
| 27 Oct (session 1) | Full CNN extraction, `train`, test extraction, `evaluate` | Result CSVs, ROC plot |
| 27 Oct (session 2) | Run app, benchmark, slides, screen recording | Working demo, limitations, next steps |
| 28 Oct | Rehearse and present | Local app + backup screenshots/video |

Only add ViT/full mode if its measured pilot time leaves room for testing and rehearsal. Do not change the architecture the last evening.

---

## 11. What to say and not say at the review

**Safe summary:**
"We match OASIS-1 MRI scans to recorded CDR labels and split participants before training. A frozen pretrained ResNet-18 converts each selected MRI slice into features. We average features across slices and train a regularised logistic regression to predict the CDR group, evaluated on held-out participants and demonstrated via a local interface. The full study compares CNN, ViT and fused features at 4, 8 and 16 slices, against computational cost."

- If only the CNN baseline is complete, call it your completed first implementation and ViT/fusion the next stage. Do not claim fusion results before running them.
- If evaluation is not done, show verified data/QC progress, not invented accuracy.
- State limitations: single dataset, ImageNet-pretrained frozen encoders, 2D slices, mean pooling discards slice order, CDR group ≠ diagnosis, small test set.
- Out of scope: encoder fine-tuning, 3D CNNs, learned attention pooling, external validation, any clinical claim.

---

## 12. Troubleshooting

| Symptom | Action |
|---|---|
| No MRI files found | Extract nested archives; check `data_root` and the real folder layout |
| Expected original session IDs | Check the CSV ID column; do not strip `_MR1` |
| Geometry not verified | Finish `inspect`; fill axis, affine, notes, slices; set `verified: true` |
| Different shape/affine | Stop and verify the image product; do not suppress validation |
| Empty/near-empty slice | Re-pick slice range from development previews |
| Slow CPU extraction | `batch_size: 1`, close other apps, finish CNN first |
| CUDA unavailable | CPU mode is valid |
| Pretrained download fails | Check internet/proxy, retry; finished features are kept |
| Training already locked | Keep results; use a new `output_dir` for a different protocol |
| Stale cache / changed lock | Restore matching config/environment |

---

## 13. Deck-to-package alignment (checklist)

| Deck item | Package support |
|---|---|
| 80/20 stratified participant split, seed fixed | `prepare` |
| 16 slices with nested 8/4 | `slice_indices` in config |
| ResNet-18 512-d, ViT-B/16 768-d, fusion 1280-d | `extract`, `train --full` |
| Nine configs + age baseline | `train --full` |
| Bootstrap CIs, paired resampling | `evaluate` |
| Extraction time, memory | `benchmark` |
| Local demo | `app.py` |
| Optional noise/blur stress test | Not in the package as described. Treat as optional/future. |
| Deck status: "OASIS-1 access in progress", "no model trained yet" | Keep this honest on slides until results exist |

---

## 14. References

1. OASIS-1: https://sites.wustl.edu/oasisbrains/home/oasis-1/
2. OASIS cross-sectional fact sheet (Marcus et al., 2007, J. Cogn. Neurosci. 19:1498–1507)
3. Hybrid CNN + transformer for AD stages, Sci. Reports, 2025: https://doi.org/10.1038/s41598-025-01072-5
4. Training ViT with limited data for AD, MICCAI 2024: https://papers.miccai.org/miccai-2024/796-Paper2724.html
5. Multi-slice attention fusion network, BMC Med. Imaging, 2024: https://doi.org/10.1186/s12880-024-01429-8
6. CNN and transformer hybrid, arXiv:2601.15202 (preprint)
7. He et al., ResNet, CVPR 2016: https://arxiv.org/abs/1512.03385
8. Dosovitskiy et al., ViT, ICLR 2021: https://arxiv.org/abs/2010.11929
9. scikit-learn common pitfalls (leakage): https://scikit-learn.org/stable/common_pitfalls.html
10. NiBabel image orientation: https://nipy.org/nibabel/image_orientation.html

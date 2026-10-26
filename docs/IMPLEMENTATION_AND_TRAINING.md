# Implementation and training plan

Academic prototype. Predicts the recorded CDR group (0 vs >0) from a brain MRI. Not a diagnosis.

## 1. What the data actually looks like (checked on this machine)

| Item | Finding |
|---|---|
| Scan to use | `discN/discN/OAS1_xxxx_MRy/PROCESSED/MPRAGE/T88_111/*_t88_masked_gfc.hdr` + `.img` |
| Note | There is an extra `MPRAGE/` level versus the guides. `n4` in the name varies, so the glob matches only the suffix: `**/T88_111/*_t88_masked_gfc.hdr` |
| Image | 176x208x176, 1 mm, atlas-aligned, skull removed, 16-bit big-endian Analyze 7.5 (fact sheet Table 1) |
| Not used | `RAW/`, `SUBJ_111/`, `*_t88_gfc` (has skull), `FSL_SEG/` (labels), `t4_files/`, `.txt`/`.xml` (same info as the spreadsheet) |
| Layout | No need to merge discs. `data_root: '..'` plus a recursive glob finds every session; each is joined to the spreadsheet by session ID |
| Spreadsheet | `oasis_cross-sectional-5708aa0a98d82080.xlsx`, 436 rows: ID, M/F, Hand, Age, Educ, SES, MMSE, CDR, eTIV, nWBV, ASF, Delay. Under-60 CDR is `N/A` |
| Full cohort (age >= 60) | 198: CDR 0 = 98, 0.5 = 70, 1 = 28, 2 = 2 (matches fact sheet Table 2) |
| On disk today | Discs 1-6 extracted = 223 scans, of which **94 are eligible** (48 CDR 0, 46 CDR > 0). Disc 7 is still downloading (`.crdownload`); remaining discs are needed to reach 198 |

Do not run the real study on the partial 94. Use them for the pipeline pilot, QC and slice choice, then run `prepare` once all discs are extracted.
Convert the xlsx to `data/metadata/oasis_cross-sectional.csv` (needs `openpyxl` in the venv).

## 2. What is trained, what is not

| Component | Trained? | Detail |
|---|---|---|
| ResNet-18 (ImageNet) | No, frozen | Head removed, eval mode, 512-d per slice |
| ViT-B/16 (ImageNet) | No, frozen | Head removed, eval mode, 768-d per slice; most expensive stage on CPU |
| StandardScaler | Yes, per training fold | Fitted inside the sklearn Pipeline, never on all data |
| Logistic regression | Yes | L2, `class_weight=balanced`, `C` in {0.001, 0.01, 0.1, 1}, `lbfgs`, `max_iter=5000` |
| Age-only logistic regression | Yes | Baseline on the same split and folds; not the app model |

Nothing deep is trained. The "training" is a small linear classifier on cached, mean-pooled features, which is why it is cheap and why the cost is in feature extraction.

## 3. How training works

1. **Cohort:** age >= 60, CDR in {0, 0.5, 1, 2}, earliest session per participant (drops MR2 repeats). Label 0 = CDR 0, label 1 = CDR > 0.
2. **Split:** stratified 80/20 by participant, seed 42, saved to `participants.csv`. Five stratified CV folds on the development part, saved and reused by every experiment.
3. **Preprocess each chosen axial slice:** clip to 1st-99th percentile of brain voxels, scale to [0,1], pad to square, resize to 224x224, repeat to 3 channels, ImageNet mean/std.
4. **Extract:** 16 slices per participant through each frozen encoder, cached per participant (`outputs/features/<enc>/<id>.npz`, shapes 16x512 and 16x768). Budgets 4 and 8 are position subsets of the 16 (4 = [0,4,11,15], 8 = [0,2,4,6,9,11,13,15]).
5. **Pool:** mean over the budget's slices per encoder; fusion = concat (512 + 768 = 1280).
6. **Fit:** `GridSearchCV` over `C` using the saved folds and ROC-AUC, refit on all development participants.
7. **Select** by development CV AUC (ties: simpler representation, fewer slices), write `lock.json`.
8. **Evaluate once** on the held-out ~20%: ROC-AUC, balanced accuracy, sensitivity, specificity, F1, confusion matrix, stratified participant bootstrap 95% CIs, paired AUC differences for fusion vs each encoder.
9. **Benchmark** uncached load, preprocess, encode and predict time per representation and budget, plus memory.

Experiments: CNN, ViT, Fusion x 4/8/16 slices = 9, plus the age baseline. Baseline path runs only CNN-16 + age.

## 4. Best way to implement, in order

| # | Build | Command | Done when |
|---|---|---|---|
| 1 | `dementia/data.py` | `python run.py prepare` | One row per participant, exclusions logged, no train/test overlap, counts per class printed |
| 2 | `dementia/preprocess.py` | `python run.py qc` | Slice montages upright, brain in view (orientation and slices are preset in `config.yaml`) |
| 3 | `dementia/features.py` | `python run.py extract --limit 5` | Files of shape (16,512)/(16,768); measured seconds per participant |
| 4 | `dementia/train.py` | `python run.py train` | Dev CV table, saved pipelines, `lock.json` |
| 5 | `dementia/evaluate.py` | `run.py extract --split test`, `run.py evaluate` | One prediction per test participant, CIs, ROC plot |
| 6 | `dementia/predict.py`, `app.py` | `run.py benchmark`, `streamlit run app.py` | Timings CSV; app score equals offline score |

Tests (synthetic only, `python -m pytest`): participant disjointness, class presence in each fold, cohort rules, pooling shapes, model selection.

## 5. Rules that protect the result

- Split by participant, never by slice; one session per participant.
- Choose slices, geometry and `C` from development data only; touch the test set once.
- Never feed CDR, MMSE, filenames or IDs to the image model. Age is used only in the age baseline.
- Do not reshuffle the split or drop hard cases after seeing results. Use a new `output_dir` for a new protocol.
- Report numbers only from your own output CSVs.

## 6. Risks and limits

- Test set will be about 40 participants, so confidence intervals will be wide.
- Only 94 eligible scans are extracted so far; results need all discs.
- Orientation of the Analyze files must be confirmed visually before choosing slice indices.
- ViT on CPU may be slow; measure with `--limit 5` before committing.
- 2D slices, ImageNet-pretrained frozen encoders and mean pooling are limitations. CDR group is not a confirmed diagnosis.

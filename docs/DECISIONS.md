# Design decisions

1. **Package, not one script.** Source guides use a single `pipeline.py`; here it is split into `src/dementia/*` modules
   (one responsibility each, matching the execution guide's module table) behind one `dementia` CLI. Same subcommands.
2. **Data stays outside git.** Scans live in `../oasis_cross-sectional_disc*`; the clinical CSV goes in `data/metadata/`.
   `.gitignore` blocks image, feature, model and archive files.
3. **Config-driven, locked.** `configs/base.yaml` (CNN-16 + age baseline, deadline path) and `configs/full.yaml`
   (9 configs + baseline, separate `output_dir`). `train` writes `lock.json`; test features/eval only after lock.
4. **Single preprocessing function** shared by training, benchmark, CLI predict and app.
5. **Features cached per participant** at 16 slices; budgets 4/8 derived by position subsets; pooled then concatenated for fusion.
6. **Only scaler + logistic regression are trained**, inside one sklearn Pipeline on fixed saved folds.
7. **Evaluation once**; bootstrap CIs; paired AUC differences in full mode.
8. **Tests are synthetic** (split disjointness, shape/pooling, leakage guards).

Open items: convert the clinical `.xlsx` to CSV; disc7 download unfinished (`.crdownload`); geometry/slice indices need manual verification.

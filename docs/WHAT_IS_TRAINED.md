# What is actually trained, and what are the 4 models?

## 1. Short answer
- **Pretrained, not trained here:** ResNet-18 and ViT-B/16 (ImageNet weights). They are frozen and only turn an MRI slice into a list of numbers (features).
- **Trained here:** a small **logistic regression** (with a StandardScaler) on top of those features. This is the only learning on our data.
- This is **transfer learning by feature extraction** (also called a "linear probe"). No gradient step ever touches the CNN or the ViT, which is why the whole thing runs on a laptop.

```
MRI slice -> [frozen ResNet-18 / ViT-B/16] -> feature vector -> [StandardScaler -> LogisticRegression]  -> P(CDR > 0)
             pretrained, NOT trained            fixed numbers      TRAINED on our dev participants
```

## 2. The 4 models in `outputs/models/` (default `review` suite)

| File | Input to the classifier | Size | What it is |
|---|---|---|---|
| `cnn_16.joblib` | ResNet-18 features, averaged over 16 slices | 512 | CNN only |
| `vit_16.joblib` | ViT-B/16 features, averaged over 16 slices | 768 | ViT only |
| `fusion_16.joblib` | CNN and ViT features joined side by side | 512 + 768 = 1280 | the project's main idea |
| `age_only.joblib` | the participant's age | 1 | baseline: checks whether age alone explains the result |

Each `.joblib` file holds a scikit-learn `Pipeline` (`StandardScaler` then `LogisticRegression`) plus its name and slice positions.
It does **not** contain ResNet or ViT. Those are re-downloaded from torchvision when needed (`get_encoder`).
The first three are "MRI models"; `age_only` is never used in the app or for picking the best model.
With `python run.py train --suite full` there are 9 MRI models (cnn/vit/fusion x 4/8/16 slices) + `age_only`.

## 3. Concepts used
| Concept | Meaning here |
|---|---|
| Transfer learning / frozen encoder | Reuse ImageNet-trained networks as fixed feature extractors |
| Feature pooling | Mean of a person's 16 slice vectors gives one vector per person |
| Feature-level fusion | Concatenate CNN and ViT vectors (no learned fusion layer) |
| Logistic regression (L2, class-balanced) | Linear classifier; `class_weight="balanced"` stops the larger class dominating |
| StandardScaler | Rescales each feature to mean 0 / std 1, refitted inside every CV fold so nothing leaks |
| Cross-validation | 5 folds by participant on the development set; picks the regularisation strength `C` by ROC-AUC |
| Hold-out test | 20% of participants never seen until `evaluate`, which runs once after the models are locked |

## 4. Where it happens in the code

### 4.1 Pretrained encoders, frozen: [dementia/features.py](../dementia/features.py) `get_encoder`
```python
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Identity()  # 512-d
...
model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
model.heads = nn.Identity()  # 768-d
model.eval().requires_grad_(False).to(device)   # frozen: no gradients, eval mode
```
The ImageNet classification heads are replaced with `Identity`, so the network outputs features instead of 1000 class scores.
`requires_grad_(False)` is what makes "not trained" literally true.

### 4.2 MRI to network input: [dementia/preprocess.py](../dementia/preprocess.py)
```python
lo, hi = np.percentile(vol[mask], percentiles)           # normalise_volume: brain voxels only
out = np.clip((vol - lo) / (hi - lo), 0, 1)
...
np.rot90(np.take(vol, i, axis=axis), rot90_k)            # extract_slices: 16 axial slices, upright
...
t = t.clamp(0, 1).repeat(1, 3, 1, 1)                     # to_model_input: grey -> 3 channels
return (t - mean) / std                                  # ImageNet normalisation, 224x224
```

### 4.3 Extract features once and cache them: [dementia/features.py](../dementia/features.py) `encode` / `extract`
```python
with torch.inference_mode():
    out = [model(x[i : i + batch_size].to(device)).float().cpu().numpy() ...]
...
np.savez(path, features=feats, config_hash=cfg.feature_hash(encoder))   # outputs/features/<enc>/<id>.npz, (16, 512|768)
```
This is the slow step. Training afterwards reads these cached files and takes seconds.

### 4.4 Pool and fuse: [dementia/features.py](../dementia/features.py) `pool` / `build_matrix`
```python
def pool(feats, positions):
    return feats[:, list(positions), :].mean(axis=1)          # (N,16,d) -> (N,d)

def build_matrix(rep, positions, feats):
    return np.concatenate([pool(feats[e], positions) for e in ENCODERS_FOR[rep]], axis=1)   # fusion = 512+768
```

### 4.5 The actual training: [dementia/train.py](../dementia/train.py) `fit_model`
```python
cv = [(np.flatnonzero(folds != f), np.flatnonzero(folds == f)) for f in np.unique(folds)]
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=seed)),
])
return GridSearchCV(pipe, {"clf__C": c_grid}, cv=cv, scoring="roc_auc", refit=True).fit(X, y)
```
- `GridSearchCV` tries `C` in {0.001, 0.01, 0.1, 1} and scores each by 5-fold ROC-AUC on development participants.
- `refit=True` retrains the best `C` on all development participants. That fitted pipeline is what gets saved.

### 4.6 One model per experiment, plus age baseline: [dementia/train.py](../dementia/train.py) `train`
```python
jobs = [(f"{r}_{b}", r, b, cfg.positions(b)) for r, b in cfg.experiments(suite)]   # cnn_16, vit_16, fusion_16
jobs.append(("age_only", "age", 0, None))                                          # baseline
for name, rep, budget, pos in jobs:
    X = dev[["age"]].to_numpy(float) if rep == "age" else build_matrix(rep, pos, feats)
    search = fit_model(X, y, folds, cfg["C_grid"], cfg["seed"])
    joblib.dump({... "pipeline": search.best_estimator_}, cfg.out("models", f"{name}.joblib"))
```

### 4.7 Choosing the model for the app and locking: [dementia/train.py](../dementia/train.py)
```python
mri = [r for r in results if r["representation"] in REP_RANK]            # age_only is excluded
return min(mri, key=lambda r: (-round(r["cv_auc"], 9), REP_RANK[r["representation"]], r["budget"]))
```
Highest development CV AUC wins; ties go to the simpler model. `lock.json` then stores a hash of `participants.csv`, the slice settings and the threshold, so nothing can be changed before the test.

### 4.8 Prediction for one new scan: [dementia/predict.py](../dementia/predict.py) `predict_volume`
```python
feat = np.concatenate([encode(get_encoder(e, device), x, device, cfg["batch_size"]).mean(0) for e in ENCODERS_FOR[model["representation"]]])
score = float(model["pipeline"].predict_proba(feat[None])[0, 1])
label = int(score >= cfg["threshold"])      # 0.5
```
Scan -> frozen encoder(s) -> mean over slices -> saved scaler + logistic regression -> score. It never reads CDR or age.

## 5. What the label and the score mean
- Label: CDR = 0 is class 0; CDR > 0 is class 1 (very mild to moderate dementia, as recorded).
- Score: the logistic regression's probability-like output for class 1. It is not a calibrated clinical probability. The threshold is a fixed 0.5.

## 6. Why this design
- With about 100-200 people, fine-tuning a 11M-parameter CNN or an 86M-parameter ViT would overfit. A linear model on frozen features has few parameters and is easy to validate.
- Because features are cached, all experiments (4/8/16 slices, CNN/ViT/fusion) cost seconds once extraction is done.
- It answers the research question cleanly: does adding ViT features to CNN features help, using the same classifier and same split?
- Limits: ImageNet features are not MRI-specific, slices are 2D, and the test set is small. Report only numbers from `outputs/results/`.

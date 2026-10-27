"""Local demo: streamlit run app.py   (uses the same predict_volume as run.py; never reads CDR to predict)."""
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from dementia.config import Config
from dementia.data import load_participants
from dementia.predict import load_model, mri_models, predict_volume
from dementia.train import check_lock

st.set_page_config(page_title="Dementia MRI demo", layout="wide")
st.title("MRI-based CDR-group prediction (OASIS-1)")
st.warning("Academic prototype; not clinically validated. The model score is not a probability of dementia.")

cfg = Config()
try:
    lock = check_lock(cfg)
    parts = load_participants(cfg)
except SystemExit as e:
    st.error(f"{e}  Follow README.md: prepare -> extract -> train first.")
    st.stop()

results = cfg.out("results")
evaluated = (results / "test_metrics.csv").exists()
models = mri_models(cfg)
name = st.sidebar.selectbox("MRI model", models, index=models.index(lock["selected"]))
st.sidebar.caption(f"Selected on development CV: {lock['selected']}. Changing this menu does not re-select.")
model = load_model(cfg, name)

mode = st.radio("Input", ["Local sample", "Upload processed MRI"], horizontal=True)
row, uploads = None, []
if mode == "Local sample":
    shown = parts if evaluated else parts[parts.split == "dev"]  # held-out people stay hidden until evaluated
    subject = st.selectbox("Participant", shown.subject_id)
    row = shown[shown.subject_id == subject].iloc[0]
else:
    st.caption("Upload a matching *_t88_masked_gfc .hdr + .img pair (or one .nii/.nii.gz in the same 176x208x176 atlas space).")
    uploads = st.file_uploader("MRI file(s)", type=["hdr", "img", "nii", "gz"], accept_multiple_files=True)

if st.button("Analyse MRI", type="primary"):
    try:
        with tempfile.TemporaryDirectory() as tmp:
            if row is not None:
                path = row.image_path
            else:
                names = [Path(f.name).name for f in uploads]
                for f, n in zip(uploads, names):
                    (Path(tmp) / n).write_bytes(f.getvalue())
                hdrs = [n for n in names if n.endswith(".hdr")]
                if len(names) == 1 and names[0].endswith((".nii", ".nii.gz")):
                    path = str(Path(tmp) / names[0])
                elif len(names) == 2 and len(hdrs) == 1 and Path(hdrs[0]).with_suffix(".img").name in names:
                    path = str(Path(tmp) / hdrs[0])
                else:
                    raise ValueError("Upload one .nii/.nii.gz, or one .hdr with its matching .img")
            with st.spinner("Running..."):
                r = predict_volume(path, cfg, model)
    except Exception as e:
        st.error(str(e))
        st.stop()
    c1, c2, c3 = st.columns(3)
    c1.metric("Research prediction", r["group"])
    c2.metric("Model score (CDR > 0)", f"{r['score']:.3f}")
    c3.metric("Time (s)", f"{r['seconds']:.2f}")
    st.caption("Shown slices are the preprocessed model inputs, not explanations of disease.")
    cols = st.columns(8)
    for i, (s, idx) in enumerate(zip(r["slices"], r["slice_indices"])):
        cols[i % 8].image(s, caption=f"axial {idx}", clamp=True)
    if row is not None:
        st.info(f"Recorded reference, shown after prediction: CDR = {row.cdr}")

with st.expander("Development CV results"):
    f = results / "dev_cv_results.csv"
    st.dataframe(pd.read_csv(f)) if f.exists() else st.write("Not available.")
if evaluated:
    with st.expander("Held-out test results"):
        st.dataframe(pd.read_csv(results / "test_metrics.csv"))
        st.image(str(results / "test_roc.png"))
        st.image([str(p) for p in sorted(results.glob("confusion_*.png"))], width=250)
if (results / "timings_summary.csv").exists():
    with st.expander("Measured prediction time and GPU memory"):
        st.dataframe(pd.read_csv(results / "timings_summary.csv"))
if (results / "cost_vs_auc.csv").exists():
    with st.expander("Accuracy vs cost (all configurations)"):
        st.dataframe(pd.read_csv(results / "cost_vs_auc.csv"))
        st.image([str(results / "cost_vs_auc.png"), str(results / "auc_vs_slices.png")])
        st.json((results / "report.json").read_text())

"""Streamlit demo. Calls dementia.predict.predict_volume only; never reads true CDR to predict."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import streamlit as st

from dementia.config import load_config
from dementia.data.manifest import load_participants
from dementia.predict import load_bundle, predict_volume

st.set_page_config(page_title="Dementia MRI research demo", layout="wide")
cfg = load_config(os.environ.get("DEMENTIA_CONFIG"))
st.title("MRI-based CDR-group prediction (OASIS-1)")
st.warning("Academic prototype; not clinically validated. The score is not a probability of dementia.")

evaluated = cfg.out("results", "evaluation_notes.json").exists()
try:
    parts = load_participants(cfg)
    bundle = load_bundle(cfg)
except SystemExit as e:
    st.error(f"{e}  Run prepare / extract / train first (see README).")
    st.stop()

if not evaluated:
    parts = parts[parts.split == "dev"]  # held-out participants stay hidden until evaluation is done
subject = st.selectbox("Sample (local, already in the required processed format)", parts.subject_id)
row = parts[parts.subject_id == subject].iloc[0]
st.caption(f"Model: {bundle['name']} | slices: {bundle['slice_indices']}")

if st.button("Analyse MRI"):
    with st.spinner("Processing..."):
        res = predict_volume(row.image_path, cfg, bundle=bundle)
    c1, c2, c3 = st.columns(3)
    c1.metric("Research prediction", res["group"])
    c2.metric("Model score", f"{res['score']:.3f}")
    c3.metric("Processing time (s)", f"{res['seconds']:.2f}")
    st.caption(
        f"preprocess {res['seconds_preprocess']:.2f}s | encode {res['seconds_encode']:.2f}s | classify {res['seconds_classify']:.3f}s"
    )
    cols = st.columns(4)
    for i, (sl, idx) in enumerate(zip(res["slices"], res["slice_indices"])):
        cols[i % 4].image(sl, caption=f"axial index {idx}", clamp=True)
    st.write(f"Recorded reference (shown after prediction): CDR = {row.cdr}")

results = cfg.out("results")
with st.expander("Development comparison (cross-validation)"):
    f = results / "development_comparison.csv"
    st.dataframe(pd.read_csv(f)) if f.exists() else st.write("Not available yet.")
with st.expander("Held-out test metrics"):
    f = results / "test_metrics.csv"
    st.dataframe(pd.read_csv(f)) if f.exists() else st.write("Run `dementia evaluate` first.")

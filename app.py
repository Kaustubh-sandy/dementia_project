"""Local demo: streamlit run app.py   (uses the same predict_volume as run.py; never reads CDR to predict)."""
import pandas as pd
import streamlit as st

from dementia.config import Config
from dementia.data import load_participants
from dementia.predict import load_model, predict_volume

st.set_page_config(page_title="Dementia MRI demo", layout="wide")
st.title("MRI-based CDR-group prediction (OASIS-1)")
st.warning("Academic prototype; not clinically validated. The model score is not a probability of dementia.")

cfg = Config()
try:
    parts = load_participants(cfg)
    model = load_model(cfg)
except SystemExit as e:
    st.error(str(e))
    st.stop()

results = cfg.out("results")
if not (results / "test_metrics.csv").exists():
    parts = parts[parts.split == "dev"]  # keep held-out participants hidden until evaluation is done

subject = st.selectbox("Local sample", parts.subject_id)
row = parts[parts.subject_id == subject].iloc[0]
st.caption(f"Model: {model['name']}")

if st.button("Analyse MRI"):
    with st.spinner("Running..."):
        r = predict_volume(row.image_path, cfg, model)
    c1, c2, c3 = st.columns(3)
    c1.metric("Research prediction", r["group"])
    c2.metric("Model score", f"{r['score']:.3f}")
    c3.metric("Time (s)", f"{r['seconds']:.2f}")
    cols = st.columns(8)
    for i, (s, idx) in enumerate(zip(r["slices"], r["slice_indices"])):
        cols[i % 8].image(s, caption=f"axial {idx}", clamp=True)
    st.info(f"Recorded reference, shown after prediction: CDR = {row.cdr}")

for title, name in [("Development CV results", "dev_cv_results.csv"), ("Held-out test metrics", "test_metrics.csv"), ("Timings", "timings.csv")]:
    with st.expander(title):
        f = results / name
        if f.exists():
            st.dataframe(pd.read_csv(f))
        else:
            st.write("Not available yet.")

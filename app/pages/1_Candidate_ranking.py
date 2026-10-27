import pandas as pd
import streamlit as st

from api_client import get_models, list_signals, model_label, require_api
from astro_style import apply_style

st.set_page_config(page_title="Candidate ranking", page_icon="🔭", layout="wide")
apply_style()
require_api()
st.title("Predictions and candidate ranking")

models = get_models()
labels = {m["model_id"]: model_label(m["name"], m["feature_set"]) for m in models}
ids = list(labels)
default_id = next((m["model_id"] for m in models
                   if m["name"] == "random_forest" and m["feature_set"] == "all"), ids[0])

model_id = st.sidebar.selectbox("Model", ids, index=ids.index(default_id),
                                format_func=lambda i: labels[i])
group = st.sidebar.radio("Signals", ["Unresolved candidates", "Labeled signals"])
min_prob = st.sidebar.slider("Minimum probability", 0.0, 1.0, 0.5, 0.05)
limit = st.sidebar.selectbox("Rows to show", [25, 50, 100, 200, 500], index=1)

kind = "candidate" if group == "Unresolved candidates" else "labeled"
result = list_signals(kind=kind, model_id=model_id,
                      min_prob=min_prob if min_prob > 0 else None,
                      limit=limit, sort="prob_desc")
df = pd.DataFrame(result["signals"])

st.write(f"**{result['total']}** signals with probability of at least {min_prob:.2f} "
         f"({labels[model_id]}); showing the top {len(df)}.")
if df.empty:
    st.info("No signals match these filters.")
    st.stop()

table = pd.DataFrame({
    "TIC ID": df["tic_id"].astype(str),
    "TOI": df["toi"].map(lambda x: f"{x:.2f}"),
    "Probability": df["probability"].round(3),
    "Period (d)": df["period"].round(3),
    "Depth (ppt)": df["depth_obs_ppt"].round(2),
    "SNR": df["snr"].round(1),
    "Tmag": df["tmag"].round(2),
})
if kind == "labeled":
    table.insert(2, "Disposition", df["disposition"])
    table["Split"] = df["split"]
st.dataframe(table, hide_index=True)

st.subheader("Open a signal in the light curve explorer")
names = {r.signal_id: f"TIC {r.tic_id} | TOI {r.toi:.2f} | probability {r.probability:.3f}"
         for r in df.itertuples()}
chosen = st.selectbox("Signal", list(names), format_func=lambda s: names[s])
if st.button("Open in explorer"):
    tic = int(df.loc[df["signal_id"] == chosen, "tic_id"].iloc[0])
    st.session_state["signal_id"] = chosen
    st.session_state["tic_prefill"] = str(tic)
    st.switch_page("pages/2_Light_curve_explorer.py")

st.caption(
    "Candidates are unresolved TOIs without ground truth: the ranking says which signals "
    "look most like confirmed planets, not which ones are planets. The candidates are "
    "fainter and noisier than the signals the models were trained on, so scores may be "
    "less reliable for them."
)
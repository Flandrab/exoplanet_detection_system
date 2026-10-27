import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from api_client import (ApiError, fmt, get_lightcurve, get_signal, list_signals,
                        model_label, require_api)
from astro_style import apply_style, show_chart

st.set_page_config(page_title="Light curve explorer", page_icon="🔭", layout="wide")
apply_style()
require_api()
st.title("Light curve explorer")

GROUPS = {
    "Unresolved candidates": {"kind": "candidate"},
    "Labeled planets": {"kind": "labeled", "label": 1},
    "Labeled false positives": {"kind": "labeled", "label": 0},
}

st.sidebar.header("Choose a signal")
tic_text = st.sidebar.text_input("Search by TIC ID (optional)",
                                 value=st.session_state.get("tic_prefill", "")).strip()
group = st.sidebar.radio("Group", list(GROUPS))
if tic_text:
    st.sidebar.caption("Searching by TIC ID; the group is ignored.")
    if not tic_text.isdigit():
        st.sidebar.warning("The TIC ID must be a number.")
        st.stop()
    result = list_signals(tic_id=int(tic_text), limit=50)
else:
    result = list_signals(limit=500, sort="prob_desc", **GROUPS[group])

signals = result["signals"]
if not signals:
    st.info("No signals found.")
    st.stop()

ids = [s["signal_id"] for s in signals]
by_id = {s["signal_id"]: s for s in signals}
start = st.session_state.get("signal_id")
signal_id = st.sidebar.selectbox(
    "Signal", ids, index=ids.index(start) if start in ids else 0,
    format_func=lambda i: f"TIC {by_id[i]['tic_id']} | TOI {by_id[i]['toi']:.2f}")

detail = get_signal(signal_id)
sig, feats = detail["signal"], detail["features"] or {}

st.subheader(f"TIC {sig['tic_id']} | TOI {sig['toi']:.2f}")
cols = st.columns(6)
cols[0].metric("Disposition", sig["disposition"] or "n/a")
cols[1].metric("Period (d)", fmt(sig["period"], 3))
cols[2].metric("Duration (h)", fmt(sig["duration_h"], 1))
cols[3].metric("Depth (ppt)", fmt(feats.get("depth_obs_ppt")))
cols[4].metric("SNR", fmt(feats.get("snr"), 1))
cols[5].metric("Tmag", fmt(sig["tmag"]))

if sig["kind"] == "labeled":
    truth = "a planet" if sig["label"] == 1 else "a false positive"
    st.info(f"Known label: {truth} (disposition {sig['disposition']}, split: {sig['split']}). "
            "The probabilities below were produced by models that never saw this star.")
else:
    st.info("Unresolved candidate: there is no ground truth. The probabilities are a ranking, "
            "not a verdict.")

preds = detail["predictions"]
if preds:
    fig = go.Figure(go.Bar(
        x=[p["probability"] for p in preds],
        y=[model_label(p["name"], p["feature_set"]) for p in preds],
        orientation="h",
        marker=dict(color=[p["probability"] for p in preds], cmin=0, cmax=1,
                    colorscale=[[0, "#3b4a8c"], [1, "#7be0b0"]])))
    fig.update_xaxes(range=[0, 1], title_text="Probability of being a planet")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    show_chart(fig, key="pred_chart")

try:
    lc = get_lightcurve(signal_id)
except ApiError as e:
    st.warning(f"No light curve available for this signal ({e}).")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Light curve", "Folded transit", "Binned views"])

with tab1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Normalized flux", "Detrended flux"))
    fig.add_trace(go.Scatter(x=lc["raw"]["time"], y=lc["raw"]["flux"], mode="markers",
                             marker=dict(size=2, color="#7aa2ff")), row=1, col=1)
    fig.add_trace(go.Scatter(x=lc["flattened"]["time"], y=lc["flattened"]["flux"],
                             mode="markers", marker=dict(size=2, color="#7be0b0")), row=2, col=1)
    for t in lc["transit_times"]:
        for row in (1, 2):
            fig.add_vline(x=t, line_dash="dot", line_color="#ffb86b", opacity=0.5, row=row, col=1)
    fig.update_xaxes(title_text="Time (BTJD)", row=2, col=1)
    fig.update_layout(height=620, showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
    show_chart(fig, key="lc_chart")
    st.caption("Orange dotted lines mark the transit times predicted by the catalog period "
               "and epoch. Long series are thinned to at most 4000 points.")

with tab2:
    period = lc["period"]
    dur_d = lc["duration_h"] / 24.0
    low, high = min(0.05, period / 4), period / 2
    default = float(min(max(3 * dur_d, low), high))
    half = st.slider("Half window around the transit (days)", float(low), float(high),
                     default, 0.01)
    phase = np.array(lc["folded"]["phase"], dtype=float)
    flux = np.array(lc["folded"]["flux"], dtype=float)
    inside = np.abs(phase) <= half
    local_x = np.linspace(-lc["local_half_width_days"], lc["local_half_width_days"], 201)
    local_y = 1.0 + np.array(lc["local_view"], dtype=float) / 1000.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=phase[inside], y=flux[inside], mode="markers",
                             marker=dict(size=3, opacity=0.5, color="#7aa2ff"), name="folded data"))
    fig.add_trace(go.Scatter(x=local_x, y=local_y, mode="lines", name="binned local view",
                             line=dict(color="#ffb86b", width=3)))
    fig.update_xaxes(range=[-half, half], title_text="Time from transit center (days)")
    fig.update_yaxes(title_text="Normalized flux")
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
    show_chart(fig, key="fold_chart")
    st.caption(f"Folded on the catalog period of {period:.3f} days. "
               f"Sector {fmt(sig['sector'], 0)}.")

with tab3:
    glob = np.array(lc["global_view"], dtype=float)
    loc = np.array(lc["local_view"], dtype=float)
    fig = go.Figure(go.Scatter(x=np.linspace(-lc["period"] / 2, lc["period"] / 2, len(glob)),
                               y=glob, mode="lines"))
    fig.update_layout(title="Global view (whole orbit)", xaxis_title="Time from transit (days)",
                      yaxis_title="Flux (ppt)", height=340)
    show_chart(fig, key="global_chart")
    half_w = lc["local_half_width_days"]
    fig = go.Figure(go.Scatter(x=np.linspace(-half_w, half_w, len(loc)), y=loc, mode="lines"))
    fig.update_layout(title="Local view (around the transit)",
                      xaxis_title="Time from transit (days)", yaxis_title="Flux (ppt)",
                      height=340)
    show_chart(fig, key="local_chart")
    st.caption("Medians in fixed-size phase bins, produced by the same pipeline as the "
               "model features. Values are in parts per thousand relative to 1.")

with st.expander("Data returned by the API for this signal"):
    st.json(detail)
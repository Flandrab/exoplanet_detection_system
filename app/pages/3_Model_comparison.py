import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api_client import FEATURE_NAMES, MODEL_NAMES, get_compare, model_label, require_api
from astro_style import apply_style, show_chart

st.set_page_config(page_title="Model comparison", page_icon="🔭", layout="wide")
apply_style()
require_api()
st.title("Model comparison")

models = get_compare()
for m in models:
    m["label"] = model_label(m["name"], m["feature_set"])

first = models[0]
n_test = first["tn"] + first["fp"] + first["fn"] + first["tp"]
n_pos = first["fn"] + first["tp"]
st.caption(f"Held-out test set: {n_test} signals ({n_pos} planets) from stars never used in "
           "training, probability threshold 0.5. Cross-validation: 5 folds over the "
           "training stars.")

table = pd.DataFrame({
    "Model": [m["label"] for m in models],
    "CV PR-AUC": [f"{m['cv_pr_auc']:.3f} +/- {m['cv_pr_auc_std']:.3f}" for m in models],
    "Precision": [round(m["test_precision"], 3) for m in models],
    "Recall": [round(m["test_recall"], 3) for m in models],
    "F1": [round(m["test_f1"], 3) for m in models],
    "ROC-AUC": [round(m["test_roc_auc"], 3) for m in models],
    "PR-AUC": [round(m["test_pr_auc"], 3) for m in models],
})
st.subheader("Test-set metrics")
st.dataframe(table, hide_index=True)

st.subheader("Effect of the period features")
fig = go.Figure()
for fs in ("all", "no_period"):
    sel = [m for m in models if m["feature_set"] == fs]
    fig.add_trace(go.Bar(name=FEATURE_NAMES[fs],
                         x=[MODEL_NAMES[m["name"]] for m in sel],
                         y=[m["test_pr_auc"] for m in sel]))
fig.add_hline(y=n_pos / n_test, line_dash="dash", line_color="#8896c4",
              annotation_text="random classifier")
fig.update_layout(barmode="group", yaxis_title="Test PR-AUC", height=380,
                  margin=dict(l=0, r=0, t=10, b=0))
show_chart(fig, key="ablation_chart")

st.subheader("ROC and precision-recall curves")
all_labels = [m["label"] for m in models]
chosen = st.multiselect("Models to plot", all_labels,
                        default=[m["label"] for m in models if m["feature_set"] == "all"])
roc, pr = go.Figure(), go.Figure()
for m in models:
    if m["label"] in chosen and "roc" in m:
        roc.add_trace(go.Scatter(x=m["roc"]["fpr"], y=m["roc"]["tpr"], mode="lines",
                                 name=m["label"]))
        pr.add_trace(go.Scatter(x=m["pr"]["recall"], y=m["pr"]["precision"], mode="lines",
                                name=m["label"]))
roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="random",
                         line=dict(dash="dash", color="#8896c4")))
pr.add_hline(y=n_pos / n_test, line_dash="dash", line_color="#8896c4")
roc.update_layout(title="ROC curve", xaxis_title="False positive rate",
                  yaxis_title="True positive rate", height=420)
pr.update_layout(title="Precision-recall curve", xaxis_title="Recall",
                 yaxis_title="Precision", height=420)
left, right = st.columns(2)
with left:
    show_chart(roc, key="roc_chart")
with right:
    show_chart(pr, key="pr_chart")

st.subheader("Confusion matrix")
pick = st.selectbox("Model", all_labels)
chosen_model = next(m for m in models if m["label"] == pick)
z = [[chosen_model["tn"], chosen_model["fp"]], [chosen_model["fn"], chosen_model["tp"]]]
fig = go.Figure(go.Heatmap(
    z=z, x=["Predicted false positive", "Predicted planet"],
    y=["True false positive", "True planet"], text=z, texttemplate="%{text}",
    colorscale=[[0, "#101733"], [1, "#4f7cff"]], showscale=False,
    textfont=dict(color="#ffffff", size=20)))
fig.update_yaxes(autorange="reversed")
fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
show_chart(fig, key="confusion_chart")
st.caption("Cross-validation and test results agree closely, and removing the period "
           "features lowers every model a little without changing the ranking of the models.")
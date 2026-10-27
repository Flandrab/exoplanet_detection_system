import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api_client import get_summary, require_api
from astro_style import apply_style, hero, show_chart, step_card

st.set_page_config(page_title="TESS exoplanet vetting", page_icon="🔭", layout="wide")
apply_style()
require_api()

hero("TRANSITING EXOPLANET SURVEY SATELLITE",
     "Exoplanet candidate vetting with TESS data",
     "Machine learning models rank the dimming events seen by NASA's TESS mission by "
     "how planet-like they look. A high score is a ranking, not a discovery.")

summary = get_summary()
counts = summary["counts"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Stars", f"{counts['stars']:,}")
c2.metric("Signals", f"{counts['signals']:,}")
c3.metric("Models", f"{counts['models']:,}")
c4.metric("Predictions", f"{counts['predictions']:,}")

st.subheader("How it works")
s1, s2, s3 = st.columns(3)
with s1:
    step_card("1", "Observe", "TESS monitors the brightness of stars in 27-day sectors.")
with s2:
    step_card("2", "Detect a dip", "A planet crossing its star briefly dims the light.")
with s3:
    step_card("3", "Score it", "We detrend, fold and score each dip with machine learning models.")


def group_name(row):
    if row["kind"] == "candidate":
        return "Unresolved candidates"
    who = "planets" if row["label"] == 1 else "false positives"
    return f"Labeled {who} ({row['split']})"


def group_color(name):
    if "planets" in name:
        return "#7aa2ff"
    if "false positives" in name:
        return "#ffb86b"
    return "#d59bff"


groups = pd.DataFrame(summary["groups"])
groups["Group"] = groups.apply(group_name, axis=1)
groups = groups.sort_values("Group")

st.subheader("Signals in the database")
fig = go.Figure(go.Bar(x=groups["Group"], y=groups["n"],
                       marker_color=[group_color(g) for g in groups["Group"]]))
fig.update_layout(yaxis_title="Signals", height=380, margin=dict(l=0, r=0, t=10, b=0))
show_chart(fig, key="groups_chart")

st.subheader("Pages")
st.markdown(
    "- **Candidate ranking**: signals sorted by the probability of being a planet.\n"
    "- **Light curve explorer**: the light curve, the folded transit and the "
    "predictions of every model for one signal.\n"
    "- **Model comparison**: metrics, ROC and precision-recall curves and "
    "confusion matrices of the six models."
)
st.caption(
    "Labeled signals have a known disposition from the TOI catalog. Their probabilities "
    "come from models that never saw their star (out-of-fold or held-out)."
)
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODELS = {"random_forest_all": "RF", "gradient_boosting_all": "GB", "cnn_1d_views": "CNN"}
THRESHOLD = 0.5
pd.set_option("display.width", 200)

QUERY = """
SELECT s.signal_id, s.tic_id, s.toi, s.kind, s.disposition, s.label, s.split,
       s.period, s.duration_h, f.depth_obs_ppt, f.snr, f.noise_ppt, f.odd_even_sigma,
       f.secondary_snr, st.tmag, m.name || '_' || m.feature_set AS model, p.probability
FROM predictions p
JOIN models m ON m.model_id = p.model_id
JOIN signals s ON s.signal_id = p.signal_id
JOIN features f ON f.signal_id = s.signal_id
JOIN stars st ON st.tic_id = s.tic_id
"""


def load_wide():
    from dotenv import load_dotenv
    from sqlalchemy import create_engine, text

    load_dotenv(ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set. Create a .env file in the project root.")
    with create_engine(url).connect() as conn:
        long = pd.read_sql(text(QUERY), conn)
    long = long[long["model"].isin(MODELS)]
    wide = long.pivot(index="signal_id", columns="model", values="probability")
    wide = wide.rename(columns=MODELS)
    base = long.drop(columns=["model", "probability"]).drop_duplicates("signal_id")
    return base.merge(wide, left_on="signal_id", right_index=True)


def error_rate_table(df, column, bins=None, labels=None):
    """Share of wrong decisions (threshold 0.5) per group, for each model."""
    g = df[column] if bins is None else pd.cut(df[column], bins=bins, labels=labels)
    rows = []
    for key, part in df.groupby(g, observed=True):
        row = {column: key, "n": len(part), "planets": int(part["label"].sum())}
        for m in MODELS.values():
            wrong = ((part[m] >= THRESHOLD).astype(int) != part["label"])
            row[f"{m} error"] = wrong.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def analyze(df):
    lab = df[df["kind"] == "labeled"].copy()
    cand = df[df["kind"] == "candidate"].copy()
    ms = list(MODELS.values())
    out = {}

    print(f"Labeled signals: {len(lab)}, candidates: {len(cand)}")

    # 1. By disposition: share predicted as planet.
    rows = []
    for disp, part in lab.groupby("disposition"):
        row = {"disposition": disp, "n": len(part), "true label": int(part["label"].mean().round())}
        for m in ms:
            row[f"{m} says planet"] = (part[m] >= THRESHOLD).mean()
        rows.append(row)
    t1 = pd.DataFrame(rows)
    print("\n=== 1. Share predicted as planet, by catalog disposition ===")
    print("(CP/KP are planets, FP/FA are false positives)")
    print(t1.round(3).to_string(index=False))
    out["by_disposition"] = t1

    # 2. Error rate by star brightness, period, SNR.
    lab["tmag group"] = pd.qcut(lab["tmag"], 3, labels=["bright", "medium", "faint"])
    t2 = error_rate_table(lab, "tmag group")
    lab["period group"] = pd.cut(lab["period"], [0, 1.5, 5, 15, 1000],
                                 labels=["<1.5 d", "1.5-5 d", "5-15 d", ">15 d"])
    t3 = error_rate_table(lab, "period group")
    lab["snr group"] = pd.qcut(lab["snr"], 3, labels=["low SNR", "medium SNR", "high SNR"])
    t4 = error_rate_table(lab, "snr group")
    for title, t in (("star brightness (Tmag terciles)", t2), ("orbital period", t3),
                     ("signal-to-noise (SNR terciles)", t4)):
        print(f"\n=== 2. Error rate by {title} ===")
        print(t.round(3).to_string(index=False))
    out["by_tmag"], out["by_period"], out["by_snr"] = t2, t3, t4

    # 3. Most confident mistakes of the random forest.
    cols = ["signal_id", "tic_id", "toi", "disposition", "period", "depth_obs_ppt", "snr",
            "tmag", "odd_even_sigma", "secondary_snr", "RF", "GB", "CNN"]
    fps = lab[lab["label"] == 0].sort_values("RF", ascending=False).head(10)[cols]
    fns = lab[lab["label"] == 1].sort_values("RF").head(10)[cols]
    print("\n=== 3a. False positives the random forest was most sure were planets ===")
    print(fps.round(3).to_string(index=False))
    print("\n=== 3b. Real planets the random forest was most sure were not planets ===")
    print(fns.round(3).to_string(index=False))
    out["top_false_positives"], out["top_missed_planets"] = fps, fns
    top = pd.concat([fps.assign(error="false positive"), fns.assign(error="missed planet")])
    top.to_csv(PROCESSED / "error_analysis_top_errors.csv", index=False)

    # 4. Agreement between the random forest and the CNN.
    rf_yes, cnn_yes = lab["RF"] >= THRESHOLD, lab["CNN"] >= THRESHOLD
    right_rf = rf_yes.astype(int) == lab["label"]
    right_cnn = cnn_yes.astype(int) == lab["label"]
    t5 = pd.DataFrame({
        "both right": [(right_rf & right_cnn).sum()],
        "only RF right": [(right_rf & ~right_cnn).sum()],
        "only CNN right": [(~right_rf & right_cnn).sum()],
        "both wrong": [(~right_rf & ~right_cnn).sum()],
    })
    print("\n=== 4. Where the random forest and the CNN are right (labeled signals) ===")
    print(t5.to_string(index=False))
    sp_lab = lab["RF"].corr(lab["CNN"], method="spearman")
    sp_cand = cand["RF"].corr(cand["CNN"], method="spearman") if len(cand) else np.nan
    print(f"Spearman rank correlation RF vs CNN: labeled {sp_lab:.3f}, candidates {sp_cand:.3f}")
    out["agreement"] = t5

    # 5. Candidates: top of the ranking and the case of TOI 2280.01.
    if len(cand):
        cand["RF rank"] = cand["RF"].rank(ascending=False).astype(int)
        cand["CNN rank"] = cand["CNN"].rank(ascending=False).astype(int)
        show = ["tic_id", "toi", "period", "depth_obs_ppt", "snr", "tmag", "RF", "GB", "CNN",
                "RF rank", "CNN rank"]
        t6 = cand.sort_values("RF", ascending=False).head(10)[show]
        print("\n=== 5a. Top 10 candidates by random forest, with the CNN's opinion ===")
        print(t6.round(3).to_string(index=False))
        both = cand[(cand["RF"] >= 0.8) & (cand["CNN"] >= 0.8)]
        print(f"\nCandidates with probability >= 0.8 from both RF and CNN: {len(both)} "
              f"(RF alone: {(cand['RF'] >= 0.8).sum()}, CNN alone: {(cand['CNN'] >= 0.8).sum()})")
        case = cand[cand["toi"].round(2) == 2280.01]
        print("\n=== 5b. TOI 2280.01 (shallow, long period) ===")
        print(case[show + ["noise_ppt", "odd_even_sigma", "secondary_snr"]].round(3)
              .to_string(index=False) if len(case) else "not found")
        out["top_candidates"] = t6
        cand.sort_values("RF", ascending=False)[["signal_id"] + show].to_csv(
            PROCESSED / "error_analysis_candidates.csv", index=False)

    # Figure: error rate by brightness and period for each model.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (t, col, title) in zip(axes, ((t2, "tmag group", "Star brightness"),
                                          (t3, "period group", "Orbital period"),
                                          (t4, "snr group", "Signal-to-noise"))):
        x = np.arange(len(t))
        for i, m in enumerate(ms):
            ax.bar(x + (i - 1) * 0.27, t[f"{m} error"], width=0.27, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{k}\n(n={n})" for k, n in zip(t[col], t["n"])])
        ax.set_title(title)
    axes[0].set_ylabel("Error rate (threshold 0.5)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIG / "error_analysis.png", dpi=110)
    print("\nSaved: docs/figures/error_analysis.png, data/processed/error_analysis_top_errors.csv, "
          "data/processed/error_analysis_candidates.csv")
    return out


def main():
    analyze(load_wide())


if __name__ == "__main__":
    main()
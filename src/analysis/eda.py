from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

FEATURES = ["period", "duration_h", "depth_obs_ppt", "noise_ppt", "snr", "tmag",
            "odd_even_sigma", "secondary_snr", "n_transits"]


def auc(values, labels):
    """ROC-AUC of one feature (0.5 = no information), computed from ranks."""
    d = pd.DataFrame({"v": values, "y": labels}).dropna()
    n1, n0 = int((d["y"] == 1).sum()), int((d["y"] == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = d["v"].rank()
    return (ranks[d["y"] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    df = pd.read_parquet(PROCESSED / "views.parquet")
    meta = df.drop(columns=["global_view", "local_view"])

    print("=== Dataset ===")
    print(f"Signals: {len(meta)}")
    print(meta["label"].value_counts().rename({1: "planet (1)", 0: "false positive (0)"}))
    per_star = meta.groupby("tic_id").size()
    print(f"Unique stars: {len(per_star)}, stars with more than one signal: {(per_star > 1).sum()}")
    print("\nDispositions:")
    print(meta["disposition"].value_counts())

    rows = []
    for feat in FEATURES:
        row = {"feature": feat}
        for lab, tag in ((0, "neg"), (1, "pos")):
            x = meta.loc[meta["label"] == lab, feat]
            row[f"median_{tag}"] = x.median()
            row[f"q25_{tag}"] = x.quantile(0.25)
            row[f"q75_{tag}"] = x.quantile(0.75)
            row[f"missing_{tag}_%"] = 100 * x.isna().mean()
        row["auc"] = auc(meta[feat], meta["label"])
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["auc_strength"] = (summary["auc"] - 0.5).abs() + 0.5
    summary.to_csv(PROCESSED / "class_summary.csv", index=False)

    print("\n=== Feature summary by class (neg = false positive, pos = planet) ===")
    print(summary.drop(columns=["auc_strength"]).round(2).to_string(index=False))
    print("\n=== Single-feature AUC, strongest first (0.5 = no information) ===")
    ranked = summary.sort_values("auc_strength", ascending=False)
    print(ranked[["feature", "auc"]].round(3).to_string(index=False))

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, feat in zip(axes.ravel(), FEATURES):
        x = meta[feat]
        lo, hi = x.quantile([0.01, 0.99])
        bins = np.linspace(lo, hi, 40)
        for lab, name in ((0, "false positive"), (1, "planet")):
            vals = x[meta["label"] == lab].dropna().clip(lo, hi)
            ax.hist(vals, bins=bins, alpha=0.55, density=True, label=name)
        ax.set_title(feat)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(FIG / "eda_features.png", dpi=110)

    local = np.vstack(df["local_view"].to_numpy())
    fig, ax = plt.subplots(figsize=(8, 4))
    for lab, name in ((0, "false positive"), (1, "planet")):
        sel = local[(df["label"] == lab).to_numpy()]
        med = np.median(sel, axis=0)
        ax.plot(med, label=f"{name} (median)")
        ax.fill_between(range(sel.shape[1]), np.percentile(sel, 25, axis=0),
                        np.percentile(sel, 75, axis=0), alpha=0.2)
    ax.set_xlabel("Local view bin")
    ax.set_ylabel("Flux (ppt)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "eda_mean_local_view.png", dpi=110)

    print("\n=== Signals with many empty global bins ===")
    for cut in (100, 300, 500):
        print(f"empty_global > {cut}: {(meta['empty_global'] > cut).sum()}")
    many = meta[meta["empty_global"] > 300]
    if len(many):
        print("Their periods (days), summary:")
        print(many["period"].describe().round(2).to_string())
        print("By label:")
        print(many["label"].value_counts().to_string())

    print("\nSaved: data/processed/class_summary.csv, docs/figures/eda_features.png, "
          "docs/figures/eda_mean_local_view.png")


if __name__ == "__main__":
    main()
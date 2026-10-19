import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lc_tools import (bin_view, clip_upper_outliers, compute_features, detrend,
                      fold_phase)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BTJD_OFFSET = 2457000.0
N_GLOBAL = 2001
N_LOCAL = 201


def process_signal(row):
    lc = pd.read_parquet(ROOT / row["file"], columns=["time", "flux"])
    t = lc["time"].to_numpy(dtype="float64")
    f = lc["flux"].to_numpy(dtype="float64")
    good = np.isfinite(t) & np.isfinite(f)
    t, f = t[good], f[good]
    f = f / np.median(f)

    period = float(row["period"])
    epoch = float(row["epoch"]) - BTJD_OFFSET
    dur_d = float(row["duration_h"]) / 24.0

    flat = detrend(t, f, period, epoch, dur_d)
    phase = fold_phase(t, period, epoch)
    keep = clip_upper_outliers(flat, phase, dur_d)
    t, flat, phase = t[keep], flat[keep], phase[keep]

    glob, empty_g = bin_view(phase, flat, -period / 2, period / 2, N_GLOBAL)
    half = min(max(3.0 * dur_d, 0.1), 0.5 * period)
    loc, empty_l = bin_view(phase, flat, -half, half, N_LOCAL)

    out = {
        "signal_id": row["signal_id"],
        "tic_id": int(row["tic_id"]),
        "label": int(row["label"]),
        "disposition": row["disposition"],
        "sector": int(row["sector"]),
        "n_transits": int(row["n_transits"]),
        "period": period,
        "duration_h": float(row["duration_h"]),
        "depth_cat_ppm": float(row["depth_ppm"]),
        "tmag": float(row["tmag"]),
        "n_points": int(len(t)),
        "empty_global": empty_g,
        "empty_local": empty_l,
    }
    out.update(compute_features(t, flat, phase, period, epoch, dur_d))
    out["global_view"] = ((glob - 1.0) * 1000.0).astype("float32").tolist()
    out["local_view"] = ((loc - 1.0) * 1000.0).astype("float32").tolist()
    return out


def plot_examples(df, path):
    pos = df[df["label"] == 1].sample(3, random_state=1)
    neg = df[df["label"] == 0].sample(3, random_state=1)
    sel = pd.concat([pos, neg])
    fig, axes = plt.subplots(len(sel), 2, figsize=(11, 2.2 * len(sel)))
    for ax_row, (_, r) in zip(axes, sel.iterrows()):
        ax_row[0].plot(r["global_view"], lw=0.8)
        ax_row[0].set_title(f"TIC {r['tic_id']} ({r['disposition']}) global view", fontsize=9)
        ax_row[1].plot(r["local_view"], lw=0.8)
        ax_row[1].set_title("local view", fontsize=9)
        ax_row[0].set_ylabel("ppt")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    log = pd.read_csv(PROCESSED / "download_log.csv")
    log = log.drop_duplicates("signal_id", keep="last")
    log = log[log["status"] == "ok"]

    labels = pd.read_parquet(PROCESSED / "toi_labels.parquet")
    labels["signal_id"] = (labels["tic_id"].astype(int).astype(str)
                           + "_" + labels["toi"].astype(str))
    cols = ["signal_id", "period", "epoch", "duration_h", "depth_ppm", "tmag", "disposition"]
    data = log.merge(labels[cols], on="signal_id", how="inner")
    if limit:
        data = data.head(limit)
    print(f"Signals to process: {len(data)}")

    rows, failures = [], []
    for n, (_, r) in enumerate(data.iterrows(), 1):
        try:
            rows.append(process_signal(r))
        except Exception as e:
            failures.append({"signal_id": r["signal_id"], "error": str(e)[:200]})
        if n % 100 == 0:
            print(f"  {n}/{len(data)} done")

    df = pd.DataFrame(rows)
    df.to_parquet(PROCESSED / "views.parquet", index=False)
    pd.DataFrame(failures).to_csv(PROCESSED / "views_failures.csv", index=False)

    print(f"\nSaved {len(df)} signals to data/processed/views.parquet")
    print(f"Failures: {len(failures)} (see data/processed/views_failures.csv)")
    print("\nBy label:")
    print(df["label"].value_counts())
    print("\nEmpty bins (global / local), median and max:")
    print(df[["empty_global", "empty_local"]].describe().loc[["50%", "max"]])
    print("\nSNR and observed depth by label (median):")
    print(df.groupby("label")[["snr", "depth_obs_ppt", "noise_ppt"]].median().round(2))

    plot_examples(df, FIG / "views_check.png")
    print("\nSaved docs/figures/views_check.png")


if __name__ == "__main__":
    main()
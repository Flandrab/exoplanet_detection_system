import os
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from baseline import FEATURE_SETS, MODEL_FACTORIES

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "data" / "models"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "signal_id", "noise_ppt", "n_in_transit", "depth_obs_ppt", "snr", "odd_even_sigma",
    "secondary_ppt", "secondary_snr", "n_points", "empty_global", "empty_local",
]


def score_candidates(cand, model_ids):
    """Probability of every model for every candidate (no labels exist for them)."""
    parts = []
    for fs_name, cols in FEATURE_SETS.items():
        for m_name in MODEL_FACTORIES:
            model = joblib.load(MODELS / f"{m_name}_{fs_name}.joblib")
            parts.append(pd.DataFrame({
                "model_id": model_ids[(m_name, fs_name)],
                "signal_id": cand["signal_id"].to_numpy(),
                "probability": model.predict_proba(cand[cols])[:, 1],
            }))
    return pd.concat(parts, ignore_index=True)


def build_ranking(cand, catalog, preds, model_ids):
    names = {mid: f"{m}_{fs}" for (m, fs), mid in model_ids.items()}
    wide = (preds.assign(key=preds["model_id"].map(names))
            .pivot(index="signal_id", columns="key", values="probability")
            .reset_index())
    ranking = (cand[["signal_id", "tic_id", "period", "duration_h", "depth_obs_ppt",
                     "snr", "tmag"]]
               .merge(catalog[["signal_id", "toi"]], on="signal_id", how="left")
               .merge(wide, on="signal_id", how="left"))
    return ranking.sort_values("random_forest_all", ascending=False).reset_index(drop=True)


def main():
    load_dotenv(ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set. Create a .env file in the project root.")
    engine = create_engine(url)

    cand = pd.read_parquet(PROCESSED / "views_candidates.parquet",
                           columns=["signal_id", "tic_id", "disposition", "sector",
                                    "n_transits", "period", "duration_h", "depth_cat_ppm",
                                    "tmag", "n_points", "empty_global", "empty_local",
                                    "noise_ppt", "n_in_transit", "depth_obs_ppt", "snr",
                                    "odd_even_sigma", "secondary_ppt", "secondary_snr"])
    catalog = pd.read_parquet(PROCESSED / "toi_candidates.parquet")
    catalog["signal_id"] = (catalog["tic_id"].astype(int).astype(str)
                            + "_" + catalog["toi"].astype(str))
    log = pd.read_csv(PROCESSED / "download_log_candidates.csv")
    log = log.drop_duplicates("signal_id", keep="last")[["signal_id", "file"]]

    with engine.connect() as conn:
        models = pd.read_sql(text("SELECT model_id, name, feature_set FROM models"), conn)
    model_ids = {(n, f): int(i) for n, f, i in
                 zip(models["name"], models["feature_set"], models["model_id"])}

    preds = score_candidates(cand, model_ids)

    signals = (cand[["signal_id", "tic_id", "disposition", "sector", "n_transits",
                     "period", "duration_h", "depth_cat_ppm"]]
               .merge(catalog[["signal_id", "toi", "epoch"]], on="signal_id", how="left")
               .merge(log, on="signal_id", how="left")
               .rename(columns={"epoch": "epoch_bjd", "file": "lightcurve_file"}))
    signals["kind"] = "candidate"
    features = cand[FEATURE_COLUMNS]
    stars = cand.groupby("tic_id", as_index=False)["tmag"].first()

    with engine.begin() as conn:
        # Replace any earlier candidates (features and predictions go with them).
        conn.exec_driver_sql("DELETE FROM signals WHERE kind = 'candidate'")
        conn.exec_driver_sql(
            "DELETE FROM stars WHERE tic_id NOT IN (SELECT tic_id FROM signals)")
        existing = set(pd.read_sql(text("SELECT tic_id FROM stars"), conn)["tic_id"].astype(int))
        new_stars = stars[~stars["tic_id"].astype(int).isin(existing)]
        for name, table in (("stars", new_stars), ("signals", signals),
                            ("features", features), ("predictions", preds)):
            table.to_sql(name, conn, if_exists="append", index=False,
                         method="multi", chunksize=500)
            print(f"loaded {name}: {len(table)} rows")

    ranking = build_ranking(cand, catalog, preds, model_ids)
    ranking.to_csv(PROCESSED / "candidate_ranking.csv", index=False)
    print("\nTop 15 candidates by random forest probability (all features):")
    show = ["tic_id", "toi", "period", "depth_obs_ppt", "snr", "tmag",
            "random_forest_all", "gradient_boosting_all"]
    print(ranking[show].head(15).round(3).to_string(index=False))

    rf_id = model_ids[("random_forest", "all")]
    with engine.connect() as conn:
        lab = pd.read_sql(text(
            "SELECT s.label, p.probability FROM predictions p "
            "JOIN signals s USING (signal_id) "
            "WHERE p.model_id = :mid AND s.kind = 'labeled'"), conn, params={"mid": rf_id})
    groups = {
        "labeled planets": lab.loc[lab["label"] == 1, "probability"].to_numpy(),
        "labeled false positives": lab.loc[lab["label"] == 0, "probability"].to_numpy(),
        "unresolved candidates": ranking["random_forest_all"].to_numpy(),
    }
    print("\nShare of each group above a probability threshold (random forest, all features):")
    print(f"{'group':26s} {'n':>6s} {'>=0.5':>8s} {'>=0.8':>8s}")
    for name, p in groups.items():
        print(f"{name:26s} {len(p):6d} {np.mean(p >= 0.5):8.2f} {np.mean(p >= 0.8):8.2f}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 21)
    for name, p in groups.items():
        ax.hist(p, bins=bins, alpha=0.5, density=True, label=f"{name} (n={len(p)})")
    ax.set_xlabel("Random forest probability of being a planet")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "candidate_scores.png", dpi=110)
    print("\nSaved: data/processed/candidate_ranking.csv, docs/figures/candidate_scores.png")


if __name__ == "__main__":
    main()
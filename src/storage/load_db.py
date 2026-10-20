import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

load_dotenv(ROOT / ".env")
url = os.environ.get("DATABASE_URL")
if not url:
    raise SystemExit("DATABASE_URL is not set. Create a .env file in the project root "
                     "(see .env.example).")
engine = create_engine(url)

VIEW_COLUMNS = [
    "signal_id", "tic_id", "label", "disposition", "sector", "n_transits", "period",
    "duration_h", "depth_cat_ppm", "tmag", "n_points", "empty_global", "empty_local",
    "noise_ppt", "n_in_transit", "depth_obs_ppt", "snr", "odd_even_sigma",
    "secondary_ppt", "secondary_snr",
]
FEATURE_COLUMNS = [
    "signal_id", "noise_ppt", "n_in_transit", "depth_obs_ppt", "snr", "odd_even_sigma",
    "secondary_ppt", "secondary_snr", "n_points", "empty_global", "empty_local",
]


def build_tables():
    views = pd.read_parquet(PROCESSED / "views.parquet", columns=VIEW_COLUMNS)

    labels = pd.read_parquet(PROCESSED / "toi_labels.parquet")
    labels["signal_id"] = (labels["tic_id"].astype(int).astype(str)
                           + "_" + labels["toi"].astype(str))
    log = pd.read_csv(PROCESSED / "download_log.csv").drop_duplicates("signal_id", keep="last")
    split = pd.read_csv(PROCESSED / "split.csv")[["signal_id", "split"]]

    stars = views.groupby("tic_id", as_index=False)["tmag"].first()

    signals = (
        views[["signal_id", "tic_id", "label", "disposition", "sector", "n_transits",
               "period", "duration_h", "depth_cat_ppm"]]
        .merge(labels[["signal_id", "toi", "epoch"]], on="signal_id", how="left")
        .merge(log[["signal_id", "file"]], on="signal_id", how="left")
        .merge(split, on="signal_id", how="left")
        .rename(columns={"epoch": "epoch_bjd", "file": "lightcurve_file"})
    )
    signals["kind"] = "labeled"

    features = views[FEATURE_COLUMNS]

    models = pd.read_csv(PROCESSED / "baseline_results.csv").rename(
        columns={"features": "feature_set", "model": "name"})
    models["model_file"] = ("data/models/" + models["name"] + "_"
                            + models["feature_set"] + ".joblib")
    models = models[["name", "feature_set", "model_file", "cv_pr_auc", "cv_pr_auc_std",
                     "cv_roc_auc", "cv_f1", "test_precision", "test_recall", "test_f1",
                     "test_roc_auc", "test_pr_auc", "tn", "fp", "fn", "tp"]]
    return stars, signals, features, models


def main():
    stars, signals, features, models = build_tables()
    schema_sql = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")

    with engine.begin() as conn:
        conn.exec_driver_sql(
            "DROP TABLE IF EXISTS predictions, models, features, signals, stars CASCADE")
        conn.exec_driver_sql(schema_sql)
        for name, table in (("stars", stars), ("signals", signals),
                            ("features", features), ("models", models)):
            table.to_sql(name, conn, if_exists="append", index=False,
                         method="multi", chunksize=500)
            print(f"loaded {name}: {len(table)} rows")

    with engine.connect() as conn:
        print("\nRow counts in the database:")
        for table in ("stars", "signals", "features", "models", "predictions"):
            n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            print(f"  {table:12s} {n}")
        print("\nSignals by label and split:")
        print(pd.read_sql(text(
            "SELECT label, split, COUNT(*) AS signals FROM signals "
            "GROUP BY label, split ORDER BY label, split"), conn).to_string(index=False))
        print("\nModels by test PR-AUC:")
        print(pd.read_sql(text(
            "SELECT name, feature_set, ROUND(CAST(test_pr_auc AS numeric), 3) AS test_pr_auc "
            "FROM models ORDER BY test_pr_auc DESC"), conn).to_string(index=False))


if __name__ == "__main__":
    main()
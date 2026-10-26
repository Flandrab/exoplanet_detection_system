import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score
from sqlalchemy import create_engine, text

from baseline import FEATURE_SETS, MODEL_FACTORIES, cross_validate_by_star

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "data" / "models"


def build_predictions(df, model_ids):
    """Honest probabilities for every signal.

    Train signals get out-of-fold probabilities (each one comes from a model that never
    saw its star). Test signals get probabilities from the model trained on the full
    train set, which never saw those stars either.
    """
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    y_train = train["label"].to_numpy()
    parts = []
    for fs_name, cols in FEATURE_SETS.items():
        for m_name, factory in MODEL_FACTORIES.items():
            oof, _ = cross_validate_by_star(factory, train[cols], y_train, train["tic_id"])
            model = joblib.load(MODELS / f"{m_name}_{fs_name}.joblib")
            test_proba = model.predict_proba(test[cols])[:, 1]
            model_id = model_ids[(m_name, fs_name)]
            parts.append(pd.DataFrame({"model_id": model_id,
                                       "signal_id": train["signal_id"],
                                       "probability": oof}))
            parts.append(pd.DataFrame({"model_id": model_id,
                                       "signal_id": test["signal_id"],
                                       "probability": test_proba}))
            print(f"predicted: {fs_name} / {m_name}")
    return pd.concat(parts, ignore_index=True)


def main():
    load_dotenv(ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set. See .env.example.")
    engine = create_engine(url)

    feature_cols = sorted(set(sum(FEATURE_SETS.values(), [])))
    df = pd.read_parquet(PROCESSED / "views.parquet",
                         columns=["signal_id", "tic_id", "label"] + feature_cols)
    split = pd.read_csv(PROCESSED / "split.csv")[["signal_id", "split"]]
    df = df.merge(split, on="signal_id", how="inner")

    with engine.connect() as conn:
        models = pd.read_sql(text("SELECT model_id, name, feature_set FROM models"), conn)
    model_ids = {(n, f): int(i) for n, f, i in
                 zip(models["name"], models["feature_set"], models["model_id"])}

    preds = build_predictions(df, model_ids)

    with engine.begin() as conn:
        conn.exec_driver_sql(
            "DELETE FROM predictions WHERE signal_id IN "
            "(SELECT signal_id FROM signals WHERE kind = 'labeled')")
        preds.to_sql("predictions", conn, if_exists="append", index=False,
                     method="multi", chunksize=1000)
    print(f"\nStored {len(preds)} predictions "
          f"({len(model_ids)} models x {df['signal_id'].nunique()} signals)")

    with engine.connect() as conn:
        check = pd.read_sql(text(
            "SELECT m.name, m.feature_set, m.cv_pr_auc, m.test_pr_auc, "
            "s.split, s.label, p.probability "
            "FROM predictions p JOIN models m USING (model_id) "
            "JOIN signals s USING (signal_id)"), conn)
    rows = []
    for (name, fs), g in check.groupby(["name", "feature_set"]):
        tr, te = g[g["split"] == "train"], g[g["split"] == "test"]
        rows.append({
            "model": name, "features": fs,
            "train_oof_pr_auc": average_precision_score(tr["label"], tr["probability"]),
            "cv_pr_auc_table": g["cv_pr_auc"].iloc[0],
            "test_pr_auc_from_db": average_precision_score(te["label"], te["probability"]),
            "test_pr_auc_table": g["test_pr_auc"].iloc[0],
        })
    print("\nCheck: PR-AUC recomputed from the stored predictions vs the models table")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
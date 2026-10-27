import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

MODEL_COLUMNS = ["name", "feature_set", "model_file", "cv_pr_auc", "cv_pr_auc_std",
                 "cv_roc_auc", "cv_f1", "test_precision", "test_recall", "test_f1",
                 "test_roc_auc", "test_pr_auc", "tn", "fp", "fn", "tp"]


def main():
    load_dotenv(ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set. Create a .env file in the project root.")
    engine = create_engine(url)

    res = pd.read_csv(PROCESSED / "cnn_results.csv").rename(
        columns={"features": "feature_set", "model": "name"})
    res["model_file"] = "data/models/" + res["name"] + "_" + res["feature_set"] + "_fold*.pt"
    res = res[MODEL_COLUMNS]
    preds = pd.read_parquet(PROCESSED / "cnn_predictions.parquet")

    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM models WHERE name = 'cnn_1d'")  # predictions cascade
        res.to_sql("models", conn, if_exists="append", index=False)
        model_id = int(conn.execute(text(
            "SELECT model_id FROM models WHERE name = 'cnn_1d'")).scalar())
        known = set(pd.read_sql(text("SELECT signal_id FROM signals"), conn)["signal_id"])
        out = preds[preds["signal_id"].isin(known)].copy()
        missing = len(preds) - len(out)
        out.insert(0, "model_id", model_id)
        out[["model_id", "signal_id", "probability"]].to_sql(
            "predictions", conn, if_exists="append", index=False,
            method="multi", chunksize=1000)
    print(f"Stored model_id {model_id} and {len(out)} predictions "
          f"({missing} signals not in the database were skipped)")

    with engine.connect() as conn:
        check = pd.read_sql(text(
            "SELECT s.split, s.label, s.kind, p.probability, m.test_pr_auc, m.cv_pr_auc "
            "FROM predictions p JOIN models m USING (model_id) "
            "JOIN signals s USING (signal_id) WHERE m.name = 'cnn_1d'"), conn)
        counts = pd.read_sql(text("SELECT COUNT(*) AS models FROM models"), conn)
    lab = check[check["kind"] == "labeled"]
    tr, te = lab[lab["split"] == "train"], lab[lab["split"] == "test"]
    print("\nCheck (recomputed from the database vs the models table):")
    print(f"  train OOF PR-AUC {average_precision_score(tr['label'], tr['probability']):.3f}"
          f" vs cv_pr_auc {check['cv_pr_auc'].iloc[0]:.3f}")
    print(f"  test PR-AUC      {average_precision_score(te['label'], te['probability']):.3f}"
          f" vs test_pr_auc {check['test_pr_auc'].iloc[0]:.3f}")
    print(f"  candidate predictions: {(check['kind'] == 'candidate').sum()}")
    print(f"  models in the database: {int(counts['models'].iloc[0])}")


if __name__ == "__main__":
    main()
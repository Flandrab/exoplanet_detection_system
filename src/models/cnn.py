import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.metrics import evaluate  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "data" / "models"
FIG = ROOT / "docs" / "figures"
MODELS.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

MODEL_NAME, FEATURE_SET = "cnn_1d", "views"


def prepare_views(df):
    """Turn the stored views into network inputs.

    Flux becomes parts per thousand below/above the median, is divided by the noise of
    the star (so depth is measured in noise units) and compressed with arcsinh so very
    deep eclipses do not dominate. Returns arrays (n, 2001) and (n, 201).
    """
    out = []
    for col in ("global_view", "local_view"):
        x = np.stack(df[col].to_numpy()).astype(float)
        x = np.nan_to_num(x, nan=1.0)
        x = (x - 1.0) * 1000.0
        x = x - np.median(x, axis=1, keepdims=True)
        out.append(x)
    xg, xl = out
    noise = df["noise_ppt"].to_numpy(dtype=float)
    fallback = 1.4826 * np.median(np.abs(xg), axis=1)
    noise = np.where(np.isfinite(noise) & (noise > 0), noise, fallback)
    noise = np.where(noise > 0, noise, 1.0)[:, None]
    return np.arcsinh(xg / noise), np.arcsinh(xl / noise)


def run(train_model, predict_proba, save_model, n_splits=5):
    views = pd.read_parquet(PROCESSED / "views.parquet",
                            columns=["signal_id", "tic_id", "label", "noise_ppt",
                                     "global_view", "local_view"])
    split = pd.read_csv(PROCESSED / "split.csv")[["signal_id", "split"]]
    df = views.merge(split, on="signal_id", how="inner")
    cand = pd.read_parquet(PROCESSED / "views_candidates.parquet",
                           columns=["signal_id", "noise_ppt", "global_view", "local_view"])

    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    y_train, y_test = train["label"].to_numpy(), test["label"].to_numpy()
    tg, tl = prepare_views(train)
    eg, el = prepare_views(test)
    cg, cl = prepare_views(cand)
    print(f"Train: {len(train)} signals, test: {len(test)}, candidates: {len(cand)}")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(train))
    test_parts, cand_parts, fold_pr, curves = [], [], [], []
    for k, (tr, va) in enumerate(splitter.split(tg, y_train, train["tic_id"]), start=1):
        model, history = train_model(tg[tr], tl[tr], y_train[tr],
                                     val=(tg[va], tl[va], y_train[va]))
        oof[va] = predict_proba(model, tg[va], tl[va])
        test_parts.append(predict_proba(model, eg, el))
        cand_parts.append(predict_proba(model, cg, cl))
        fold_pr.append(evaluate(y_train[va], oof[va])["pr_auc"])
        curves.append(history)
        save_model(model, MODELS / f"{MODEL_NAME}_{FEATURE_SET}_fold{k}.pt")
        print(f"fold {k}/{n_splits} done: validation PR-AUC {fold_pr[-1]:.3f}")

    # Test and candidate probabilities: average of the five fold networks (an ensemble).
    test_proba = np.mean(test_parts, axis=0)
    cand_proba = np.mean(cand_parts, axis=0)
    cv, te = evaluate(y_train, oof), evaluate(y_test, test_proba)
    row = {
        "features": FEATURE_SET, "model": MODEL_NAME,
        "cv_pr_auc": cv["pr_auc"], "cv_pr_auc_std": float(np.std(fold_pr)),
        "cv_roc_auc": cv["roc_auc"], "cv_f1": cv["f1"],
        "test_precision": te["precision"], "test_recall": te["recall"],
        "test_f1": te["f1"], "test_roc_auc": te["roc_auc"], "test_pr_auc": te["pr_auc"],
        "tn": te["tn"], "fp": te["fp"], "fn": te["fn"], "tp": te["tp"],
    }
    pd.DataFrame([row]).to_csv(PROCESSED / "cnn_results.csv", index=False)
    preds = pd.concat([
        pd.DataFrame({"signal_id": train["signal_id"], "probability": oof, "part": "train_oof"}),
        pd.DataFrame({"signal_id": test["signal_id"], "probability": test_proba, "part": "test"}),
        pd.DataFrame({"signal_id": cand["signal_id"], "probability": cand_proba,
                      "part": "candidate"}),
    ], ignore_index=True)
    preds.to_parquet(PROCESSED / "cnn_predictions.parquet", index=False)

    # Comparison with the baselines on the same held-out stars.
    base = pd.read_csv(PROCESSED / "baseline_results.csv")
    show = ["features", "model", "cv_pr_auc", "test_precision", "test_recall", "test_f1",
            "test_roc_auc", "test_pr_auc"]
    table = pd.concat([base[show], pd.DataFrame([row])[show]], ignore_index=True)
    print("\n=== Test set: baselines and the CNN (same split, same metrics) ===")
    print(table.sort_values("test_pr_auc", ascending=False).round(3).to_string(index=False))
    print(f"\nPositive share in test set (random classifier PR-AUC): {y_test.mean():.3f}")
    print(f"CNN confusion matrix on test (tn fp fn tp): {row['tn']} {row['fp']} "
          f"{row['fn']} {row['tp']}")
    share = (cand_proba >= 0.5).mean()
    print(f"CNN: share of candidates with probability >= 0.5: {share:.2f} "
          f"(labeled train planets share: {(oof[y_train == 1] >= 0.5).mean():.2f}, "
          f"labeled false positives: {(oof[y_train == 0] >= 0.5).mean():.2f})")

    # Learning curves: loss on training folds and PR-AUC on the held-out fold.
    loss = np.array([[h[0] for h in c] for c in curves])
    val = np.array([[h[1] for h in c] for c in curves])
    epochs = np.arange(1, loss.shape[1] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, loss.mean(0))
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss (mean of folds)")
    axes[1].plot(epochs, val.mean(0))
    axes[1].fill_between(epochs, val.min(0), val.max(0), alpha=0.25)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation PR-AUC (mean, min-max of folds)")
    fig.tight_layout()
    fig.savefig(FIG / "cnn_learning_curves.png", dpi=110)
    best = int(np.argmax(val.mean(0))) + 1
    print(f"\nValidation PR-AUC by epoch (mean of folds): best epoch {best}, "
          f"last epoch {val.mean(0)[-1]:.3f}, best {val.mean(0).max():.3f}")
    print("Saved: data/processed/cnn_results.csv, cnn_predictions.parquet, "
          "docs/figures/cnn_learning_curves.png, networks in data/models/")


def main():
    from cnn_model import predict_proba, save_model, train_model
    run(train_model, predict_proba, save_model)


if __name__ == "__main__":
    main()
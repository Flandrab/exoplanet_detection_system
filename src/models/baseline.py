import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.metrics import evaluate  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "data" / "models"
FIG = ROOT / "docs" / "figures"
MODELS.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

FEATURE_SETS = {
    "all": ["period", "duration_h", "depth_obs_ppt", "noise_ppt", "snr", "tmag",
            "odd_even_sigma", "secondary_ppt", "secondary_snr", "n_transits"],
    "no_period": ["duration_h", "depth_obs_ppt", "noise_ppt", "snr", "tmag",
                  "odd_even_sigma", "secondary_ppt", "secondary_snr"],
}


def make_logreg():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        QuantileTransformer(n_quantiles=200, output_distribution="normal"),
        LogisticRegression(max_iter=2000, class_weight="balanced"))


def make_forest():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                               class_weight="balanced", n_jobs=-1, random_state=42))


def make_boosting():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.0, class_weight="balanced", random_state=42)


MODEL_FACTORIES = {
    "logistic_regression": make_logreg,
    "random_forest": make_forest,
    "gradient_boosting": make_boosting,
}


def cross_validate_by_star(make_model, X, y, groups, n_splits=5):
    """Out-of-fold probabilities, with all signals of a star in the same fold."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    fold_pr = []
    for tr, va in splitter.split(X, y, groups):
        model = make_model()
        model.fit(X.iloc[tr], y[tr])
        proba = model.predict_proba(X.iloc[va])[:, 1]
        oof[va] = proba
        fold_pr.append(evaluate(y[va], proba)["pr_auc"])
    return oof, np.array(fold_pr)


def main():
    df = pd.read_parquet(PROCESSED / "views.parquet").drop(columns=["global_view", "local_view"])
    split = pd.read_csv(PROCESSED / "split.csv")[["signal_id", "split"]]
    df = df.merge(split, on="signal_id", how="inner")
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    y_train, y_test = train["label"].to_numpy(), test["label"].to_numpy()
    print(f"Train: {len(train)} signals, test: {len(test)} signals")

    rows = []
    test_proba = {}
    fitted = {}
    for fs_name, cols in FEATURE_SETS.items():
        for m_name, factory in MODEL_FACTORIES.items():
            oof, fold_pr = cross_validate_by_star(factory, train[cols], y_train, train["tic_id"])
            cv = evaluate(y_train, oof)
            model = factory().fit(train[cols], y_train)
            proba = model.predict_proba(test[cols])[:, 1]
            te = evaluate(y_test, proba)
            rows.append({
                "features": fs_name, "model": m_name,
                "cv_pr_auc": cv["pr_auc"], "cv_pr_auc_std": fold_pr.std(),
                "cv_roc_auc": cv["roc_auc"], "cv_f1": cv["f1"],
                "test_precision": te["precision"], "test_recall": te["recall"],
                "test_f1": te["f1"], "test_roc_auc": te["roc_auc"],
                "test_pr_auc": te["pr_auc"],
                "tn": te["tn"], "fp": te["fp"], "fn": te["fn"], "tp": te["tp"],
            })
            test_proba[(fs_name, m_name)] = proba
            fitted[(fs_name, m_name)] = model
            joblib.dump(model, MODELS / f"{m_name}_{fs_name}.joblib")
            print(f"done: {fs_name} / {m_name}")

    results = pd.DataFrame(rows)
    results.to_csv(PROCESSED / "baseline_results.csv", index=False)
    show = ["features", "model", "cv_pr_auc", "cv_pr_auc_std", "test_precision",
            "test_recall", "test_f1", "test_roc_auc", "test_pr_auc"]
    print("\n=== Results (cv = 5-fold by star on train, test = held-out stars) ===")
    print(results[show].round(3).to_string(index=False))
    print(f"\nPositive share in test set (a random classifier's PR-AUC): {y_test.mean():.3f}")
    print("\n=== Confusion matrices on the test set (tn fp fn tp) ===")
    print(results[["features", "model", "tn", "fp", "fn", "tp"]].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for m_name in MODEL_FACTORIES:
        proba = test_proba[("all", m_name)]
        fpr, tpr, _ = roc_curve(y_test, proba)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        axes[0].plot(fpr, tpr, label=m_name)
        axes[1].plot(rec, prec, label=m_name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC (test set, all features)")
    axes[1].axhline(y_test.mean(), color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall (test set, all features)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIG / "baseline_curves.png", dpi=110)

    cols = FEATURE_SETS["all"]
    imp = permutation_importance(fitted[("all", "random_forest")], test[cols], y_test,
                                 scoring="average_precision", n_repeats=10, random_state=42)
    order = np.argsort(imp.importances_mean)[::-1]
    print("\n=== Permutation importance, random forest, all features (drop in test PR-AUC) ===")
    for i in order:
        print(f"{cols[i]:>16}: {imp.importances_mean[i]:.4f} +/- {imp.importances_std[i]:.4f}")
    print("\nSaved: data/processed/baseline_results.csv, docs/figures/baseline_curves.png, "
          "models in data/models/")


if __name__ == "__main__":
    main()
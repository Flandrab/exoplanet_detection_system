import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
from error_analysis import load_wide  # noqa: E402
from evaluation.metrics import evaluate  # noqa: E402

PROCESSED = ROOT / "data" / "processed"

# Simple average of probabilities with equal weights: nothing is tuned, so the test set
# stays untouched by any choice.
COMBOS = {
    "RF": ["RF"],
    "GB": ["GB"],
    "CNN": ["CNN"],
    "RF + GB": ["RF", "GB"],
    "RF + CNN": ["RF", "CNN"],
    "RF + GB + CNN": ["RF", "GB", "CNN"],
}
N_BOOT = 1000


def combo_proba(df, cols):
    return df[cols].mean(axis=1).to_numpy()


def bootstrap_vs_rf(test, n_boot=N_BOOT, seed=42):
    """Paired bootstrap over stars: PR-AUC of each combination minus PR-AUC of the RF."""
    rng = np.random.default_rng(seed)
    y = test["label"].to_numpy()
    probas = {name: combo_proba(test, cols) for name, cols in COMBOS.items()}
    star_rows = list(test.groupby("tic_id").indices.values())
    diffs = {name: [] for name in COMBOS}
    for _ in range(n_boot):
        pick = rng.integers(0, len(star_rows), len(star_rows))
        rows = np.concatenate([star_rows[i] for i in pick])
        if y[rows].min() == y[rows].max():
            continue
        base = average_precision_score(y[rows], probas["RF"][rows])
        for name, p in probas.items():
            diffs[name].append(average_precision_score(y[rows], p[rows]) - base)
    return {name: (np.mean(d), np.percentile(d, 2.5), np.percentile(d, 97.5))
            for name, d in diffs.items()}


def run(df):
    lab = df[df["kind"] == "labeled"]
    train, test = lab[lab["split"] == "train"], lab[lab["split"] == "test"].reset_index(drop=True)
    y_tr, y_te = train["label"].to_numpy(), test["label"].to_numpy()
    print(f"Train (out-of-fold probabilities): {len(train)}, test: {len(test)}")

    boot = bootstrap_vs_rf(test)
    rows = []
    for name, cols in COMBOS.items():
        te = evaluate(y_te, combo_proba(test, cols))
        cv = evaluate(y_tr, combo_proba(train, cols))
        mean, lo, hi = boot[name]
        rows.append({
            "combination": name, "train_oof_pr_auc": cv["pr_auc"],
            "test_precision": te["precision"], "test_recall": te["recall"],
            "test_f1": te["f1"], "test_roc_auc": te["roc_auc"], "test_pr_auc": te["pr_auc"],
            "pr_auc_minus_RF": mean, "ci95_low": lo, "ci95_high": hi,
            "tn": te["tn"], "fp": te["fp"], "fn": te["fn"], "tp": te["tp"],
        })
    res = pd.DataFrame(rows)
    res.to_csv(PROCESSED / "ensemble_results.csv", index=False)
    show = ["combination", "train_oof_pr_auc", "test_precision", "test_recall", "test_f1",
            "test_roc_auc", "test_pr_auc"]
    print("\n=== Averaging the probabilities (equal weights, nothing tuned) ===")
    print(res[show].round(3).to_string(index=False))
    print("\n=== Test PR-AUC minus the random forest's, paired bootstrap over stars ===")
    print("(if the 95% interval contains 0, the difference could be chance)")
    print(res[["combination", "pr_auc_minus_RF", "ci95_low", "ci95_high"]]
          .round(3).to_string(index=False))
    print("\nConfusion matrices on test (tn fp fn tp):")
    print(res[["combination", "tn", "fp", "fn", "tp"]].to_string(index=False))

    cand = df[df["kind"] == "candidate"].copy()
    if len(cand):
        cand["avg"] = combo_proba(cand, ["RF", "GB", "CNN"])
        top = cand.sort_values("avg", ascending=False).head(10)
        print("\n=== Top 10 candidates by the average of RF, GB and CNN ===")
        print(top[["tic_id", "toi", "period", "depth_obs_ppt", "snr", "tmag", "RF", "GB",
                   "CNN", "avg"]].round(3).to_string(index=False))
        print(f"\nCandidates with average >= 0.8: {(cand['avg'] >= 0.8).sum()}, "
              f">= 0.7: {(cand['avg'] >= 0.7).sum()} of {len(cand)}")
        cand.sort_values("avg", ascending=False)[
            ["signal_id", "tic_id", "toi", "RF", "GB", "CNN", "avg"]].to_csv(
            PROCESSED / "ensemble_candidates.csv", index=False)
    print("\nSaved: data/processed/ensemble_results.csv, ensemble_candidates.csv")
    return res


def main():
    run(load_wide())


if __name__ == "__main__":
    main()
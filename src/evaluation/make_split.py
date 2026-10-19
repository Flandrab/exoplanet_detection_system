from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

meta = pd.read_parquet(PROCESSED / "views.parquet", columns=["signal_id", "tic_id", "label"])

# One fixed split for ALL models: about 20% of the stars are held out as a test set.
# Grouping by star means no star appears in both train and test.
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(splitter.split(meta, meta["label"], groups=meta["tic_id"]))

meta["split"] = "train"
meta.loc[meta.index[test_idx], "split"] = "test"
meta.to_csv(PROCESSED / "split.csv", index=False)

print(meta.groupby("split")["label"].agg(signals="size", planets="sum"))
print("\nShare of planets per split:")
print(meta.groupby("split")["label"].mean().round(3))
train_stars = set(meta.loc[meta["split"] == "train", "tic_id"])
test_stars = set(meta.loc[meta["split"] == "test", "tic_id"])
print(f"\nStars in both train and test (must be 0): {len(train_stars & test_stars)}")
print("Saved data/processed/split.csv")
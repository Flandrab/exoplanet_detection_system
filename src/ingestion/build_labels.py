from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "toi_2026-10-17.csv"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

COLS = {
    "tid": "tic_id",
    "toi": "toi",
    "tfopwg_disp": "disposition",
    "pl_orbper": "period",
    "pl_tranmid": "epoch",
    "pl_trandurh": "duration_h",
    "pl_trandep": "depth_ppm",
    "st_tmag": "tmag",
}
NEEDED = ["disposition", "period", "epoch", "duration_h"]

df = pd.read_csv(RAW, comment="#", low_memory=False)
print(f"Raw shape: {df.shape}")

missing = set(COLS) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

print("\nDisposition counts (raw, including empty):")
print(df["tfopwg_disp"].value_counts(dropna=False))

df = df[list(COLS)].rename(columns=COLS)
df["disposition"] = df["disposition"].str.strip().str.upper()

before = len(df)
df = df.dropna(subset=NEEDED)
print(f"\nDropped {before - len(df)} rows missing disposition/period/epoch/duration")

labeled = df[df["disposition"].isin(["CP", "KP", "FP", "FA"])].copy()
labeled["label"] = labeled["disposition"].isin(["CP", "KP"]).astype(int)
candidates = df[df["disposition"] == "PC"].copy()

print("\nLabeled set (Option A):")
print(labeled["label"].value_counts())
print(f"Rows: {len(labeled)}, unique stars: {labeled['tic_id'].nunique()}")
print(f"Stars with multiple TOIs: {labeled['tic_id'].duplicated().sum()}")
print(f"Candidates (PC) kept for later prediction: {len(candidates)}")

labeled.to_parquet(OUT / "toi_labels.parquet", index=False)
candidates.to_parquet(OUT / "toi_candidates.parquet", index=False)
import re
from pathlib import Path
import pandas as pd
import lightkurve as lk

ROOT = Path(__file__).resolve().parents[2]
labels = pd.read_parquet(ROOT / "data" / "processed" / "toi_labels.parquet")

sample = labels.drop_duplicates("tic_id").sample(50, random_state=42)


def get_sectors(result):
    sectors = set()
    for text in result.table["mission"]:
        match = re.search(r"Sector\s+(\d+)", str(text))
        if match:
            sectors.add(int(match.group(1)))
    return sorted(sectors)


rows = []
for _, r in sample.iterrows():
    tic = int(r["tic_id"])
    sectors = []
    try:
        res = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author="SPOC", exptime=120)
        if len(res) > 0:
            sectors = get_sectors(res)
    except Exception as e:
        print(f"TIC {tic}: search failed ({e})")
    rows.append({
        "tic_id": tic,
        "label": int(r["label"]),
        "period": float(r["period"]),
        "n_sectors": len(sectors),
        "sectors": sectors,
    })
    print(f"TIC {tic}: {len(sectors)} sectors")

out = pd.DataFrame(rows)
out.to_csv(ROOT / "data" / "processed" / "availability_sample.csv", index=False)

has_data = out["n_sectors"] > 0
print(f"\nStars with SPOC 2-min data: {has_data.sum()} / {len(out)}")
print("By label (share with data):")
print(out.groupby("label")["n_sectors"].apply(lambda s: (s > 0).mean()).round(2))
print(f"Median sectors per star (with data): {out.loc[has_data, 'n_sectors'].median()}")
print(f"Share with period <= 15 d: {(out['period'] <= 15).mean():.2f}")
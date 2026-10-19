import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightkurve as lk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

labels = pd.read_parquet(ROOT / "data" / "processed" / "toi_labels.parquet")
idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
row = labels.iloc[idx]

tic = int(row["tic_id"])
period = float(row["period"])
epoch = float(row["epoch"]) - 2457000.0
dur_d = float(row["duration_h"]) / 24.0
print(f"Row {idx}: TIC {tic}, {row['disposition']}, period {period:.3f} d, duration {dur_d * 24:.1f} h")

res = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author="SPOC", exptime=120)
if len(res) == 0:
    raise SystemExit("No SPOC 2-min data for this star.")

lc = res[0].download().remove_nans().normalize()
t = lc.time.value
k_min = int(np.ceil((t.min() - epoch) / period))
k_max = int(np.floor((t.max() - epoch) / period))
n_transits = max(0, k_max - k_min + 1)
print(f"Points: {len(lc)}, transits in this sector: {n_transits}")

mask = lc.create_transit_mask(period=period, transit_time=epoch, duration=dur_d * 2)
win = max(101, int(dur_d * 24 * 30 * 3) | 1)
flat = lc.flatten(window_length=win, mask=mask)
folded = flat.fold(period=period, epoch_time=epoch)

fig, ax = plt.subplots(2, 1, figsize=(10, 7))
ax[0].scatter(t, flat.flux.value, s=1)
ax[0].set_title(f"TIC {tic}, {row['disposition']}: flattened light curve")
ax[0].set_xlabel("Time (BTJD)")
ax[0].set_ylabel("Normalized flux")
ax[1].scatter(folded.time.value, folded.flux.value, s=1)
ax[1].set_xlim(-3 * dur_d, 3 * dur_d)
ax[1].set_title(f"Folded on period {period:.3f} d")
ax[1].set_xlabel("Time from transit (days)")
ax[1].set_ylabel("Normalized flux")
fig.tight_layout()

out = FIG / f"inspect_row{idx}_TIC{tic}.png"
fig.savefig(out, dpi=120)
print(f"Saved {out}")
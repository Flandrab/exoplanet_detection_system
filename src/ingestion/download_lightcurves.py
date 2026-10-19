import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightkurve as lk

ROOT = Path(__file__).resolve().parents[2]
LC_DIR = ROOT / "data" / "raw" / "lightcurves"
CACHE = ROOT / "data" / "cache"
LOG = ROOT / "data" / "processed" / "download_log.csv"
LC_DIR.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

MIN_TRANSITS = 2
BTJD_OFFSET = 2457000.0
MAX_SPAN_D = 27.5


def retry(fn, tries=3):
    for attempt in range(tries):
        try:
            return fn()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def count_transits(t, period, epoch, dur_d):
    half = max(dur_d / 2, 0.02)
    k_min = int(np.ceil((t.min() - epoch) / period))
    k_max = int(np.floor((t.max() - epoch) / period))
    n = 0
    for k in range(k_min, k_max + 1):
        if np.any(np.abs(t - (epoch + k * period)) < half):
            n += 1
    return n


def get_sector_list(tic):
    res = retry(lambda: lk.search_lightcurve(
        f"TIC {tic}", mission="TESS", author="SPOC", exptime=120))
    if len(res) == 0:
        return res, []
    found = {}
    for i, text in enumerate(res.table["mission"]):
        m = re.search(r"Sector\s+(\d+)", str(text))
        if m:
            found.setdefault(int(m.group(1)), i)
    return res, sorted(found.items())

def load_sector(tic, res, sector, idx):
    path = LC_DIR / f"tic{tic}_s{sector:04d}.parquet"
    if path.exists():
        return path, pd.read_parquet(path, columns=["time"])["time"].to_numpy()
    lc = retry(lambda: res[idx].download(download_dir=str(CACHE))).remove_nans()
    if "quality" in lc.colnames:
        quality = np.asarray(lc["quality"]).astype("int32")
    else:
        quality = np.zeros(len(lc), dtype="int32")
    df = pd.DataFrame({
        "time": np.asarray(lc.time.value, dtype="float64"),
        "flux": np.asarray(lc.flux.value, dtype="float64"),
        "flux_err": np.asarray(lc.flux_err.value, dtype="float64"),
        "quality": quality,
    })
    df.to_parquet(path, index=False)
    shutil.rmtree(CACHE, ignore_errors=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    return path, df["time"].to_numpy()


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    labels = pd.read_parquet(ROOT / "data" / "processed" / "toi_labels.parquet")
    labels["signal_id"] = (labels["tic_id"].astype(int).astype(str)
                           + "_" + labels["toi"].astype(str))

    done = set()
    if LOG.exists():
        old = pd.read_csv(LOG)
        done = set(old.loc[old["status"] != "error", "signal_id"])

    tics = labels["tic_id"].drop_duplicates().sample(frac=1, random_state=42).tolist()
    if limit:
        tics = tics[:limit]

    start = time.time()
    for n, tic in enumerate(tics, 1):
        group = labels[labels["tic_id"] == tic]
        todo = group[~group["signal_id"].isin(done)]
        if todo.empty:
            continue

        res, sectors, search_error = None, [], None
        try:
            res, sectors = get_sector_list(int(tic))
        except Exception as e:
            search_error = str(e)[:200]

        rows = []
        for _, r in todo.iterrows():
            rec = {"signal_id": r["signal_id"], "tic_id": int(tic), "toi": r["toi"],
                   "label": int(r["label"]), "status": "", "sector": np.nan,
                   "n_transits": 0, "file": "", "note": ""}
            period = float(r["period"])
            epoch = float(r["epoch"]) - BTJD_OFFSET
            dur_d = float(r["duration_h"]) / 24.0

            if search_error:
                rec.update(status="error", note=search_error)
            elif not sectors:
                rec.update(status="no_data")
            elif period * (MIN_TRANSITS - 1) > MAX_SPAN_D:
                rec.update(status="too_few_transits", note="period too long for one sector")
            else:
                rec["status"] = "too_few_transits"
                try:
                    for sector, idx in sectors:
                        path, t = load_sector(int(tic), res, sector, idx)
                        nt = count_transits(t, period, epoch, dur_d)
                        rec["n_transits"] = max(rec["n_transits"], nt)
                        if nt >= MIN_TRANSITS:
                            rec.update(status="ok", sector=sector, n_transits=nt,
                                       file=str(path.relative_to(ROOT)))
                            break
                except Exception as e:
                    rec.update(status="error", note=str(e)[:200])
            rows.append(rec)

        pd.DataFrame(rows).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
        done.update(x["signal_id"] for x in rows if x["status"] != "error")
        statuses = ", ".join(x["status"] for x in rows)
        print(f"[{n}/{len(tics)}] TIC {tic}: {statuses} ({(time.time() - start) / 60:.1f} min)")

    log = pd.read_csv(LOG).drop_duplicates("signal_id", keep="last")
    print("\nStatus counts:")
    print(log["status"].value_counts())
    print("\nBy label:")
    print(pd.crosstab(log["label"], log["status"]))


if __name__ == "__main__":
    main()
import sys
import time

import numpy as np
import pandas as pd

from download_lightcurves import (BTJD_OFFSET, MAX_SPAN_D, MIN_TRANSITS, ROOT,
                                  count_transits, get_sector_list, load_sector)

PROCESSED = ROOT / "data" / "processed"
LOG = PROCESSED / "download_log_candidates.csv"


def find_sector(tic, res, sectors, period, epoch, dur_d):
    """First sector with at least MIN_TRANSITS transits, as in the labeled downloader."""
    rec = {"status": "too_few_transits", "sector": np.nan, "n_transits": 0,
           "file": "", "note": ""}
    if period * (MIN_TRANSITS - 1) > MAX_SPAN_D:
        rec["note"] = "period too long for one sector"
        return rec
    for sector, idx in sectors:
        path, t = load_sector(tic, res, sector, idx)
        n_transits = count_transits(t, period, epoch, dur_d)
        rec["n_transits"] = max(rec["n_transits"], n_transits)
        if n_transits >= MIN_TRANSITS:
            rec.update(status="ok", sector=sector, n_transits=n_transits,
                       file=str(path.relative_to(ROOT)))
            break
    return rec


def main():
    n_signals = int(sys.argv[1]) if len(sys.argv) > 1 else 600

    cand = pd.read_parquet(PROCESSED / "toi_candidates.parquet")
    cand["signal_id"] = (cand["tic_id"].astype(int).astype(str)
                         + "_" + cand["toi"].astype(str))
    cand = cand.sample(frac=1, random_state=42).head(n_signals)
    print(f"Candidates selected: {len(cand)} signals from {cand['tic_id'].nunique()} stars")

    done = set()
    if LOG.exists():
        old = pd.read_csv(LOG)
        done = set(old.loc[old["status"] != "error", "signal_id"])

    start = time.time()
    for n, tic in enumerate(cand["tic_id"].drop_duplicates().tolist(), 1):
        todo = cand[(cand["tic_id"] == tic) & ~cand["signal_id"].isin(done)]
        if todo.empty:
            continue

        res, sectors, search_error = None, [], None
        try:
            res, sectors = get_sector_list(int(tic))
        except Exception as e:
            search_error = str(e)[:200]

        rows = []
        for _, r in todo.iterrows():
            base = {"signal_id": r["signal_id"], "tic_id": int(tic), "toi": r["toi"]}
            if search_error:
                rec = {"status": "error", "sector": np.nan, "n_transits": 0,
                       "file": "", "note": search_error}
            elif not sectors:
                rec = {"status": "no_data", "sector": np.nan, "n_transits": 0,
                       "file": "", "note": ""}
            else:
                try:
                    rec = find_sector(int(tic), res, sectors, float(r["period"]),
                                      float(r["epoch"]) - BTJD_OFFSET,
                                      float(r["duration_h"]) / 24.0)
                except Exception as e:
                    rec = {"status": "error", "sector": np.nan, "n_transits": 0,
                           "file": "", "note": str(e)[:200]}
            rows.append({**base, **rec})

        pd.DataFrame(rows).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
        done.update(x["signal_id"] for x in rows if x["status"] != "error")
        statuses = ", ".join(x["status"] for x in rows)
        print(f"[{n}] TIC {tic}: {statuses} ({(time.time() - start) / 60:.1f} min)")

    log = pd.read_csv(LOG).drop_duplicates("signal_id", keep="last")
    print("\nStatus counts:")
    print(log["status"].value_counts())


if __name__ == "__main__":
    main()
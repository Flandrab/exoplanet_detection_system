import pandas as pd

from make_views import PROCESSED, process_signal


def main():
    log = pd.read_csv(PROCESSED / "download_log_candidates.csv")
    log = log.drop_duplicates("signal_id", keep="last")
    log = log[log["status"] == "ok"]

    cand = pd.read_parquet(PROCESSED / "toi_candidates.parquet")
    cand["signal_id"] = (cand["tic_id"].astype(int).astype(str)
                         + "_" + cand["toi"].astype(str))
    cols = ["signal_id", "period", "epoch", "duration_h", "depth_ppm", "tmag", "disposition"]
    data = log.merge(cand[cols], on="signal_id", how="inner")
    data["label"] = -1  # placeholder only: candidates have no label, dropped below
    print(f"Candidate signals to process: {len(data)}")

    rows, failures = [], []
    for n, (_, r) in enumerate(data.iterrows(), 1):
        try:
            rows.append(process_signal(r))
        except Exception as e:
            failures.append({"signal_id": r["signal_id"], "error": str(e)[:200]})
        if n % 100 == 0:
            print(f"  {n}/{len(data)} done")

    df = pd.DataFrame(rows).drop(columns=["label"])
    df.to_parquet(PROCESSED / "views_candidates.parquet", index=False)
    pd.DataFrame(failures).to_csv(PROCESSED / "views_candidates_failures.csv", index=False)

    print(f"\nSaved {len(df)} candidates to data/processed/views_candidates.parquet")
    print(f"Failures: {len(failures)} (see data/processed/views_candidates_failures.csv)")
    print("\nEmpty bins (global / local), median and max:")
    print(df[["empty_global", "empty_local"]].describe().loc[["50%", "max"]])
    print("\nMedian SNR, observed depth and noise:")
    print(df[["snr", "depth_obs_ppt", "noise_ppt"]].median().round(2).to_string())


if __name__ == "__main__":
    main()
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.metrics import precision_recall_curve, roc_curve
from werkzeug.exceptions import BadRequest, HTTPException, NotFound

import db

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "processing"))
from lc_tools import bin_view, clip_upper_outliers, detrend, fold_phase  # noqa: E402

LC_DIR = (ROOT / "data" / "raw" / "lightcurves").resolve()
BTJD_OFFSET = 2457000.0
MAX_POINTS = 4000

MODEL_COLUMNS = ("model_id, name, feature_set, model_file, cv_pr_auc, cv_pr_auc_std, "
                 "cv_roc_auc, cv_f1, test_precision, test_recall, test_f1, test_roc_auc, "
                 "test_pr_auc, tn, fp, fn, tp")

SORTS = {
    "prob_desc": "p.probability DESC NULLS LAST",
    "prob_asc": "p.probability ASC NULLS LAST",
    "snr_desc": "f.snr DESC NULLS LAST",
    "depth_desc": "f.depth_obs_ppt DESC NULLS LAST",
    "period_asc": "s.period ASC",
    "period_desc": "s.period DESC",
}

SIGNALS_FROM = """
FROM signals s
JOIN stars st ON st.tic_id = s.tic_id
JOIN features f ON f.signal_id = s.signal_id
LEFT JOIN predictions p ON p.signal_id = s.signal_id AND p.model_id = :model_id
"""

app = Flask(__name__)
app.json.sort_keys = False


@app.errorhandler(HTTPException)
def handle_http_error(e):
    return jsonify({"error": e.description}), e.code


@app.errorhandler(Exception)
def handle_unexpected_error(e):
    app.logger.exception("Unhandled error")
    return jsonify({"error": f"internal error: {e}"}), 500


# ---------- helpers ----------

def records(df):
    """DataFrame to a list of dicts with NaN replaced by None (valid JSON)."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


def int_arg(name, default, lo, hi):
    raw = request.args.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise BadRequest(f"{name} must be an integer")
    if not lo <= value <= hi:
        raise BadRequest(f"{name} must be between {lo} and {hi}")
    return value


def float_arg(name):
    raw = request.args.get(name)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        raise BadRequest(f"{name} must be a number")


def thin(n, max_points=MAX_POINTS):
    """Indices that keep at most max_points evenly spaced points."""
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points).astype(int)


def as_list(values, ndigits=5):
    return [round(float(v), ndigits) if np.isfinite(v) else None for v in values]


def default_model_id():
    df = db.query_df("SELECT model_id FROM models "
                     "WHERE name = 'random_forest' AND feature_set = 'all'")
    return int(df["model_id"].iloc[0]) if len(df) else None


def signal_filters():
    where, params = [], {}
    kind = request.args.get("kind")
    if kind:
        if kind not in ("labeled", "candidate"):
            raise BadRequest("kind must be labeled or candidate")
        where.append("s.kind = :kind")
        params["kind"] = kind
    label = request.args.get("label")
    if label not in (None, ""):
        if label not in ("0", "1"):
            raise BadRequest("label must be 0 or 1")
        where.append("s.label = :label")
        params["label"] = int(label)
    split = request.args.get("split")
    if split:
        if split not in ("train", "test"):
            raise BadRequest("split must be train or test")
        where.append("s.split = :split")
        params["split"] = split
    disposition = request.args.get("disposition")
    if disposition:
        disposition = disposition.upper()
        if disposition not in ("CP", "KP", "FP", "FA", "PC"):
            raise BadRequest("disposition must be one of CP, KP, FP, FA, PC")
        where.append("s.disposition = :disposition")
        params["disposition"] = disposition
    tic_id = int_arg("tic_id", None, 1, 10**12)
    if tic_id is not None:
        where.append("s.tic_id = :tic_id")
        params["tic_id"] = tic_id
    min_prob = float_arg("min_prob")
    if min_prob is not None:
        where.append("p.probability >= :min_prob")
        params["min_prob"] = min_prob
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    return clause, params


# ---------- routes ----------

@app.get("/api/health")
def health():
    db.query_df("SELECT 1 AS ok")
    return jsonify({"status": "ok"})


@app.get("/api/summary")
def summary():
    counts = db.query_df(
        "SELECT (SELECT COUNT(*) FROM stars) AS stars, "
        "(SELECT COUNT(*) FROM signals) AS signals, "
        "(SELECT COUNT(*) FROM models) AS models, "
        "(SELECT COUNT(*) FROM predictions) AS predictions")
    groups = db.query_df(
        "SELECT kind, label, split, COUNT(*) AS n FROM signals "
        "GROUP BY kind, label, split ORDER BY kind, label, split")
    return jsonify({"counts": records(counts)[0], "groups": records(groups)})


@app.get("/api/models")
def models():
    df = db.query_df(f"SELECT {MODEL_COLUMNS} FROM models ORDER BY test_pr_auc DESC")
    return jsonify({"models": records(df)})


@app.get("/api/signals")
def signals():
    model_id = int_arg("model_id", default_model_id(), 1, 10**9)
    if model_id is None:
        raise NotFound("no models in the database")
    sort = request.args.get("sort", "prob_desc")
    if sort not in SORTS:
        raise BadRequest(f"sort must be one of {', '.join(SORTS)}")
    limit = int_arg("limit", 50, 1, 500)
    offset = int_arg("offset", 0, 0, 10**7)
    clause, params = signal_filters()
    params["model_id"] = model_id

    total = db.query_df("SELECT COUNT(*) AS n " + SIGNALS_FROM + clause, params)["n"].iloc[0]
    df = db.query_df(
        "SELECT s.signal_id, s.tic_id, s.toi, s.kind, s.disposition, s.label, s.split, "
        "s.period, s.duration_h, f.depth_obs_ppt, f.snr, st.tmag, p.probability "
        + SIGNALS_FROM + clause
        + f" ORDER BY {SORTS[sort]}, s.signal_id LIMIT :limit OFFSET :offset",
        {**params, "limit": limit, "offset": offset})
    return jsonify({"total": int(total), "limit": limit, "offset": offset,
                    "model_id": model_id, "signals": records(df)})


@app.get("/api/signals/<signal_id>")
def signal_detail(signal_id):
    sig = db.query_df("SELECT s.*, st.tmag FROM signals s "
                      "JOIN stars st ON st.tic_id = s.tic_id WHERE s.signal_id = :sid",
                      {"sid": signal_id})
    if sig.empty:
        raise NotFound("unknown signal")
    feats = db.query_df("SELECT * FROM features WHERE signal_id = :sid", {"sid": signal_id})
    preds = db.query_df(
        "SELECT m.model_id, m.name, m.feature_set, p.probability "
        "FROM predictions p JOIN models m ON m.model_id = p.model_id "
        "WHERE p.signal_id = :sid ORDER BY m.model_id", {"sid": signal_id})
    return jsonify({"signal": records(sig)[0],
                    "features": records(feats)[0] if len(feats) else None,
                    "predictions": records(preds)})


@app.get("/api/signals/<signal_id>/lightcurve")
def lightcurve(signal_id):
    row = db.query_df("SELECT period, epoch_bjd, duration_h, lightcurve_file "
                      "FROM signals WHERE signal_id = :sid", {"sid": signal_id})
    if row.empty:
        raise NotFound("unknown signal")
    values = row[["period", "epoch_bjd", "duration_h"]].to_numpy(dtype="float64")[0]
    file = row["lightcurve_file"].iloc[0]
    if not file or not np.all(np.isfinite(values)):
        raise NotFound("no light curve for this signal")
    period, epoch, dur_d = values[0], values[1] - BTJD_OFFSET, values[2] / 24.0

    path = (ROOT / file).resolve()
    if LC_DIR not in path.parents or not path.exists():
        raise NotFound("light curve file not found")

    lc = pd.read_parquet(path, columns=["time", "flux"])
    t = lc["time"].to_numpy(dtype="float64")
    f = lc["flux"].to_numpy(dtype="float64")
    good = np.isfinite(t) & np.isfinite(f)
    t, f = t[good], f[good]
    f = f / np.median(f)

    flat = detrend(t, f, period, epoch, dur_d)
    phase = fold_phase(t, period, epoch)
    keep = clip_upper_outliers(flat, phase, dur_d)
    tk, fk, pk = t[keep], flat[keep], phase[keep]

    glob, _ = bin_view(pk, fk, -period / 2, period / 2, 2001)
    half = min(max(3.0 * dur_d, 0.1), 0.5 * period)
    loc, _ = bin_view(pk, fk, -half, half, 201)

    i_raw, i_flat = thin(len(t)), thin(len(tk))
    k_min = int(np.ceil((t.min() - epoch) / period))
    k_max = int(np.floor((t.max() - epoch) / period))
    return jsonify({
        "signal_id": signal_id,
        "period": period,
        "epoch_btjd": epoch,
        "duration_h": dur_d * 24.0,
        "transit_times": as_list([epoch + k * period for k in range(k_min, k_max + 1)]),
        "raw": {"time": as_list(t[i_raw]), "flux": as_list(f[i_raw], 6)},
        "flattened": {"time": as_list(tk[i_flat]), "flux": as_list(fk[i_flat], 6)},
        "folded": {"phase": as_list(pk[i_flat]), "flux": as_list(fk[i_flat], 6)},
        "global_view": as_list((glob - 1.0) * 1000.0, 4),
        "local_view": as_list((loc - 1.0) * 1000.0, 4),
        "local_half_width_days": half,
    })


@app.get("/api/compare")
def compare():
    mdf = db.query_df(f"SELECT {MODEL_COLUMNS} FROM models ORDER BY test_pr_auc DESC")
    pdf = db.query_df(
        "SELECT p.model_id, p.probability, s.label FROM predictions p "
        "JOIN signals s ON s.signal_id = p.signal_id WHERE s.split = 'test'")
    out = []
    for rec in records(mdf):
        g = pdf[pdf["model_id"] == rec["model_id"]]
        item = dict(rec)
        if len(g) and g["label"].nunique() == 2:
            y, p = g["label"].to_numpy(dtype=int), g["probability"].to_numpy(dtype=float)
            fpr, tpr, _ = roc_curve(y, p)
            prec, rec_, _ = precision_recall_curve(y, p)
            i, j = thin(len(fpr), 200), thin(len(prec), 200)
            item["roc"] = {"fpr": as_list(fpr[i], 4), "tpr": as_list(tpr[i], 4)}
            item["pr"] = {"recall": as_list(rec_[j], 4), "precision": as_list(prec[j], 4)}
        out.append(item)
    return jsonify({"models": out})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
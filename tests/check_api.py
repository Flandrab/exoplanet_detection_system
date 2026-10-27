import sys

import requests

BASE = "http://127.0.0.1:5000"
EXPECTED_TOP5 = [432078760, 941340744, 265979849, 427730490, 115583726]
LIGHTCURVE_ID = "432078760_3765.01"

results = []


def get(path):
    return requests.get(BASE + path, timeout=120)


def get_json(path):
    r = get(path)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {path}: {r.text[:200]}")
    return r.json()


def check(name, fn):
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"error: {e}"
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


def t_health():
    status = get_json("/api/health").get("status")
    return status == "ok", f"status={status}"


def t_summary():
    c = get_json("/api/summary")["counts"]
    ok = (c["signals"] == 2825 and c["models"] == 6 and c["predictions"] == 16950
          and c["stars"] >= 2256)
    return ok, f"{c} (expected signals 2825, models 6, predictions 16950, stars >= 2256)"


def t_models():
    models = get_json("/api/models")["models"]
    first = models[0]
    ok = (len(models) == 6 and first["name"] == "random_forest"
          and first["feature_set"] == "all" and abs(first["test_pr_auc"] - 0.8507) < 0.001)
    return ok, (f"{len(models)} models, best = {first['name']}/{first['feature_set']} "
                f"test_pr_auc={first['test_pr_auc']:.4f} (expected 6, random_forest/all, 0.8507)")


def t_candidates():
    r = get_json("/api/signals?kind=candidate&limit=5")
    tics = [s["tic_id"] for s in r["signals"]]
    ok = r["total"] == 466 and tics == EXPECTED_TOP5
    return ok, f"total={r['total']}, top5={tics} (expected 466 and {EXPECTED_TOP5})"


def t_paging():
    r = get_json("/api/signals?kind=candidate&limit=2&offset=1")
    tics = [s["tic_id"] for s in r["signals"]]
    return tics == EXPECTED_TOP5[1:3], f"{tics} (expected {EXPECTED_TOP5[1:3]})"


def t_test_split():
    planets = get_json("/api/signals?kind=labeled&label=1&split=test")["total"]
    false_pos = get_json("/api/signals?kind=labeled&label=0&split=test")["total"]
    return (planets, false_pos) == (257, 216), f"planets={planets}, false positives={false_pos} (expected 257 and 216)"


def t_lightcurve():
    j = get_json(f"/api/signals/{LIGHTCURVE_ID}/lightcurve")
    n_glob, n_loc = len(j["global_view"]), len(j["local_view"])
    n_raw, n_tr = len(j["raw"]["time"]), len(j["transit_times"])
    ok = (n_glob == 2001 and n_loc == 201 and 0 < n_raw <= 4000
          and 6 <= n_tr <= 10 and abs(j["period"] - 3.148) < 0.001)
    return ok, (f"global={n_glob}, local={n_loc}, raw points={n_raw}, transits={n_tr}, "
                f"period={j['period']:.4f} (expected 2001, 201, <=4000, 6 to 10, 3.148)")


def t_compare():
    models = get_json("/api/compare")["models"]
    ok = len(models) == 6 and all("roc" in m and "pr" in m for m in models)
    return ok, f"{len(models)} models, all with ROC and PR curves: {all('roc' in m and 'pr' in m for m in models)}"


def t_errors():
    a = get("/api/signals?kind=x")
    b = get("/api/signals/nope")
    return (a.status_code, b.status_code) == (400, 404), (
        f"invalid kind -> {a.status_code} {a.json()}, unknown signal -> {b.status_code} {b.json()}")


try:
    get("/api/health")
except requests.exceptions.ConnectionError:
    print(f"The API is not reachable at {BASE}. Start it first with: python api/app.py")
    sys.exit(2)

check("health", t_health)
check("summary counts", t_summary)
check("models", t_models)
check("top five candidates", t_candidates)
check("paging", t_paging)
check("labeled test split", t_test_split)
check("light curve", t_lightcurve)
check("model comparison", t_compare)
check("error handling", t_errors)

print(f"\nResult: {sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
import numpy as np


def fold_phase(t, period, epoch):
    """Time from the nearest transit centre, in days, in [-period/2, period/2)."""
    return (t - epoch + 0.5 * period) % period - 0.5 * period


def robust_sigma(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def detrend(t, f, period, epoch, dur_d, gap=0.5, step=5):
    """Divide out a running median that ignores in-transit points.

    The light curve is split at gaps longer than `gap` days (the orbit gap),
    and each piece is detrended on its own.
    """
    window = max(3.0 * dur_d, 0.5)
    in_transit = np.abs(fold_phase(t, period, epoch)) < dur_d
    trend = np.full(len(t), np.nan)
    breaks = np.where(np.diff(t) > gap)[0] + 1
    for seg in np.split(np.arange(len(t)), breaks):
        ts, fs, ms = t[seg], f[seg], in_transit[seg]
        centers = np.arange(0, len(seg), step)
        lo = np.searchsorted(ts, ts[centers] - window / 2)
        hi = np.searchsorted(ts, ts[centers] + window / 2)
        vals = np.full(len(centers), np.nan)
        for j in range(len(centers)):
            chunk = fs[lo[j]:hi[j]][~ms[lo[j]:hi[j]]]
            if chunk.size >= 10:
                vals[j] = np.median(chunk)
        good = ~np.isnan(vals)
        if good.sum() >= 2:
            trend[seg] = np.interp(ts, ts[centers][good], vals[good])
    return f / trend


def clip_upper_outliers(flat, phase, dur_d, n_sigma=5.0):
    """Boolean mask of points to keep: drops points far ABOVE the baseline."""
    out_of_transit = np.abs(phase) >= dur_d
    sigma = robust_sigma(flat[out_of_transit])
    if not np.isfinite(sigma) or sigma == 0:
        return np.isfinite(flat)
    return np.isfinite(flat) & (flat < 1.0 + n_sigma * sigma)


def bin_view(phase, flux, lo, hi, n_bins):
    """Median-bin flux over phase in [lo, hi); empty bins are interpolated."""
    idx = np.floor((phase - lo) / (hi - lo) * n_bins).astype(int)
    keep = (idx >= 0) & (idx < n_bins)
    idx, fl = idx[keep], flux[keep]
    view = np.full(n_bins, np.nan)
    if idx.size == 0:
        return view, n_bins
    order = np.argsort(idx, kind="stable")
    idx, fl = idx[order], fl[order]
    uniq, start = np.unique(idx, return_index=True)
    for u, chunk in zip(uniq, np.split(fl, start[1:])):
        view[u] = np.median(chunk)
    empty = np.isnan(view)
    n_empty = int(empty.sum())
    if n_empty and n_empty < n_bins:
        pos = np.arange(n_bins)
        view[empty] = np.interp(pos[empty], pos[~empty], view[~empty])
    return view, n_empty


def compute_features(t, flat, phase, period, epoch, dur_d):
    """Simple vetting features. Depths and noise are in parts per thousand."""
    out_of_transit = np.abs(phase) >= dur_d
    sigma = robust_sigma(flat[out_of_transit]) * 1000.0
    inner = np.abs(phase) < dur_d / 4.0
    n_in = int(inner.sum())
    feats = {"noise_ppt": sigma, "n_in_transit": n_in}

    def depth(mask):
        return (1.0 - np.median(flat[mask])) * 1000.0

    if n_in >= 5:
        d = depth(inner)
        feats["depth_obs_ppt"] = d
        feats["snr"] = d / sigma * np.sqrt(n_in)
    else:
        feats["depth_obs_ppt"] = np.nan
        feats["snr"] = np.nan

    k = np.round((t - epoch) / period).astype(int)
    odd, even = inner & (k % 2 == 1), inner & (k % 2 == 0)
    n_odd_tr = len(np.unique(k[odd]))
    n_even_tr = len(np.unique(k[even]))
    if n_odd_tr >= 2 and n_even_tr >= 2 and odd.sum() >= 5 and even.sum() >= 5:
        se = sigma * np.sqrt(1.0 / odd.sum() + 1.0 / even.sum())
        feats["odd_even_sigma"] = abs(depth(odd) - depth(even)) / se
    else:
        feats["odd_even_sigma"] = np.nan

    sec = np.abs(np.abs(phase) - period / 2.0) < dur_d / 4.0
    if sec.sum() >= 5:
        feats["secondary_ppt"] = depth(sec)
        feats["secondary_snr"] = feats["secondary_ppt"] / sigma * np.sqrt(sec.sum())
    else:
        feats["secondary_ppt"] = np.nan
        feats["secondary_snr"] = np.nan
    return feats
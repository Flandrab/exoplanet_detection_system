import numpy as np

from processing.lc_tools import (bin_view, clip_upper_outliers, compute_features, detrend,
                                 fold_phase, robust_sigma)

PERIOD, EPOCH, DUR = 3.0, 1.0, 0.1  # days


def make_light_curve(depth=0.01, slope=0.0, noise=0.0005, seed=0):
    """Two-minute cadence over 27 days with a box-shaped transit every PERIOD days."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 27, 2 / 1440)
    flux = 1.0 + slope * t + rng.normal(0, noise, len(t))
    flux[np.abs(fold_phase(t, PERIOD, EPOCH)) < DUR / 2] -= depth
    return t, flux


def test_fold_phase_range_and_center():
    t = np.linspace(0, 30, 5000)
    phase = fold_phase(t, PERIOD, EPOCH)
    assert phase.min() >= -PERIOD / 2 - 1e-9
    assert phase.max() < PERIOD / 2
    assert abs(fold_phase(np.array([EPOCH + 2 * PERIOD]), PERIOD, EPOCH)[0]) < 1e-9


def test_robust_sigma_ignores_outliers_and_nans():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 2.0, 5000)
    x[:20] = 1000.0
    x[20:30] = np.nan
    assert abs(robust_sigma(x) - 2.0) < 0.15
    assert np.isnan(robust_sigma(np.array([np.nan, np.nan])))


def test_detrend_removes_slow_trend_and_keeps_transit():
    t, flux = make_light_curve(depth=0.01, slope=0.002)
    flat = detrend(t, flux, PERIOD, EPOCH, DUR)
    phase = fold_phase(t, PERIOD, EPOCH)
    out = np.isfinite(flat) & (np.abs(phase) > DUR)
    assert abs(np.median(flat[out]) - 1.0) < 1e-3
    assert flat[out].std() < 0.002  # the trend (0.054 over the run) is gone
    in_transit = np.isfinite(flat) & (np.abs(phase) < DUR / 4)
    assert 1.0 - np.median(flat[in_transit]) > 0.008  # the dip survives


def test_detrend_handles_gap_between_orbits():
    t, flux = make_light_curve()
    keep = (t < 12) | (t > 13.5)  # 1.5 day gap, like the TESS orbit gap
    t, flux = t[keep], flux[keep]
    flat = detrend(t, flux, PERIOD, EPOCH, DUR)
    assert np.isfinite(flat).mean() > 0.95


def test_clip_upper_outliers_removes_high_points_only():
    t, flux = make_light_curve(depth=0.01)
    flux[100] += 0.05   # flare-like point above the baseline
    flux[200] -= 0.05   # point far below the baseline (kept: could be a transit)
    phase = fold_phase(t, PERIOD, EPOCH)
    keep = clip_upper_outliers(flux, phase, DUR)
    assert not keep[100]
    assert keep[200]
    assert keep.mean() > 0.99


def test_bin_view_shape_and_median():
    phase = np.array([-0.45, -0.44, 0.05, 0.06, 0.45])
    flux = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    view, n_empty = bin_view(phase, flux, -0.5, 0.5, 10)
    assert view.shape == (10,)
    assert view[0] == 2.0          # median of 1 and 3 in the first bin
    assert view[5] == 6.0          # median of 5 and 7
    assert np.isfinite(view).all()  # empty bins are interpolated
    assert n_empty == 7


def test_bin_view_all_empty():
    view, n_empty = bin_view(np.array([5.0]), np.array([1.0]), -0.5, 0.5, 20)
    assert n_empty == 20
    assert np.isnan(view).all()


def test_compute_features_on_clean_transit():
    t, flux = make_light_curve(depth=0.01, noise=0.0005)
    phase = fold_phase(t, PERIOD, EPOCH)
    f = compute_features(t, flux, phase, PERIOD, EPOCH, DUR)
    assert abs(f["depth_obs_ppt"] - 10.0) < 0.5          # 1% = 10 ppt
    assert abs(f["noise_ppt"] - 0.5) < 0.1
    assert f["snr"] > 50
    assert f["odd_even_sigma"] < 3                        # same depth on odd and even
    assert abs(f["secondary_ppt"]) < 0.5                  # no secondary eclipse


def test_compute_features_detects_odd_even_difference():
    t, flux = make_light_curve(depth=0.01, noise=0.0005)
    k = np.round((t - EPOCH) / PERIOD).astype(int)
    phase = fold_phase(t, PERIOD, EPOCH)
    in_tr = np.abs(phase) < DUR / 2
    flux[in_tr & (k % 2 == 1)] -= 0.01  # odd transits twice as deep: an eclipsing binary
    f = compute_features(t, flux, phase, PERIOD, EPOCH, DUR)
    assert f["odd_even_sigma"] > 10


def test_compute_features_too_few_points_gives_nan():
    t = np.arange(0, 3, 0.1)
    flat = np.ones(len(t))
    phase = fold_phase(t, 100.0, 1.0)
    f = compute_features(t, flat, phase, 100.0, 1.0, 0.01)
    assert np.isnan(f["snr"]) and np.isnan(f["odd_even_sigma"]) and np.isnan(f["secondary_snr"])
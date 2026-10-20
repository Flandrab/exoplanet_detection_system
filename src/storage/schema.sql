-- Schema for the exoplanet detection system.
-- The loader (load_db.py) drops and recreates these tables, then fills them
-- from the Parquet and CSV files, which are the source of truth.

CREATE TABLE stars (
    tic_id  BIGINT PRIMARY KEY,
    tmag    DOUBLE PRECISION
);

CREATE TABLE signals (
    signal_id        TEXT PRIMARY KEY,
    tic_id           BIGINT NOT NULL REFERENCES stars (tic_id),
    toi              DOUBLE PRECISION,
    kind             TEXT NOT NULL DEFAULT 'labeled'
                     CHECK (kind IN ('labeled', 'candidate')),
    disposition      TEXT,
    label            SMALLINT CHECK (label IN (0, 1)),
    split            TEXT CHECK (split IN ('train', 'test')),
    period           DOUBLE PRECISION NOT NULL,
    epoch_bjd        DOUBLE PRECISION,
    duration_h       DOUBLE PRECISION,
    depth_cat_ppm    DOUBLE PRECISION,
    sector           INTEGER,
    n_transits       INTEGER,
    lightcurve_file  TEXT
);

CREATE TABLE features (
    signal_id       TEXT PRIMARY KEY REFERENCES signals (signal_id) ON DELETE CASCADE,
    noise_ppt       DOUBLE PRECISION,
    n_in_transit    INTEGER,
    depth_obs_ppt   DOUBLE PRECISION,
    snr             DOUBLE PRECISION,
    odd_even_sigma  DOUBLE PRECISION,
    secondary_ppt   DOUBLE PRECISION,
    secondary_snr   DOUBLE PRECISION,
    n_points        INTEGER,
    empty_global    INTEGER,
    empty_local     INTEGER
);

CREATE TABLE models (
    model_id         INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name             TEXT NOT NULL,
    feature_set      TEXT NOT NULL,
    model_file       TEXT,
    created_at       TIMESTAMP NOT NULL DEFAULT now(),
    cv_pr_auc        DOUBLE PRECISION,
    cv_pr_auc_std    DOUBLE PRECISION,
    cv_roc_auc       DOUBLE PRECISION,
    cv_f1            DOUBLE PRECISION,
    test_precision   DOUBLE PRECISION,
    test_recall      DOUBLE PRECISION,
    test_f1          DOUBLE PRECISION,
    test_roc_auc     DOUBLE PRECISION,
    test_pr_auc      DOUBLE PRECISION,
    tn               INTEGER,
    fp               INTEGER,
    fn               INTEGER,
    tp               INTEGER,
    UNIQUE (name, feature_set)
);

CREATE TABLE predictions (
    model_id     INTEGER NOT NULL REFERENCES models (model_id) ON DELETE CASCADE,
    signal_id    TEXT NOT NULL REFERENCES signals (signal_id) ON DELETE CASCADE,
    probability  DOUBLE PRECISION NOT NULL CHECK (probability BETWEEN 0 AND 1),
    created_at   TIMESTAMP NOT NULL DEFAULT now(),
    PRIMARY KEY (model_id, signal_id)
);

CREATE INDEX idx_signals_tic   ON signals (tic_id);
CREATE INDEX idx_signals_label ON signals (label);
CREATE INDEX idx_signals_split ON signals (split);
CREATE INDEX idx_predictions_signal ON predictions (signal_id);
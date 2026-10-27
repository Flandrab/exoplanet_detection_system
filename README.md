# Exoplanet detection from TESS data

An end-to-end system that finds planet-like dimming events in light curves from NASA's TESS
telescope. It downloads real TESS data, cleans and processes it, stores it in Parquet files
and PostgreSQL, trains and compares machine learning models, and shows the results in a web
interface.

This is a thesis project (AI and Data Engineering). A high score from a model is a
**ranking of how planet-like a signal looks, not a discovery**.

## What it does

1. **Collects** labels from the NASA Exoplanet Archive TOI catalog (confirmed and known
   planets versus false positives) and the matching 2-minute light curves of the TESS SPOC
   pipeline from MAST.
2. **Processes** each signal: removes the slow stellar trend, folds the light curve on the
   catalog period, builds a global view (2001 bins) and a local view (201 bins) of the
   transit, and computes vetting features (depth, signal-to-noise, odd-even difference,
   secondary eclipse, and others).
3. **Stores** large arrays in Parquet files and the structured data (stars, signals,
   features, models, predictions) in PostgreSQL.
4. **Trains and compares** seven models on one fixed split in which no star appears in both
   training and test: logistic regression, random forest and gradient boosting (each with
   all features and without the period features; the period ablation covers the classical
   models), and a 1D CNN on the folded views.
5. **Scores** unresolved TOI candidates, which have no ground truth.
6. **Serves** everything through a Flask REST API and a Streamlit app with interactive
   light curves, model predictions and a model comparison.

## Results

Held-out test set: 473 signals from stars never used in training (257 planets, so a random
classifier has a PR-AUC of 0.543).

| Model | Features | Test PR-AUC |
|---|---|---|
| Random forest | all | 0.851 |
| Gradient boosting | all | 0.848 |
| Random forest | without period | 0.826 |
| Gradient boosting | without period | 0.820 |
| 1D CNN | light-curve views | 0.760 |
| Logistic regression | all | 0.714 |
| Logistic regression | without period | 0.699 |

Averaging the models did not give a significant improvement over the random forest. The
unresolved candidates are fainter and noisier than the labeled signals the models were
trained on, so their scores are less reliable.

## Project structure

```
data/raw/          untouched downloads (TOI catalog, light curves)      (not in Git)
data/processed/    Parquet and CSV files built by the pipeline          (not in Git)
data/models/       trained models                                       (not in Git)
src/ingestion/     labels and light curve downloaders
src/processing/    detrending, folding, views and features
src/evaluation/    metrics and the fixed train/test split
src/models/        baselines, CNN, candidate scoring, ensemble experiment
src/storage/       PostgreSQL schema and loader
src/analysis/      exploratory and error analysis
api/               Flask REST API
app/               Streamlit app
tests/             API check
```

## Requirements

- Python 3.10 or newer
- PostgreSQL 16 or newer (a local server)
- Internet access for the downloads (NASA Exoplanet Archive and MAST)
- Several hours for the light curve download (it can be stopped and resumed)

## Setup

Commands are for Windows PowerShell, from the project root.

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Database.** Create a PostgreSQL user and database, for example with pgAdmin or psql:

```
CREATE USER exo_user WITH PASSWORD 'choose_a_password';
CREATE DATABASE exoplanets OWNER exo_user;
```

**The `.env` file.** The scripts read the database connection from a file called `.env`
in the project root. It is not stored in Git because it contains a password. Create it
with one line, using your own password (letters and digits only):

```
DATABASE_URL=postgresql+psycopg2://exo_user:YOUR_PASSWORD@localhost:5432/exoplanets
```

**TOI catalog.** Download the TOI table as CSV from the NASA Exoplanet Archive
(exoplanetarchive.ipac.caltech.edu, table "TESS Project Candidates") and save it in
`data/raw/`. The file name that `src/ingestion/build_labels.py` expects is set at the top of
that script.

## Running the pipeline

Run each script from the project root, in this order. Each step needs the output of the
previous ones.

```
# 1. Labels and light curves
python src/ingestion/build_labels.py
python src/ingestion/download_lightcurves.py      # long; resumable

# 2. Processing and split
python src/processing/make_views.py
python src/evaluation/make_split.py

# 3. Baseline models
python src/models/baseline.py

# 4. Database
python src/storage/load_db.py
python src/models/store_predictions.py

# 5. Unresolved candidates
python src/ingestion/download_candidates.py
python src/processing/make_views_candidates.py
python src/models/score_candidates.py

# 6. 1D CNN (needs PyTorch: pip install torch)
python src/models/cnn.py
python src/models/store_cnn.py

# 7. Analysis (optional)
python src/analysis/eda.py
python src/analysis/error_analysis.py
python src/models/ensemble.py
```

`load_db.py` drops and recreates all tables, so after running it repeat steps 4 to 6 for the
predictions and the CNN.

## Running the app

Use two terminals, both in the project root with the virtual environment active.

```
python api/app.py
```

```
streamlit run app/Home.py
```

The API listens on http://127.0.0.1:5000 and the app opens in the browser. To check that
the API returns what is expected:

```
python tests/check_api.py
```

## Pages of the app

- **Candidate ranking:** signals sorted by the probability of being a planet, with filters
  for the model, the group and the minimum probability.
- **Light curve explorer:** the light curve, the folded transit, the model inputs and the
  predictions of every model for one signal.
- **Model comparison:** metrics, ROC and precision-recall curves, and confusion matrices.

## Limitations

- Only light-curve features are used: there is no pixel-level or centroid information, so
  eclipsing binaries and blended stars cannot be excluded reliably.
- Long-period signals (above 5 days) have few transits and are classified less well.
- Some labeled planets have no visible transit in the processed light curve.
- Only a random sample of the unresolved candidates (600 signals, 466 with usable data) was
  downloaded and scored.

## Data sources

- NASA Exoplanet Archive, TESS Project Candidates (TOI) table
- TESS SPOC 2-minute light curves from the Mikulski Archive for Space Telescopes (MAST),
  accessed with the `lightkurve` package
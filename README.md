# Adversarial Telemetry Intelligence

A beginner research project exploring anomaly detection in public Formula 1 telemetry. The original Streamlit grip-proxy dashboard remains available, alongside a separate experiment that asks:

> How does feature selection affect unsupervised anomaly detection in Formula 1 telemetry?

This is an exploratory educational investigation. An anomaly means a sample differs from patterns learned from earlier telemetry; it is not evidence of a driving mistake, tire failure, or measured tire slip.

## Current project status

Implemented: FastF1 session/lap loading with cache, telemetry-derived features, the original grip-proxy interface, Isolation Forest and robust-z statistical comparison, three feature sets, chronological holdout scoring, CSV/HTML output, and a small anomaly section in Streamlit.

Not yet established: results on a verified collection of sessions, ground-truth labels, independent validation of flagged events, or a scientific claim that this method detects real driving errors. No results are reported here because no live telemetry experiment has yet been run and checked.

## Dataset and access

The experiment obtains public session data using [FastF1](https://docs.fastf1.dev/) at runtime. It selects a driver's fastest available lap and uses its telemetry channels. The first request may need network access and can take time. FastF1's cache is stored in `f1_cache/` beside the process working directory. Do not commit the cache or generated results.

## Features and preprocessing

Feature sets are compared on the same time-ordered eligible rows:

- **A:** Speed, RPM.
- **B:** Speed, RPM, Throttle, Brake (unavailable channels are omitted).
- **C:** Speed, RPM, and all available Throttle, Brake, Gear, Brake_Rate, TyreLife channels.

Numeric infinities are treated as missing. Median imputation is fitted on the training period only, followed by StandardScaler fitted on training values only. Channels entirely missing in training are left out. Rows without Speed or RPM cannot form the required baseline and are excluded.

The telemetry processor additionally estimates longitudinal/lateral acceleration proxies, distance, and driver input rates where the required source channels exist.

## Models and experimental design

The time-ordered first 80% of eligible samples is the training period; the final 20% is held out for scoring. IsolationForest uses 200 trees and random seed 42. The configurable `contamination` setting (default 0.02) tells the algorithm the expected fraction of anomalies in the training data and determines its decision threshold. It is a modeling assumption, not an observed error rate.

The explainable statistical baseline computes a robust standardized deviation (median and scaled median absolute deviation) per feature, then uses the maximum absolute deviation across features as its score. Its flag threshold is the training-score quantile corresponding to contamination. Constant channels receive a safe fallback scale.

## Results to date

No real F1 session has been run through this experiment in this project, so there are no F1 metrics or result plots to report yet. Synthetic data has only been used for software smoke checks; those outputs are not research results and are not included as F1 findings.
## Evaluation and interpretation

The experiment reports holdout anomaly counts and rates for each feature set and method, overlap between methods (Jaccard similarity), and overlap between feature-set results for the same method. The anomaly score is a ranking signal; flagged/unflagged is a thresholded decision. Similarity indicates agreement in sample identities, not correctness. No ground-truth labels are available, so precision, recall, accuracy, and real-world mistake-detection claims are not supported.

Plots show Speed/Throttle/Brake over lap distance (or sample time where distance is absent), Isolation Forest scores and flagged samples, and flagged-rate comparison by feature set and method. A complete scored-sample table is saved as CSV.

## Reproducibility

Requires Python 3.10+ and network access for downloading FastF1 session data.

```powershell
cd outputs
python -m pip install -r requirements.txt
python anomaly_experiment.py --year 2024 --track Monza --session Q --driver LEC --contamination 0.02
python -m streamlit run app.py
```

Replace the example season, event name, session and driver with a combination available in FastF1. Session codes include Q (qualifying) and R (race). Output files are written to `outputs/results/`: `anomaly_scores.csv`, `anomaly_summary.csv`, and interactive HTML plots. The script reports clear errors when FastF1 data or the required Speed/RPM channels are unavailable.

For testing the experiment without network data, import `run_experiment` and pass a pandas DataFrame containing synthetic or locally available telemetry with Speed and RPM columns. Do not present synthetic-data metrics as Formula 1 results.

## Limitations

- Public telemetry channels and sampling are not physical tire measurements.
- The experiment studies one lap at a time and its chronological split may place different corner types or phases in train and test.
- Telemetry samples are temporally dependent; ordinary sample counts overstate the amount of independent evidence.
- Feature-set comparisons may be affected by missing channels, collinearity and different feature dimensionalities.
- Contamination is user supplied; changing it changes thresholds and flagged counts.
- There are no verified labels, so unsupervised flags cannot establish that an event was a genuine driving error.
- The existing grip proxy is exploratory and should not be described as a validated tire-degradation or friction estimator.

## Future research

Evaluate multiple drivers, sessions and circuits; define event-level windows; examine stability across contamination settings; compare unsupervised methods; and seek independently reviewed event labels before making claims about driving errors. Report all exclusions and session selections, and treat repeated telemetry samples as grouped/time-dependent observations.


project/
├── outputs/
│   ├── anomaly_experiment.py
│   ├── app.py
│   ├── requirements.txt
│   └── results/
├── f1_cache/
└── README.md

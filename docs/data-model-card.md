# Data and model card

## Data provenance

The dataset is generated in `generate()` using NumPy's seeded random generator. It represents eight invented machines with hourly temperature, vibration, pressure and load. A persistent latent stress variable influences both sensor values and event probability; a modest final-period shift supplies a drift-monitoring example. This is original simulation code under MIT, with no downloaded industrial data or external data license.

The benchmark's success partly reflects the generator's own assumptions. Real failure definitions, operating regimes, maintenance interventions, missing measurements, downtime and censoring are absent. Reusing the same seed is a reproducibility check, not an independent validation set.

## Model purpose

Binary next-six-hour synthetic event risk, using sixteen causal features. Suitable for demonstrating a testable evaluation pipeline, model inspection and batch integration. Unsuitable for maintenance decisions, operational alarms or claims about cost savings.

The committed seed-42 evaluation uses 2,832 future test rows and gives logistic average precision 0.3265 at a test prevalence of 0.1402. Scores are not calibrated on real observations. F1 optimization assigns equal relative weight to false positives and false negatives and does not encode a business cost function.

## Limitations

- Same assets in train and future test; no leave-asset-out evaluation.
- Overlapping six-hour targets produce serial dependence.
- Only one fixed seed is reported; no uncertainty estimate or multi-seed comparison.
- No missing-data imputation, irregular sampling or concept-drift adaptation.
- No production API, persistent online state, scheduling, model registry or deployed service.
- PSI flags distribution shift but has no validated alert threshold here.

## Reproducibility

Run tests and `python -m industrial_risk demo --seed 42`. The JSON artifacts contain the measured split bounds, model parameters, thresholds, baseline metrics and drift results. Large sensor and prediction CSVs regenerate locally. CI config is provided, but a successful remote run must be verified separately on GitHub.

Before adapting to a real dataset, freeze a failure definition, document sensor availability at prediction time, split by calendar and asset deployment, tune decisions against actual maintenance costs, and evaluate calibration and uncertainty on a later untouched period.

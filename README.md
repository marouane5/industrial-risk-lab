# Industrial Risk Lab

**A reproducible time-series risk-modeling prototype with explicit leakage controls, simple baselines and drift monitoring.**

Generate an hourly synthetic machine-sensor dataset, predict whether a failure event occurs in the next six hours, evaluate on future observations, export a portable JSON model and score unlabeled CSV batches.

This is an **AI-assisted initial portfolio implementation**, built and verified on 16 November 2026. All measurements and failures are synthetic. This is not an industrial deployment, real predictive-maintenance validation or remaining-useful-life estimator.

## Quick start

Python 3.10+ and NumPy:

```console
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m industrial_risk demo
python -m industrial_risk score --input artifacts/synthetic_sensors.csv
```

The demo runs in seconds on a CPU. It creates a synthetic CSV, `evaluation.json`, `model.json`, `drift_reference.json` and test predictions under `artifacts/`. Large generated CSVs are ignored by Git; recreate them with the demo. The checked-in JSON reports make the measured run inspectable.

## What is implemented

- Eight simulated assets, 1,800 hourly samples each; four sensor channels.
- Sixteen causal features: current value, trailing 12-hour mean, standard deviation and trend for each sensor.
- A next-six-hour event target, chronological 60/20/20 split and purging of labels that cross a split boundary.
- Train-only scaling and NumPy logistic regression with L2 regularization.
- Validation-only alarm-threshold selection and comparisons against prevalence and temperature baselines.
- PR average precision, ROC-AUC, precision, recall, F1, Brier score and confusion counts.
- Population Stability Index using fixed train-only quantile bins, portable JSON inference and CSV batch scoring without labels.
- Sixteen unit tests and a CI workflow that reruns tests and the demo.

## Measured synthetic result

Seed 42 produced 8,504 train rows, 2,832 validation rows and 2,832 test rows after warm-up and boundary purging. Test prevalence was 14.02%.

| Test metric | Logistic model | Prevalence baseline | Temperature baseline |
| --- | ---: | ---: | ---: |
| Average precision | 0.3265 | 0.1402 | 0.2852 |
| ROC-AUC | 0.7274 | 0.5000 | 0.7073 |
| F1 at validation-selected threshold | 0.3493 | 0.2459 | 0.3265 |
| Precision | 0.3326 | 0.1402 | 0.2846 |
| Recall | 0.3678 | 1.0000 | 0.3829 |

These numbers describe one invented sensor process, not a real factory. Adjacent six-hour targets overlap, so row-level observations are correlated. No confidence interval or cross-asset generalization claim is made. The temperature baseline is an uncalibrated sensor-score proxy; its Brier score is not a calibrated-model comparison.

## Bring your own data

CSV schema: `asset,hour,temperature,vibration,pressure,load`, optionally `failure_event`. Hours must be contiguous integers per asset, duplicate asset/hour pairs are rejected, and sensor values must be finite. At least twelve historical sensor observations per asset are needed for scoring.

For training, supply `failure_event` as binary 0/1, use a common global hour index beginning at zero, and provide enough observations and positive/negative examples for chronological train/validation/test partitions. Adapt your sensor column names and units to this explicit schema before running:

```console
python -m industrial_risk train --input my-labeled-telemetry.csv --output artifacts/custom --horizon 6 --data-description "Dataset name, source, license and preprocessing version"
python -m industrial_risk score --model artifacts/custom/model.json --input my-unlabeled-telemetry.csv --output artifacts/custom/scores.csv
```

The train command uses the same causal feature and evaluation pipeline as the demo and does not relabel user data as synthetic. It cannot verify your data provenance or whether a recorded failure was observable at the stated hour. Document those facts and freeze a suitable evaluation protocol before interpreting results.

Model artifacts store feature order, means, scales, weights, intercept, horizon and the validation-selected threshold in JSON. No pickle deserialization or network calls are needed for inference. A batch contains its own trailing history; this version does not maintain a persistent online feature state.

See [architecture](docs/architecture.md), [data/model card](docs/data-model-card.md) and [interview guide](docs/interview-guide.md) for the actual causal contract, limitations and next experiments.

**Résumé en français :** prototype de risque de panne sur séries temporelles synthétiques, avec fenêtres causales, séparation chronologique purgée, normalisation sur le train, baselines, inférence CSV et dérive PSI. Les résultats ne sont pas des performances industrielles réelles.

MIT licensed.

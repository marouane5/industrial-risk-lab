from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

SENSORS = ("temperature", "vibration", "pressure", "load")
FEATURES = [f"{s}_{stat}" for s in SENSORS for stat in ("current", "mean12", "std12", "trend12")]


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -35, 35)))


def generate(seed: int = 42, assets: int = 8, hours: int = 1800) -> list[dict]:
    """Invented sensor/failure process; not a public industrial dataset."""
    if assets < 1 or hours < 100:
        raise ValueError("Use at least one asset and 100 hours")
    rng = np.random.default_rng(seed)
    rows = []
    for asset in range(assets):
        stress = float(rng.normal(0, 0.2))
        for hour in range(hours):
            drift = max(0, hour / hours - 0.8) * 1.2
            stress = 0.975 * stress + float(rng.normal(0, 0.14)) + drift * 0.03
            load = 0.55 + 0.22 * np.sin(hour / 36 + asset) + float(rng.normal(0, 0.05))
            event = int(rng.random() < float(sigmoid(-4.6 + 1.8 * stress + 0.7 * load)))
            rows.append({"asset": f"machine-{asset:02}", "hour": hour,
                         "temperature": 42 + 7 * stress + 12 * load + float(rng.normal(0, 1.2)),
                         "vibration": 2.2 + 0.85 * stress + 0.35 * load + float(rng.normal(0, 0.15)),
                         "pressure": 7.3 - 0.45 * stress + float(rng.normal(0, 0.12)),
                         "load": load, "failure_event": event})
    return rows


def write_csv(path: str | Path, rows: list[dict]):
    if not rows:
        raise ValueError("No rows to write")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for r in rows:
        r["hour"] = int(r["hour"])
        for s in SENSORS:
            r[s] = float(r[s])
        if "failure_event" in r:
            r["failure_event"] = int(r["failure_event"])
    return rows


def validate_rows(rows: list[dict]):
    seen = set()
    for row in rows:
        key = (row["asset"], row["hour"])
        if key in seen:
            raise ValueError("Duplicate asset/hour")
        seen.add(key)
        if any(not np.isfinite(row[s]) for s in SENSORS):
            raise ValueError("Sensors must be finite")
        if "failure_event" in row and row["failure_event"] not in (0, 1):
            raise ValueError("failure_event must be binary")


def make_features(rows: list[dict], horizon: int = 6, labeled: bool = True):
    """Features use [t-11,t]; label uses (t,t+horizon], with no centered windows."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    validate_rows(rows)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["asset"], []).append(row)
    x, y, metadata = [], [], []
    for asset in sorted(groups):
        group = sorted(groups[asset], key=lambda r: r["hour"])
        if any(b["hour"] - a["hour"] != 1 for a, b in zip(group, group[1:])):
            raise ValueError("Hourly data must be contiguous within each asset")
        end = len(group) - horizon if labeled else len(group)
        for i in range(11, end):
            feature = []
            for sensor in SENSORS:
                v = np.array([r[sensor] for r in group[i - 11:i + 1]], dtype=float)
                feature.extend([float(v[-1]), float(v.mean()), float(v.std()), float((v[-1] - v[0]) / 11)])
            x.append(feature)
            y.append(int(any(r["failure_event"] for r in group[i + 1:i + 1 + horizon])) if labeled else -1)
            metadata.append({"asset": asset, "hour": group[i]["hour"], "label_end_hour": group[i]["hour"] + horizon})
    if not x:
        raise ValueError("Insufficient rows after warm-up and label horizon")
    return np.asarray(x), np.asarray(y), metadata


def chronological_split(metadata: list[dict], total_hours: int, horizon: int = 6):
    cut_train, cut_val = int(total_hours * 0.6), int(total_hours * 0.8)
    hours = np.array([r["hour"] for r in metadata])
    ends = np.array([r["label_end_hour"] for r in metadata])
    # Purge labels crossing the next split; a future failure may not enter training.
    train = np.flatnonzero(ends < cut_train)
    val = np.flatnonzero((hours >= cut_train) & (ends < cut_val))
    test = np.flatnonzero(hours >= cut_val)
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Chronological split has an empty partition")
    if ends[train].max() >= hours[val].min() or ends[val].max() >= hours[test].min():
        raise ValueError("Label horizons cross a split boundary")
    return train, val, test


def fit_logistic(x, y, iterations: int = 1200, lr: float = 0.08, l2: float = 0.002) -> dict:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (x.ndim != 2 or x.shape[1] != len(FEATURES) or y.ndim != 1 or len(x) != len(y)
            or not np.isfinite(x).all() or not np.isfinite(y).all() or set(np.unique(y)) != {0.0, 1.0}):
        raise ValueError("Need finite 2D features and both binary classes")
    if iterations < 1 or lr <= 0 or l2 < 0:
        raise ValueError("Invalid optimizer parameters")
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-9] = 1
    z = (x - mean) / std
    w = np.zeros(x.shape[1])
    intercept = float(np.log(y.mean() / (1 - y.mean())))
    for _ in range(iterations):
        error = sigmoid(z @ w + intercept) - y
        w -= lr * (z.T @ error / len(y) + l2 * w)
        intercept -= lr * float(error.mean())
    return {"feature_names": FEATURES, "mean": mean.tolist(), "std": std.tolist(), "weights": w.tolist(),
            "intercept": intercept, "training_rows": len(y), "training_prevalence": float(y.mean()),
            "algorithm": "NumPy batch logistic regression", "l2": l2, "iterations": iterations}


def predict(model: dict, x):
    x = np.asarray(x, dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    mean, std = np.asarray(model["mean"], dtype=float), np.asarray(model["std"], dtype=float)
    if (x.ndim != 2 or x.shape[1] != len(FEATURES) or weights.shape != (len(FEATURES),)
            or mean.shape != weights.shape or std.shape != weights.shape or not np.isfinite(x).all()
            or not np.isfinite(weights).all() or not np.isfinite(mean).all() or not np.isfinite(std).all()
            or (std <= 0).any() or not np.isfinite(model["intercept"])):
        raise ValueError("Invalid feature shape or nonfinite sensor values")
    return sigmoid(((x - mean) / std) @ weights + model["intercept"])


def metrics(y, score, threshold: float) -> dict:
    y, score = np.asarray(y, dtype=float), np.asarray(score, dtype=float)
    if (y.ndim != 1 or score.ndim != 1 or len(y) != len(score) or not len(y)
            or not np.isfinite(y).all() or not set(np.unique(y)).issubset({0.0, 1.0})
            or not np.isfinite(score).all() or not np.isfinite(threshold)):
        raise ValueError("Scores and labels must be finite, nonempty and aligned")
    alarm = score >= threshold
    tp = int(((y == 1) & alarm).sum())
    fp = int(((y == 0) & alarm).sum())
    fn = int(((y == 1) & ~alarm).sum())
    tn = int(((y == 0) & ~alarm).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    # Group tied scores before constructing ranking metrics.
    order = np.argsort(-score, kind="stable")
    yy, ss = y[order], score[order]
    last = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]
    tps, fps = np.cumsum(yy)[last], np.cumsum(1 - yy)[last]
    positive, negative = int(y.sum()), len(y) - int(y.sum())
    recalls = tps / max(1, positive)
    precisions = tps / (tps + fps)
    ap = float(np.sum(np.diff(np.r_[0, recalls]) * precisions)) if positive else None
    tpr, fpr = np.r_[0, tps / max(1, positive)], np.r_[0, fps / max(1, negative)]
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)) if positive and negative else None
    return {"rows": len(y), "prevalence": float(y.mean()), "threshold": threshold, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall), "average_precision": ap, "roc_auc": auc,
            "brier_score": float(np.mean((score - y) ** 2)), "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}


def select_threshold(y, score) -> float:
    """Maximize validation F1, with a higher threshold as a deterministic tie break."""
    return max((metrics(y, score, float(t))["f1"], float(t)) for t in np.linspace(0.03, 0.75, 73))[1]


def fit_drift_reference(x, bins: int = 10) -> dict:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not len(x) or not np.isfinite(x).all() or bins < 2:
        raise ValueError("Drift reference requires nonempty finite features and at least two bins")
    edges, fractions = [], []
    for column in x.T:
        internal = np.unique(np.quantile(column, np.linspace(0, 1, bins + 1)[1:-1]))
        e = np.r_[-np.inf, internal, np.inf]
        counts, _ = np.histogram(column, bins=e)
        # Store only finite internal edges: JSON supports portable finite numbers.
        edges.append(internal.tolist())
        fractions.append((counts / len(column)).tolist())
    return {"feature_names": FEATURES, "internal_bin_edges": edges, "reference_fractions": fractions, "reference_rows": len(x)}


def drift_report(reference: dict, x) -> list[dict]:
    x = np.asarray(x, dtype=float)
    nfeatures = len(reference["feature_names"])
    if (x.ndim != 2 or x.shape[1] != nfeatures or not len(x) or not np.isfinite(x).all()
            or len(reference["internal_bin_edges"]) != nfeatures or len(reference["reference_fractions"]) != nfeatures):
        raise ValueError("Drift comparison requires finite observations")
    report = []
    for name, internal, expected, column in zip(reference["feature_names"], reference["internal_bin_edges"], reference["reference_fractions"], x.T):
        if len(expected) != len(internal) + 1:
            raise ValueError("Drift bin edge/fraction shape mismatch")
        counts, _ = np.histogram(column, bins=np.r_[-np.inf, internal, np.inf])
        p = np.maximum(np.asarray(expected), 1e-6)
        q = np.maximum(counts / len(column), 1e-6)
        p, q = p / p.sum(), q / q.sum()
        psi = float(np.sum((q - p) * np.log(q / p)))
        report.append({"feature": name, "psi": psi, "comparison_rows": len(column)})
    return sorted(report, key=lambda r: -r["psi"])


def run_experiment(rows: list[dict], output: str | Path, horizon: int = 6, data_kind: str = "user-supplied; provenance not verified", seed: int | None = None) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not rows or min(r["hour"] for r in rows) != 0:
        raise ValueError("Training hours must be nonnegative integers beginning at zero")
    hours = max(r["hour"] for r in rows) + 1
    x, y, metadata = make_features(rows, horizon)
    train, val, test = chronological_split(metadata, hours, horizon)
    model = fit_logistic(x[train], y[train])
    model["threshold"] = select_threshold(y[val], predict(model, x[val]))
    model["horizon_hours"] = horizon
    reference = fit_drift_reference(x[train])
    prior = float(y[train].mean())
    baseline_threshold = select_threshold(y[val], np.full(len(val), prior))
    # A simple train-only temperature cutoff baseline, tuned on validation.
    temp_mu, temp_sd = float(x[train, 1].mean()), float(x[train, 1].std())
    temp_val = sigmoid((x[val, 1] - temp_mu) / temp_sd)
    temp_test = sigmoid((x[test, 1] - temp_mu) / temp_sd)
    temp_threshold = select_threshold(y[val], temp_val)
    test_scores = predict(model, x[test])
    report = {"data_kind": data_kind, "seed": seed,
              "target": f"at least one failure event in next {horizon} hours", "feature_count": x.shape[1],
              "split_policy": "60/20/20 chronological; labels crossing boundaries purged; no random split",
              "splits": {name: {"rows": len(ids), "first_feature_hour": min(metadata[i]["hour"] for i in ids),
                                "last_feature_hour": max(metadata[i]["hour"] for i in ids),
                                "last_label_hour": max(metadata[i]["label_end_hour"] for i in ids),
                                "prevalence": float(y[ids].mean())} for name, ids in (("train", train), ("validation", val), ("test", test))},
              "threshold_selection": "validation F1 only; test never used for fitting or selection",
              "test": {"logistic_regression": metrics(y[test], test_scores, model["threshold"]),
                       "train_prevalence_baseline": metrics(y[test], np.full(len(test), prior), baseline_threshold),
                       "temperature_baseline": metrics(y[test], temp_test, temp_threshold)},
              "drift_test_vs_train": drift_report(reference, x[test])}
    for name, obj in (("model.json", model), ("drift_reference.json", reference), ("evaluation.json", report)):
        (output / name).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(output / "test_predictions.csv", [{"asset": metadata[i]["asset"], "hour": metadata[i]["hour"], "label": int(y[i]), "risk": float(p), "alarm": int(p >= model["threshold"])} for i, p in zip(test, test_scores)])
    return report


def run_demo(output: str | Path = "artifacts", seed: int = 42) -> dict:
    rows = generate(seed=seed)
    write_csv(Path(output) / "synthetic_sensors.csv", rows)
    return run_experiment(rows, output, data_kind="fully synthetic; no real industrial measurements", seed=seed)

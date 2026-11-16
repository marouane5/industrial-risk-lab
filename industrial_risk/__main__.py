import argparse
import json
from pathlib import Path

from .pipeline import FEATURES, make_features, predict, read_csv, run_demo, run_experiment, write_csv


def main():
    parser = argparse.ArgumentParser(description="Leakage-aware synthetic maintenance-risk lab")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--output", default="artifacts")
    demo.add_argument("--seed", type=int, default=42)
    train = sub.add_parser("train", help="Train/evaluate on a compatible, labeled sensor CSV")
    train.add_argument("--input", required=True)
    train.add_argument("--output", default="artifacts/custom")
    train.add_argument("--horizon", type=int, default=6)
    train.add_argument("--data-description", default="user-supplied; provenance not verified")
    score = sub.add_parser("score")
    score.add_argument("--model", default="artifacts/model.json")
    score.add_argument("--input", required=True, help="Contiguous hourly asset sensor CSV; labels optional")
    score.add_argument("--output", default="artifacts/batch_predictions.csv")
    args = parser.parse_args()
    if args.command in ("demo", "train"):
        report = run_demo(args.output, args.seed) if args.command == "demo" else run_experiment(read_csv(args.input), args.output, args.horizon, args.data_description)
        print(json.dumps({k: v for k, v in report.items() if k != "drift_test_vs_train"}, indent=2))
    else:
        model = json.loads(Path(args.model).read_text(encoding="utf-8"))
        if model["feature_names"] != FEATURES:
            raise ValueError("Model feature contract differs from this runtime")
        x, _, metadata = make_features(read_csv(args.input), labeled=False)
        scores = predict(model, x)
        rows = [{"asset": m["asset"], "hour": m["hour"], "risk": float(p), "alarm": int(p >= model["threshold"])} for m, p in zip(metadata, scores)]
        write_csv(args.output, rows)
        print(json.dumps({"scored_rows": len(rows), "output": args.output}))


if __name__ == "__main__":
    main()

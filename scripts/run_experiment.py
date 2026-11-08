#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adgb.experiment import evaluate_policies
from adgb.io import RESULT_FIELDS, read_csv, write_csv
from adgb.model import TinyMLP


def main():
    parser = argparse.ArgumentParser(description="Train the tiny neural selector and evaluate ADGB policies")
    parser.add_argument("--data", default="data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    train_path = os.path.join(args.data, "train.csv")
    test_path = os.path.join(args.data, "test.csv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        parser.error("dataset not found; run scripts/generate_dataset.py first")
    train = read_csv(train_path)
    test = read_csv(test_path)
    model = TinyMLP(hidden=args.hidden, seed=args.seed).fit(train, epochs=args.epochs)
    predictions = evaluate_policies(test, model)
    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "episode_metrics.csv"), predictions, RESULT_FIELDS)
    with open(os.path.join(args.out, "run_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "benchmark": "ADGB",
            "version": "0.1.0",
            "train_episodes": len(train),
            "test_episodes": len(test),
            "epochs": args.epochs,
            "hidden_units": args.hidden,
            "model_seed": args.seed,
            "test_oracle_label_accuracy": model.accuracy(test),
            "note": "Oracle accuracy is label agreement, not real-agent task success.",
        }, handle, indent=2)
        handle.write("\n")
    print("Training rows: %d | Test rows: %d" % (len(train), len(test)))
    print("MLP oracle-label accuracy: %.3f" % model.accuracy(test))
    print("Wrote %d policy-episode rows to %s" % (len(predictions), args.out))


if __name__ == "__main__":
    main()

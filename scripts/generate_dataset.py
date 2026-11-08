#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adgb.io import DATA_FIELDS, write_csv
from adgb.simulator import generate_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate reproducible ADGB synthetic train/test CSV files")
    parser.add_argument("--train-episodes", type=int, default=600, help="episodes per training seed")
    parser.add_argument("--test-episodes", type=int, default=300, help="episodes per test seed")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--test-seeds", type=int, nargs="+", default=[71, 101])
    parser.add_argument("--out", default="data")
    args = parser.parse_args()
    if set(args.train_seeds) & set(args.test_seeds):
        parser.error("training and test seeds must be disjoint")
    train = generate_dataset(args.train_seeds, args.train_episodes)
    test = generate_dataset(args.test_seeds, args.test_episodes)
    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "train.csv"), train, DATA_FIELDS)
    write_csv(os.path.join(args.out, "test.csv"), test, DATA_FIELDS)
    print("Wrote %d training and %d test episodes to %s" % (len(train), len(test), args.out))


if __name__ == "__main__":
    main()

"""Retrain ATS model script.

Usage:
    python backend/scripts/retrain_model.py [--csv path/to/data.csv]

CSV format (optional): columns f0..f16 and score column named `score`.
If no CSV is provided the script will train on built-in SEED data only.

This script calls `train_model(extra_X, extra_y)` from the trainer module and
persists the resulting model+scaler to disk (same locations used by the API).
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.model.trainer import train_model


def load_csv(path):
    import csv
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        return None, None
    # Expect columns f0..f16 and score
    X = []
    y = []
    for row in rows:
        try:
            x = [float(row.get(f'f{i}', 0.0)) for i in range(17)]
            s = float(row.get('score', 0.0))
            X.append(x)
            y.append(s)
        except Exception:
            continue
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', help='Optional CSV with training rows (f0..f16,score)')
    args = p.parse_args()

    extra_X = None
    extra_y = None
    if args.csv:
        if not os.path.exists(args.csv):
            print('CSV not found:', args.csv)
            return
        X, y = load_csv(args.csv)
        if X is None or len(X) == 0:
            print('No valid rows in CSV; aborting.')
            return
        extra_X, extra_y = X, y
        print(f'Loaded {len(X)} extra training rows from {args.csv}')

    print('Starting training...')
    m = train_model(extra_X=extra_X, extra_y=extra_y)
    print('Training complete. Model saved to backend/model/ats_model.pkl (or joblib equivalent).')


if __name__ == '__main__':
    main()

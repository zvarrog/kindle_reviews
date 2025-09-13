"""Простой мониторинг дрейфа числовых признаков (PSI).

Запуск:
  python -m scripts.drift_monitor --new-path data/processed/test.parquet
"""

from __future__ import annotations

from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
from scripts.settings import PROCESSED_DATA_DIR, MODEL_DIR
from scripts.train import NUMERIC_COLS


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 10 or len(actual) < 10:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, quantiles))
    if len(cuts) <= 2:
        return 0.0
    exp_counts = np.histogram(expected, bins=cuts)[0].astype(float)
    act_counts = np.histogram(actual, bins=cuts)[0].astype(float)
    exp_pct = (exp_counts + 1e-6) / (exp_counts.sum() + 1e-6 * len(exp_counts))
    act_pct = (act_counts + 1e-6) / (act_counts.sum() + 1e-6 * len(act_counts))
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--new-path", default=str(Path(PROCESSED_DATA_DIR) / "test.parquet")
    )
    ap.add_argument("--threshold", type=float, default=0.2)
    args = ap.parse_args()

    baseline_path = Path(MODEL_DIR) / "baseline_numeric_stats.json"
    if not baseline_path.exists():
        raise SystemExit("Нет baseline_numeric_stats.json — сначала запустите обучение")
    with open(baseline_path, "r", encoding="utf-8") as f:
        base_stats = json.load(f)

    new_df = pd.read_parquet(args.new_path)
    report = []
    for col in NUMERIC_COLS:
        if col not in new_df.columns or col not in base_stats:
            continue
        mean = base_stats[col]["mean"]
        std = base_stats[col]["std"] or 1e-6
        synth_expected = np.random.normal(mean, std, size=min(5000, len(new_df)))
        actual = new_df[col].dropna().values
        val = psi(synth_expected, actual)
        report.append({"feature": col, "psi": val, "drift": val > args.threshold})

    out_path = Path(MODEL_DIR) / "drift_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    bad = [r for r in report if r["drift"]]
    if bad:
        print("ДРИФТ обнаружен:", [r["feature"] for r in bad])
    else:
        print("Дрифт не обнаружен")


if __name__ == "__main__":  # pragma: no cover
    main()

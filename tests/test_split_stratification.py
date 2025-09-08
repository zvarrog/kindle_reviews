import pandas as pd
import numpy as np
from pathlib import Path

from scripts.train import load_splits, NUMERIC_COLS, PROCESSED_DATA_DIR


def test_stratified_ratios():
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
    # Проверяем что классы присутствуют во всех выборках
    for y in [y_train, y_val, y_test]:
        assert len(np.unique(y)) >= 2

    # Проверяем что доли классов не расходятся > 5 процентных пунктов между train и val
    def dist(y):
        v, c = np.unique(y, return_counts=True)
        total = c.sum()
        return {int(k): c[i] / total for i, k in enumerate(v)}

    d_train = dist(y_train)
    d_val = dist(y_val)
    for k in d_train:
        if k in d_val:
            assert abs(d_train[k] - d_val[k]) < 0.05

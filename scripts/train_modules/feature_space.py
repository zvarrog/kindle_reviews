"""
Модуль обработки признаков для обучения моделей.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging

log = logging.getLogger("feature_space")

NUMERIC_COLS = [
    "text_len",
    "word_count",
    "kindle_freq",
    "sentiment",
    "user_avg_len",
    "user_review_count",
    "item_avg_len",
    "item_review_count",
]

MEM_WARN_MB = 1024.0  # Можно централизовать через settings


class DenseTransformer(TransformerMixin, BaseEstimator):
    """
    Превращает scipy sparse в dense numpy (для HistGradientBoosting) + контроль памяти.
    """

    def fit(self, _X, _y=None):
        return self

    def transform(self, X):
        if hasattr(X, "toarray"):
            arr = X.toarray()
        else:
            arr = X
        size_mb = arr.nbytes / (1024 * 1024)
        if size_mb > MEM_WARN_MB:
            log.warning(
                "DenseTransformer: размер dense=%.1f MB > %.0f MB (риск памяти)",
                size_mb,
                MEM_WARN_MB,
            )
        return arr

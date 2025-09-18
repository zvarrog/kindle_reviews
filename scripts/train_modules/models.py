"""
Модуль моделей для обучения: классические, SimpleMLP, DistilBERT.
"""

import os
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scripts.settings import SEED
import logging

log = logging.getLogger("models")

# Torch (опционально)
try:
    import torch as _TORCH
except ImportError:
    _TORCH = None


def _select_device(preferred=None):
    if preferred in {"cpu", "cuda"}:
        return preferred
    if _TORCH is not None and getattr(_TORCH.cuda, "is_available", lambda: False)():
        return "cuda"
    return "cpu"


class SimpleMLP(BaseEstimator, ClassifierMixin):
    """
    Простейшая MLP (sklearn-совместимая) поверх плотных признаков.
    """

    def __init__(
        self, hidden_dim=128, epochs=5, lr=1e-3, batch_size=256, seed=SEED, device=None
    ):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self._fitted = False
        self._classes_ = None
        self._model = None
        self._device = device
        self._device_actual = None
        if _TORCH is None:
            log.warning(
                "SimpleMLP: torch не установлен — модель будет недоступна при fit()"
            )

    def fit(self, X, y):
        if _TORCH is None:
            raise ImportError("Для SimpleMLP требуется пакет torch. Установите torch.")
        from torch import nn

        _TORCH.Generator().manual_seed(self.seed)
        device_str = _select_device(self._device)
        # Логируем устройство только один раз за запуск
        if not hasattr(self, "_device_logged"):
            log.info(f"SimpleMLP: обучение на устройстве {device_str}")
            self._device_logged = True
        device = _TORCH.device(device_str)
        unique_labels = np.unique(y)
        self._classes_ = unique_labels
        label2idx = {lab: i for i, lab in enumerate(unique_labels)}
        y_idx = np.vectorize(label2idx.get)(y).astype(int)
        X_ = _TORCH.tensor(X.astype(np.float32), device=device)
        y_ = _TORCH.tensor(y_idx, device=device)
        in_dim = X_.shape[1]
        n_classes = len(unique_labels)
        model = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, n_classes),
        )
        model.to(device)
        opt = _TORCH.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        model.train()
        for _ in range(self.epochs):
            for i in range(0, len(X_), self.batch_size):
                xb = X_[i : i + self.batch_size]
                yb = y_[i : i + self.batch_size]
                opt.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
        self._model = model
        self._device_actual = device
        self._fitted = True
        return self

    def predict(self, X):
        self._ensure_fitted()
        if _TORCH is None:
            raise RuntimeError("torch недоступен во время predict")
        with _TORCH.no_grad():
            device = getattr(self, "_device_actual", _TORCH.device("cpu"))
            t = _TORCH.tensor(X.astype(np.float32), device=device)
            logits = self._model(t)
            pred_idx = logits.argmax(dim=1).cpu().numpy()
        return self._classes_[pred_idx]

    def _ensure_fitted(self):
        if not self._fitted:
            raise RuntimeError("SimpleMLP не обучена")


# DistilBERT вынесен в отдельный модуль scripts/models/distilbert.py
from scripts.models.distilbert import DistilBertClassifier
from scripts.models.kinds import ModelKind

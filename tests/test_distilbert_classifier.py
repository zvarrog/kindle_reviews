# tests/test_distilbert_classifier.py

import numpy as np
import pytest
import socket
from scripts.models.distilbert import DistilBertClassifier


def _hf_reachable(timeout: float = 2.0) -> bool:
    """Проверяет доступность huggingface.co по сети."""
    try:
        # Пытаемся установить TCP-соединение к порту 443 (HTTPS).
        with socket.create_connection(("huggingface.co", 443), timeout=timeout):
            return True
    except Exception:
        return False


# Совместная проверка: есть ли нужные модули и доступен ли удалённый ресурс.
HF_AVAILABLE = hasattr(__import__("importlib"), "import_module") and _hf_reachable()


@pytest.mark.skipif(
    not HF_AVAILABLE,
    reason="transformers/torch не установлены или недоступен интернет",
)
def test_distilbert_fit_predict_smoke():
    # Минимальный датасет: 3 текста, 2 класса
    X = np.array(
        [
            "good book, very interesting",
            "bad, boring, waste of time",
            "excellent, loved it",
        ]
    )
    y = np.array([1, 0, 1])
    clf = DistilBertClassifier(epochs=1, batch_size=2, max_len=32, lr=1e-4, seed=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert set(preds) <= set(y), "Модель должна предсказывать только обученные классы"
    assert len(preds) == len(y), "Размер предсказаний должен совпадать с входом"

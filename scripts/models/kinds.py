"""Перечисление поддерживаемых моделей.

Используется для централизованного контроля списка моделей в настройках и коде обучения.
"""

from enum import Enum


class ModelKind(str, Enum):
    """Типы моделей (значения — стабильные строковые ключи)."""

    logreg = "logreg"
    rf = "rf"
    hist_gb = "hist_gb"
    mlp = "mlp"
    distilbert = "distilbert"

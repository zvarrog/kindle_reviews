"""Feature contract для валидации входных данных и признаков модели."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json

# Константы из train.py для валидации
EXPECTED_NUMERIC_COLS = [
    "text_len",
    "word_count",
    "kindle_freq",
    "sentiment",
    "user_avg_len",
    "user_review_count",
    "item_avg_len",
    "item_review_count",
]

REQUIRED_TEXT_COL = "reviewText"


@dataclass
class FeatureContract:
    """Контракт признаков для валидации входных данных."""

    # Обязательные текстовые колонки
    required_text_columns: List[str]
    # Ожидаемые числовые колонки (могут быть заполнены дефолтами)
    expected_numeric_columns: List[str]
    # Базовые статистики для валидации числовых признаков
    baseline_stats: Optional[Dict[str, Dict[str, float]]] = None

    @classmethod
    def from_model_artifacts(cls, model_dir: Path) -> "FeatureContract":
        """Создаёт контракт из артефактов модели."""
        baseline_path = model_dir / "baseline_numeric_stats.json"
        baseline_stats = None

        if baseline_path.exists():
            with open(baseline_path, "r", encoding="utf-8") as f:
                baseline_stats = json.load(f)

        return cls(
            required_text_columns=[REQUIRED_TEXT_COL],
            expected_numeric_columns=EXPECTED_NUMERIC_COLS,
            baseline_stats=baseline_stats,
        )

    def validate_input_data(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Валидирует входные данные и возвращает предупреждения."""
        warnings = {}

        # Проверка обязательных текстовых колонок
        missing_text = []
        for col in self.required_text_columns:
            if col not in data:
                missing_text.append(col)
        if missing_text:
            warnings["missing_required_columns"] = missing_text

        # Проверка числовых колонок
        missing_numeric = []
        outlier_warnings = []

        for col in self.expected_numeric_columns:
            if col not in data:
                missing_numeric.append(col)
            elif self.baseline_stats and col in self.baseline_stats:
                # Проверяем на выбросы (простая проверка на 3 сигмы)
                baseline = self.baseline_stats[col]
                mean = baseline.get("mean", 0.0)
                std = baseline.get("std", 1.0)

                if isinstance(data[col], (list, tuple)):
                    values = data[col]
                else:
                    values = [data[col]]

                for i, val in enumerate(values):
                    if isinstance(val, (int, float)):
                        if std > 0 and abs(val - mean) > 3 * std:
                            outlier_warnings.append(
                                f"{col}[{i}]: {val} (expected ~{mean:.2f}±{3*std:.2f})"
                            )

        if missing_numeric:
            warnings["missing_numeric_columns"] = missing_numeric
        if outlier_warnings:
            warnings["potential_outliers"] = outlier_warnings

        return warnings

    def get_feature_info(self) -> Dict[str, Any]:
        """Возвращает информацию о контракте признаков."""
        info = {
            "required_text_columns": self.required_text_columns,
            "expected_numeric_columns": self.expected_numeric_columns,
            "total_features": len(self.required_text_columns)
            + len(self.expected_numeric_columns),
        }

        if self.baseline_stats:
            info["baseline_stats_available"] = True
            info["baseline_features"] = list(self.baseline_stats.keys())
        else:
            info["baseline_stats_available"] = False

        return info

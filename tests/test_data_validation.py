"""Тест валидации parquet данных."""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from scripts.data_validation import (
    validate_parquet_file,
    validate_parquet_dataset,
    KINDLE_REVIEWS_SCHEMA,
    DataSchema,
)


def test_valid_parquet_validation():
    """Тест валидации корректного parquet файла."""
    # Создаём валидные тестовые данные
    test_data = pd.DataFrame(
        {
            "reviewText": ["Great product!", "Terrible quality", "Average item"],
            "overall": [5, 1, 3],
            "text_len": [14.0, 16.0, 12.0],
            "word_count": [2.0, 2.0, 2.0],
            "kindle_freq": [0.0, 0.0, 0.0],
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test.parquet"
        test_data.to_parquet(tmp_path)

        result = validate_parquet_file(tmp_path, KINDLE_REVIEWS_SCHEMA)

        assert result.is_valid
        assert len(result.errors) == 0
        assert result.schema_info["rows"] == 3
        assert "reviewText" in result.schema_info["columns"]


def test_invalid_parquet_validation():
    """Тест валидации некорректного parquet файла."""
    # Создаём невалидные данные (отсутствует обязательная колонка)
    test_data = pd.DataFrame(
        {
            "reviewText": ["Great product!", "Terrible quality"],
            # Отсутствует "overall"
            "text_len": [14.0, 16.0],
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test.parquet"
        test_data.to_parquet(tmp_path)

        result = validate_parquet_file(tmp_path, KINDLE_REVIEWS_SCHEMA)

        assert not result.is_valid
        assert len(result.errors) > 0
        assert any("overall" in error for error in result.errors)


def test_data_quality_validation():
    """Тест валидации качества данных."""
    # Данные с проблемами качества
    test_data = pd.DataFrame(
        {
            "reviewText": ["Good", "", "Test"],  # Одна пустая строка
            "overall": [5, 6, 3],  # Значение 6 вне диапазона 1-5
            "text_len": [4.0, 0.0, 4.0],
            "sentiment_score": [0.5, 2.0, -0.3],  # Значение 2.0 вне диапазона [-1, 1]
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test.parquet"
        test_data.to_parquet(tmp_path)

        result = validate_parquet_file(tmp_path, KINDLE_REVIEWS_SCHEMA)

        # Должны быть предупреждения о качестве данных
        assert len(result.warnings) > 0
        warning_text = " ".join(result.warnings)
        assert "выше максимума" in warning_text  # Для overall и sentiment_score


def test_parquet_dataset_validation():
    """Тест валидации полного набора данных."""
    # Создаём тестовые данные для train/val/test
    base_data = {
        "reviewText": ["Great!", "Bad", "OK"],
        "overall": [5, 1, 3],
        "text_len": [6.0, 3.0, 2.0],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Создаём файлы train, val, test
        for split in ["train", "val", "test"]:
            df = pd.DataFrame(base_data)
            df.to_parquet(tmp_path / f"{split}.parquet")

        results = validate_parquet_dataset(tmp_path)

        # Проверяем что все три файла валидированы
        assert "train" in results
        assert "val" in results
        assert "test" in results

        # Проверяем консистентность
        assert "consistency" in results
        consistency_result = results["consistency"]
        # Не должно быть ошибок консистентности (одинаковые схемы)
        assert consistency_result.is_valid


def test_schema_consistency_check():
    """Тест проверки консистентности схем между файлами."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Train с одной схемой
        train_data = pd.DataFrame(
            {"reviewText": ["Great!"], "overall": [5], "text_len": [6.0]}
        )
        train_data.to_parquet(tmp_path / "train.parquet")

        # Val с дополнительной колонкой
        val_data = pd.DataFrame(
            {
                "reviewText": ["Bad"],
                "overall": [1],
                "text_len": [3.0],
                "extra_column": [99],  # Дополнительная колонка
            }
        )
        val_data.to_parquet(tmp_path / "val.parquet")

        results = validate_parquet_dataset(tmp_path)

        # Должно быть предупреждение о консистентности
        assert "consistency" in results
        consistency_result = results["consistency"]
        assert len(consistency_result.warnings) > 0
        warning_text = " ".join(consistency_result.warnings)
        assert "дополнительные колонки" in warning_text


def test_custom_schema_validation():
    """Тест валидации с кастомной схемой."""
    custom_schema = DataSchema(
        required_columns={"text", "label"},
        optional_columns={"score"},
        column_types={"text": "object", "label": "int64", "score": "float64"},
        numeric_ranges={"label": (0, 1), "score": (0.0, 1.0)},
        text_constraints={"text": {"min_length": 1, "allow_null": False}},
    )

    test_data = pd.DataFrame({"text": ["Hello world"], "label": [1], "score": [0.8]})

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test.parquet"
        test_data.to_parquet(tmp_path)

        result = validate_parquet_file(tmp_path, custom_schema)

        assert result.is_valid
        assert len(result.errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

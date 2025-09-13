"""Быстрый тест валидации на реальных данных."""

from scripts.data_validation import validate_parquet_dataset, log_validation_results
from pathlib import Path


def test_real_data():
    print("🔍 Тестирование валидации на реальных данных...")
    results = validate_parquet_dataset(Path("data/processed"))
    all_valid = log_validation_results(results)
    print(f"\n✅ Общий результат: {'Успешно' if all_valid else 'Есть проблемы'}")
    return all_valid


if __name__ == "__main__":
    test_real_data()

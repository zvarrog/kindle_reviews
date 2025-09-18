"""Валидация схемы и качества данных в parquet файлах."""

import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
import pandas as pd

# Используем централизованную систему логирования
from .logging_config import get_logger

log = get_logger(__name__)


@dataclass
class DataValidationResult:
    """Результат валидации данных."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    schema_info: Dict[str, Any]


@dataclass
class DataSchema:
    """Ожидаемая схема данных."""

    required_columns: Set[str]
    optional_columns: Set[str]
    column_types: Dict[str, str]  # колонка -> ожидаемый тип
    numeric_ranges: Dict[str, tuple]  # числовая колонка -> (min, max)
    text_constraints: Dict[str, Dict[str, Any]]  # текстовая колонка -> ограничения


# Схема для данных отзывов Kindle (обновлённая под реальные данные)
KINDLE_REVIEWS_SCHEMA = DataSchema(
    required_columns={"reviewText", "overall"},
    optional_columns={
        "text_len",
        "word_count",
        "kindle_freq",
        "sentiment_score",
        "has_punctuation",
        "avg_word_length",
        "upper_ratio",
        "digit_ratio",
        # Дополнительные колонки от Spark процессора
        "reviewTime",
        "unixReviewTime",
        "asin",
        "reviewerID",
        "sentiment",
        "words",
        "user_review_count",
        "item_review_count",
        "user_avg_len",
        "item_avg_len",
        "rawFeatures",
        "tfidfFeatures",  # Spark vector колонки
    },
    column_types={
        "reviewText": "object",
        "overall": ["int64", "int32"],  # Допускаем оба типа
        "text_len": ["float64", "int32", "int64"],  # Гибкие типы
        "word_count": ["float64", "int32", "int64"],
        "kindle_freq": ["float64", "int32"],
        "sentiment_score": ["float64", "int32"],
        "has_punctuation": ["float64", "int32"],
        "avg_word_length": ["float64", "int32"],
        "upper_ratio": ["float64", "int32"],
        "digit_ratio": ["float64", "int32"],
    },
    numeric_ranges={
        "overall": (1, 5),
        "text_len": (0, 50000),
        "word_count": (0, 10000),
        "kindle_freq": (0.0, 50.0),  # Увеличили максимум
        "sentiment_score": (-1.0, 1.0),
        "has_punctuation": (0.0, 1.0),
        "avg_word_length": (0.0, 50.0),
        "upper_ratio": (0.0, 1.0),
        "digit_ratio": (0.0, 1.0),
    },
    text_constraints={
        "reviewText": {"min_length": 1, "max_length": 100000, "allow_null": False}
    },
)


def validate_column_schema(
    df: pd.DataFrame, schema: DataSchema
) -> tuple[List[str], List[str]]:
    """Проверка схемы колонок."""
    errors = []
    warnings = []

    # Проверяем обязательные колонки
    missing_required = schema.required_columns - set(df.columns)
    if missing_required:
        errors.append(f"Отсутствуют обязательные колонки: {missing_required}")

    # Проверяем типы колонок
    for col, expected_types in schema.column_types.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            # Поддерживаем как один тип, так и список типов
            if isinstance(expected_types, list):
                if actual_type not in expected_types:
                    warnings.append(
                        f"Колонка '{col}': ожидался один из типов {expected_types}, найден {actual_type}"
                    )
            else:
                if actual_type != expected_types:
                    warnings.append(
                        f"Колонка '{col}': ожидался тип {expected_types}, найден {actual_type}"
                    )

    # Проверяем неожиданные колонки
    all_expected = schema.required_columns | schema.optional_columns
    unexpected = set(df.columns) - all_expected
    if unexpected:
        warnings.append(f"Неожиданные колонки: {unexpected}")

    return errors, warnings


def validate_data_quality(
    df: pd.DataFrame, schema: DataSchema
) -> tuple[List[str], List[str]]:
    """Проверка качества данных."""
    errors = []
    warnings = []

    # Проверяем размер датасета
    if len(df) == 0:
        errors.append("Датасет пуст")
        return errors, warnings

    # Проверяем пропуски в обязательных колонках
    for col in schema.required_columns:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                errors.append(f"Колонка '{col}': {null_count} пропущенных значений")

    # Проверяем диапазоны числовых значений
    for col, (min_val, max_val) in schema.numeric_ranges.items():
        if col in df.columns:
            numeric_data = pd.to_numeric(df[col], errors="coerce")
            if numeric_data.isnull().any():
                non_numeric_count = numeric_data.isnull().sum() - df[col].isnull().sum()
                if non_numeric_count > 0:
                    warnings.append(
                        f"Колонка '{col}': {non_numeric_count} не-числовых значений"
                    )

            valid_data = numeric_data.dropna()
            if len(valid_data) > 0:
                below_min = (valid_data < min_val).sum()
                above_max = (valid_data > max_val).sum()
                if below_min > 0:
                    warnings.append(
                        f"Колонка '{col}': {below_min} значений ниже минимума {min_val}"
                    )
                if above_max > 0:
                    warnings.append(
                        f"Колонка '{col}': {above_max} значений выше максимума {max_val}"
                    )

    # Проверяем текстовые ограничения
    for col, constraints in schema.text_constraints.items():
        if col in df.columns:
            text_data = df[col].astype(str)

            if not constraints.get("allow_null", True):
                null_or_empty = (
                    df[col].isnull() | (text_data == "") | (text_data == "nan")
                ).sum()
                if null_or_empty > 0:
                    errors.append(f"Колонка '{col}': {null_or_empty} пустых значений")

            min_len = constraints.get("min_length", 0)
            max_len = constraints.get("max_length", float("inf"))

            valid_texts = text_data[~df[col].isnull()]
            if len(valid_texts) > 0:
                lengths = valid_texts.str.len()
                too_short = (lengths < min_len).sum()
                too_long = (lengths > max_len).sum()

                if too_short > 0:
                    warnings.append(
                        f"Колонка '{col}': {too_short} текстов короче {min_len} символов"
                    )
                if too_long > 0:
                    warnings.append(
                        f"Колонка '{col}': {too_long} текстов длиннее {max_len} символов"
                    )

    return errors, warnings


def validate_data_consistency(
    dataframes: Dict[str, pd.DataFrame],
) -> tuple[List[str], List[str]]:
    """Проверка консистентности между разными наборами данных."""
    errors = []
    warnings = []

    if len(dataframes) < 2:
        return errors, warnings

    # Получаем схемы всех датафреймов
    schemas = {name: set(df.columns) for name, df in dataframes.items()}

    # Проверяем, что схемы совпадают
    reference_schema = next(iter(schemas.values()))
    reference_name = next(iter(schemas.keys()))

    for name, schema in schemas.items():
        if name == reference_name:
            continue

        missing_cols = reference_schema - schema
        extra_cols = schema - reference_schema

        if missing_cols:
            warnings.append(
                f"В '{name}' отсутствуют колонки из '{reference_name}': {missing_cols}"
            )
        if extra_cols:
            warnings.append(f"В '{name}' есть дополнительные колонки: {extra_cols}")

    # Проверяем распределения целевой переменной
    if all("overall" in df.columns for df in dataframes.values()):
        distributions = {}
        for name, df in dataframes.items():
            dist = df["overall"].value_counts(normalize=True).sort_index()
            distributions[name] = dist

        # Сравниваем распределения
        reference_dist = distributions[reference_name]
        for name, dist in distributions.items():
            if name == reference_name:
                continue

            # Проверяем значительные различия (>10% разницы)
            for rating in reference_dist.index:
                if rating in dist.index:
                    diff = abs(reference_dist[rating] - dist[rating])
                    if diff > 0.1:
                        warnings.append(
                            f"Распределение рейтинга {rating}: "
                            f"{reference_name}={reference_dist[rating]:.3f}, "
                            f"{name}={dist[rating]:.3f} (разница {diff:.3f})"
                        )

    return errors, warnings


def validate_parquet_file(
    file_path: Path,
    schema: DataSchema = KINDLE_REVIEWS_SCHEMA,
    check_duplicates: bool = True,
) -> DataValidationResult:
    """Валидация одного parquet файла."""
    errors = []
    warnings = []
    schema_info = {}

    try:
        # Читаем файл
        log.info("Валидация файла: %s", file_path)
        df = pd.read_parquet(file_path)

        # Безопасно обрабатываем dtypes (избегаем numpy arrays)
        safe_dtypes = {}
        for col, dtype in df.dtypes.items():
            try:
                safe_dtypes[col] = str(dtype)
            except Exception:
                safe_dtypes[col] = "complex_type"

        schema_info.update(
            {
                "file_path": str(file_path),
                "rows": len(df),
                "columns": list(df.columns),
                "dtypes": safe_dtypes,
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            }
        )

        # Проверка схемы
        schema_errors, schema_warnings = validate_column_schema(df, schema)
        errors.extend(schema_errors)
        warnings.extend(schema_warnings)

        # Проверка качества данных
        quality_errors, quality_warnings = validate_data_quality(df, schema)
        errors.extend(quality_errors)
        warnings.extend(quality_warnings)

        # Проверка дубликатов
        if check_duplicates and len(df) > 0:
            import numpy as np

            try:
                # Преобразуем все нехэшируемые типы в хэшируемые
                df_for_dupes = df.copy()
                for col in df_for_dupes.columns:
                    # Проверяем первые несколько значений на нехэшируемые типы
                    sample_vals = df_for_dupes[col].dropna().head(10)
                    if len(sample_vals) > 0:
                        has_unhashable = any(
                            isinstance(x, (np.ndarray, dict, list, set))
                            for x in sample_vals
                        )
                        if has_unhashable:
                            df_for_dupes[col] = df_for_dupes[col].apply(
                                lambda x: str(x) if x is not None else None
                            )
                duplicates = df_for_dupes.duplicated().sum()
                if duplicates > 0:
                    warnings.append(f"Найдено {duplicates} дублированных строк")
                    schema_info["duplicates"] = duplicates
            except Exception as e:
                warnings.append(f"Не удалось проверить дубликаты: {e}")
                schema_info["duplicates"] = "error_checking"

        # Дополнительная статистика
        if "overall" in df.columns:
            try:
                schema_info["rating_distribution"] = (
                    df["overall"].value_counts().to_dict()
                )
            except Exception:
                schema_info["rating_distribution"] = "error_calculating"

        if "reviewText" in df.columns:
            try:
                text_stats = {
                    "avg_text_length": df["reviewText"].astype(str).str.len().mean(),
                    "min_text_length": df["reviewText"].astype(str).str.len().min(),
                    "max_text_length": df["reviewText"].astype(str).str.len().max(),
                }
                schema_info["text_stats"] = text_stats
            except Exception:
                schema_info["text_stats"] = "error_calculating"

    except Exception as e:
        errors.append(f"Ошибка чтения файла: {e}")
        schema_info["error"] = str(e)

    is_valid = len(errors) == 0

    return DataValidationResult(
        is_valid=is_valid, errors=errors, warnings=warnings, schema_info=schema_info
    )


def validate_parquet_dataset(
    data_dir: Path,
    schema: DataSchema = KINDLE_REVIEWS_SCHEMA,
    check_consistency: bool = True,
) -> Dict[str, DataValidationResult]:
    """Валидация полного набора parquet файлов (train/val/test)."""
    log.info(f"Валидация dataset в директории: {data_dir}")

    results = {}
    dataframes = {}

    # Валидируем каждый файл
    for split_name in ["train", "val", "test"]:
        file_path = data_dir / f"{split_name}.parquet"

        if file_path.exists():
            result = validate_parquet_file(file_path, schema)
            results[split_name] = result

            # Сохраняем датафреймы для проверки консистентности
            if result.is_valid and check_consistency:
                try:
                    dataframes[split_name] = pd.read_parquet(file_path)
                except Exception as e:
                    log.warning(
                        f"Не удалось загрузить {split_name} для проверки консистентности: {e}"
                    )
        else:
            log.warning(f"Файл {file_path} не найден")
            results[split_name] = DataValidationResult(
                is_valid=False,
                errors=[f"Файл не найден: {file_path}"],
                warnings=[],
                schema_info={"file_path": str(file_path), "exists": False},
            )

    # Проверка консистентности между файлами
    if check_consistency and len(dataframes) > 1:
        consistency_errors, consistency_warnings = validate_data_consistency(dataframes)

        # Всегда добавляем результат консистентности
        consistency_result = DataValidationResult(
            is_valid=len(consistency_errors) == 0,
            errors=consistency_errors,
            warnings=consistency_warnings,
            schema_info={
                "type": "consistency_check",
                "files_compared": list(dataframes.keys()),
            },
        )
        results["consistency"] = consistency_result

    return results


def log_validation_results(results: Dict[str, DataValidationResult]) -> bool:
    """Логирование результатов валидации. Возвращает True если все валидации прошли."""
    all_valid = True

    for name, result in results.items():
        log.info(f"\nВалидация '{name}':")
        log.info(f"  Статус: {'Успешно' if result.is_valid else 'Ошибки'}")

        if result.errors:
            all_valid = False
            log.error(f"  Ошибки ({len(result.errors)}):")
            for error in result.errors:
                log.error(f"    - {error}")

        if result.warnings:
            log.warning(f"  Предупреждения ({len(result.warnings)}):")
            for warning in result.warnings:
                log.warning(f"    - {warning}")

        # Краткая статистика
        if "rows" in result.schema_info:
            log.info(f"  Строк: {result.schema_info['rows']:,}")
        if "columns" in result.schema_info:
            log.info(f"  Колонок: {len(result.schema_info['columns'])}")
        if "memory_usage_mb" in result.schema_info:
            log.info(f"  Память: {result.schema_info['memory_usage_mb']:.1f} MB")

    return all_valid

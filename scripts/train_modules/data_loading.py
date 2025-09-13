"""
Модуль загрузки и подготовки данных для обучения моделей.
"""

from pathlib import Path
from typing import Tuple, Dict, List
import pandas as pd
from .feature_space import NUMERIC_COLS
from scripts.data_validation import validate_parquet_dataset, log_validation_results
from scripts.settings import PROCESSED_DATA_DIR, SEED
import logging

log = logging.getLogger("data_loading")


def load_splits() -> (
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
):
    """
    Загружает сплиты; если отсутствуют val/test — формирует стратифицированно.
    """
    from sklearn.model_selection import train_test_split

    base = Path(PROCESSED_DATA_DIR)

    # Валидация всех parquet файлов
    log.info("🔍 Запуск валидации parquet данных...")
    validation_results = validate_parquet_dataset(base)
    all_valid = log_validation_results(validation_results)

    if not all_valid:
        log.warning("⚠️ Обнаружены проблемы с данными, но продолжаем выполнение...")
    else:
        log.info("✅ Валидация данных успешно завершена")

    train_p = base / "train.parquet"
    val_p = base / "val.parquet"
    test_p = base / "test.parquet"
    if not train_p.exists():
        raise FileNotFoundError(f"Нет train.parquet: {train_p}")
    df_train = pd.read_parquet(train_p)
    if not {"reviewText", "overall"}.issubset(df_train.columns):
        raise KeyError("Ожидаются колонки 'reviewText' и 'overall'")

    frames = {"train": df_train}
    if val_p.exists():
        frames["val"] = pd.read_parquet(val_p)
    if test_p.exists():
        frames["test"] = pd.read_parquet(test_p)

    if "val" not in frames or "test" not in frames:
        full = df_train.copy()
        if "val" in frames:
            full = pd.concat([full, frames["val"]], ignore_index=True)
        if "test" in frames:
            full = pd.concat([full, frames["test"]], ignore_index=True)
        full = full.dropna(subset=["reviewText", "overall"])
        y_full = full["overall"].astype(int)
        temp_X, X_test, temp_y, y_test = train_test_split(
            full, y_full, test_size=0.15, stratify=y_full, random_state=SEED
        )
        X_train, X_val, y_train, y_val = train_test_split(
            temp_X, temp_y, test_size=0.17647, stratify=temp_y, random_state=SEED
        )
    else:
        X_train = frames["train"]
        X_val = frames["val"]
        X_test = frames["test"]
        y_train = X_train["overall"].astype(int)
        y_val = X_val["overall"].astype(int)
        y_test = X_test["overall"].astype(int)

    def clean(df: pd.DataFrame):
        keep = ["reviewText", "overall"] + [c for c in NUMERIC_COLS if c in df.columns]
        return df[keep].dropna(subset=["reviewText", "overall"])

    X_train = clean(X_train)
    X_val = clean(X_val)
    X_test = clean(X_test)
    y_train = X_train["overall"].astype(int)
    y_val = X_val["overall"].astype(int)
    y_test = X_test["overall"].astype(int)

    return X_train, X_val, X_test, y_train, y_val, y_test

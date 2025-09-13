"""
Тест валидации processed parquet файлов через pandera.
"""

import pandera.pandas as pa
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")
FILES = ["train.parquet", "val.parquet", "test.parquet"]

# Схема для основных признаков (можно расширить)
schema = pa.DataFrameSchema(
    {
        # Текстовая колонка может быть object/строкой; оставляем без приведения
        "reviewText": pa.Column(pa.String, nullable=True, required=True),
        # Все числовые колонки приводим к float, чтобы принять int/float/числовые строки
        "text_len": pa.Column(pa.Float, nullable=True, coerce=True, required=False),
        "word_count": pa.Column(pa.Float, nullable=True, coerce=True, required=False),
        "kindle_freq": pa.Column(pa.Float, nullable=True, coerce=True, required=False),
        "sentiment": pa.Column(pa.Float, nullable=True, coerce=True, required=False),
        "user_avg_len": pa.Column(pa.Float, nullable=True, coerce=True, required=False),
        "user_review_count": pa.Column(
            pa.Float, nullable=True, coerce=True, required=False
        ),
        "item_avg_len": pa.Column(pa.Float, nullable=True, coerce=True, required=False),
        "item_review_count": pa.Column(
            pa.Float, nullable=True, coerce=True, required=False
        ),
    }
)


def test_parquet_files_valid():
    """
    Проверяет, что все parquet-файлы соответствуют схеме pandera.
    """
    for fname in FILES:
        fpath = DATA_DIR / fname
        assert fpath.exists(), f"Файл не найден: {fpath}"
        df = pd.read_parquet(fpath)
        # Валидация через pandera
        schema.validate(df)

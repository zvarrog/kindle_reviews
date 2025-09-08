"""Простой скрипт для скачивания и распаковки kindle_reviews.csv с удалением первого столбца, чтобы избежать предупреждений.

Модуль вызывает kaggle CLI через subprocess, скачивает zip-архив датасета
в папку RAW_DATA_DIR, извлекает файл CSV и удаляет zip.

Параметры задаются в `config.py` (FORCE_DOWNLOAD, KAGGLE_DATASET, CSV_NAME,
RAW_DATA_DIR, ZIP_FILENAME).
"""

import subprocess
from pathlib import Path
from zipfile import ZipFile
import logging
import pandas as pd
from config import (
    FORCE_DOWNLOAD,
    KAGGLE_DATASET,
    CSV_NAME,
    RAW_DATA_DIR,
    ZIP_FILENAME,
    CSV_PATH,
)


def remove_leading_index_column(csv_path: str = CSV_PATH) -> None:
    """
    Удаляет лишний первый столбец-индекс в CSV.

    Функция проверяет имя первого столбца и, если он похож на индекс, удаляет этот столбец
    и перезаписывает CSV без индекса.

    Args:
        csv_path (str): путь к CSV-файлу (по умолчанию используется CSV_PATH из config.py).
    """
    df = pd.read_csv(csv_path)
    first_col = str(df.columns[0])
    if first_col in ("", "_c0"):
        df.drop(df.columns[0], axis=1).to_csv(csv_path, index=False)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

if Path(CSV_PATH).exists() and not FORCE_DOWNLOAD:
    log.warning(
        "%s уже существует в %s, пропуск скачивания. Для форсированного скачивания установите флаг FORCE_DOWNLOAD = True в config.py.",
        CSV_NAME,
        RAW_DATA_DIR,
    )
else:
    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                KAGGLE_DATASET,
                "-p",
                RAW_DATA_DIR,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        log.error(
            "Ошибка при скачивании датасета через kaggle CLI: %s\nПроверьте наличие файла kaggle.json в директории ~/.kaggle.\n"
            "Если его нет, получите API токен на https://www.kaggle.com/settings и поместите файл в ~/.kaggle/kaggle.json.",
            e,
        )
        raise

    with ZipFile(ZIP_FILENAME, "r") as zip_ref:
        zip_ref.extract(CSV_NAME, RAW_DATA_DIR)

    Path(ZIP_FILENAME).unlink()

    # Удаление первого столбца (индекс)
    remove_leading_index_column()

    log.info("Абсолютный путь к CSV: %s", str(Path(CSV_PATH).resolve()))

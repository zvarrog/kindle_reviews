"""Простой скрипт для скачивания и распаковки kindle_reviews.csv.

Модуль вызывает kaggle CLI через subprocess, скачивает zip-архив датасета
в папку RAW_DATA_DIR, извлекает файл CSV и удаляет zip.

Параметры задаются в `config.py` (FORCE_DOWNLOAD, KAGGLE_DATASET, CSV_NAME,
RAW_DATA_DIR, ZIP_FILENAME).
"""

import subprocess
from pathlib import Path
from zipfile import ZipFile
import logging
from config import (
    FORCE_DOWNLOAD,
    KAGGLE_DATASET,
    CSV_NAME,
    RAW_DATA_DIR,
    ZIP_FILENAME,
    CSV_PATH,
)

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

    log.info("%s скачан и находится в %s", CSV_NAME, RAW_DATA_DIR)
    log.info("Абсолютный путь к CSV: %s", str(Path(CSV_PATH).resolve()))

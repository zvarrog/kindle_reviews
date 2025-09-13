"""
Тест для проверки логики пропуска обработки данных в контейнере.
Проверяет, правильно ли скрипт определяет существующие обработанные данные.
"""

import sys
import os
import logging
from pathlib import Path

# Добавляем путь к скриптам для импорта
sys.path.insert(0, "/opt/airflow/scripts")

try:
    from scripts.settings import (
        FORCE_PROCESS,
        RAW_DATA_DIR,
        CSV_NAME,
        PROCESSED_DATA_DIR,
    )

    CSV_PATH = str(RAW_DATA_DIR / CSV_NAME)
    print(f"✅ Успешно импортированы настройки:")
    print(f"   FORCE_PROCESS = {FORCE_PROCESS}")
    print(f"   CSV_PATH = {CSV_PATH}")
    print(f"   PROCESSED_DATA_DIR = {PROCESSED_DATA_DIR}")
except ImportError as e:
    print(f"❌ Ошибка импорта конфигурации: {e}")
    sys.exit(1)

# Используем централизованную систему логирования
from scripts.logging_config import setup_test_logging

log = setup_test_logging()


def test_file_logic():
    """Тестирует логику проверки существования файлов"""
    log.info("🧪 ТЕСТ ЛОГИКИ ПРОПУСКА ОБРАБОТКИ")
    log.info("=" * 50)

    # Определяем пути (как в spark_process.py)
    TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(PROCESSED_DATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "test.parquet")

    log.info(f"Проверяемые пути:")
    log.info(f"  TRAIN_PATH: {TRAIN_PATH}")
    log.info(f"  VAL_PATH: {VAL_PATH}")
    log.info(f"  TEST_PATH: {TEST_PATH}")

    # Проверяем существование каждого файла
    train_exists = Path(TRAIN_PATH).exists()
    val_exists = Path(VAL_PATH).exists()
    test_exists = Path(TEST_PATH).exists()

    log.info(f"Результаты проверки:")
    log.info(f"  📊 train.parquet: {'✅' if train_exists else '❌'} {TRAIN_PATH}")
    log.info(f"  📊 val.parquet: {'✅' if val_exists else '❌'} {VAL_PATH}")
    log.info(f"  📊 test.parquet: {'✅' if test_exists else '❌'} {TEST_PATH}")

    # Воспроизводим логику из spark_process.py
    should_skip = not FORCE_PROCESS and train_exists and val_exists and test_exists

    log.info(f"Анализ логики пропуска:")
    log.info(f"  FORCE_PROCESS = {FORCE_PROCESS}")
    log.info(f"  not FORCE_PROCESS = {not FORCE_PROCESS}")
    log.info(f"  Все файлы существуют = {train_exists and val_exists and test_exists}")
    log.info(
        f"  🎯 ДОЛЖНО ПРОПУСТИТЬ ОБРАБОТКУ: {'ДА ✅' if should_skip else 'НЕТ ❌'}"
    )

    # Проверяем размеры файлов для убедительности
    if train_exists:
        try:
            train_files = list(Path(TRAIN_PATH).iterdir())
            log.info(f"  📁 train.parquet содержит {len(train_files)} файлов")
        except Exception as e:
            log.error(f"  ❌ Ошибка чтения train.parquet: {e}")

    if val_exists:
        try:
            val_files = list(Path(VAL_PATH).iterdir())
            log.info(f"  📁 val.parquet содержит {len(val_files)} файлов")
        except Exception as e:
            log.error(f"  ❌ Ошибка чтения val.parquet: {e}")

    if test_exists:
        try:
            test_files = list(Path(TEST_PATH).iterdir())
            log.info(f"  📁 test.parquet содержит {len(test_files)} файлов")
        except Exception as e:
            log.error(f"  ❌ Ошибка чтения test.parquet: {e}")

    # Проверяем также исходный CSV
    csv_exists = Path(CSV_PATH).exists()
    log.info(f"Исходные данные:")
    log.info(f"  📄 CSV файл: {'✅' if csv_exists else '❌'} {CSV_PATH}")

    if csv_exists:
        try:
            csv_size = Path(CSV_PATH).stat().st_size / (1024 * 1024)
            log.info(f"      Размер: {csv_size:.2f} MB")
        except Exception as e:
            log.error(f"      Ошибка получения размера: {e}")

    log.info("=" * 50)
    log.info("🏁 ТЕСТ ЗАВЕРШЕН")

    return should_skip


if __name__ == "__main__":
    result = test_file_logic()
    if result:
        print("✅ РЕЗУЛЬТАТ: Обработка ДОЛЖНА быть пропущена")
    else:
        print("❌ РЕЗУЛЬТАТ: Обработка НЕ будет пропущена")

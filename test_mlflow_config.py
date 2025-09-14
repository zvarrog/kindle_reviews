#!/usr/bin/env python3
"""
Простой тест для проверки MLflow настроек в контейнере.
"""
import os
import sys
from pathlib import Path

# Добавляем scripts в путь
sys.path.insert(0, "/opt/airflow/scripts")

try:
    from scripts.train import _configure_mlflow_tracking
    import mlflow
    from scripts.logging_config import setup_training_logging

    # Настраиваем логирование
    setup_training_logging()

    # Логируем исходные переменные окружения
    print(
        f"MLFLOW_TRACKING_URI (env): {os.environ.get('MLFLOW_TRACKING_URI', 'НЕ ЗАДАНО')}"
    )
    print(f"Текущий рабочий каталог: {os.getcwd()}")
    print(f"Существует /opt/airflow/mlruns: {Path('/opt/airflow/mlruns').exists()}")

    # Вызываем нашу функцию настройки
    _configure_mlflow_tracking()

    # Проверяем результат
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # Попробуем создать тестовый run
    with mlflow.start_run(run_name="config_test") as run:
        print(f"Run ID: {run.info.run_id}")
        print(f"Artifact URI: {run.info.artifact_uri}")

        # Логируем тестовый параметр
        mlflow.log_param("test_param", "success")
        print("Тест успешно завершён!")

except Exception as e:
    print(f"ОШИБКА: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

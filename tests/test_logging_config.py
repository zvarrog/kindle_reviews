"""Тесты для централизованной системы логирования."""

import tempfile
import logging
import os
from pathlib import Path
from unittest.mock import patch
from scripts.logging_config import (
    setup_logging,
    setup_training_logging,
    setup_api_logging,
    setup_spark_logging,
    setup_test_logging,
    get_logger,
    get_logging_config,
    setup_for_environment,
    auto_detect_environment,
    setup_auto_logging,
)


def test_basic_logging_setup():
    """Тест базовой настройки логирования."""
    logger = setup_logging(level="DEBUG", component_name="TEST")

    assert logger.name == "TEST"
    assert logger.level == logging.DEBUG

    # Проверяем что логирование работает
    logger.info("Test message")
    logger.debug("Debug message")


def test_file_logging():
    """Тест логирования в файл."""
    import tempfile
    import os

    # Создаем временную директорию
    tmp_dir = tempfile.mkdtemp()
    try:
        log_file = os.path.join(tmp_dir, "test.log")

        # Настраиваем логирование в файл
        logger = setup_logging(
            level="INFO",
            log_file=log_file,
            component_name="FILE_TEST",
            force_setup=True,
        )

        # Тестируем логирование
        logger.info("Test file logging")
        logger.warning("Warning message")

        # Принудительно сбрасываем буферы
        for handler in logger.handlers:
            if hasattr(handler, "flush"):
                handler.flush()

        # Проверяем что файл создан и содержит логи
        assert os.path.exists(log_file)

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Test file logging" in content
            assert "Warning message" in content
            assert "FILE_TEST" in content

    finally:
        # Очищаем handlers перед удалением файлов
        for handler in logging.getLogger("FILE_TEST").handlers[:]:
            if hasattr(handler, "close"):
                handler.close()
            logging.getLogger("FILE_TEST").removeHandler(handler)

        # Удаляем временные файлы
        try:
            if os.path.exists(log_file):
                os.unlink(log_file)
            os.rmdir(tmp_dir)
        except (PermissionError, OSError):
            pass  # Игнорируем ошибки удаления на Windows


def test_component_specific_logging():
    """Тест специфичных для компонентов настроек."""
    # Тестируем разные компоненты
    training_logger = setup_training_logging()
    api_logger = setup_api_logging()
    spark_logger = setup_spark_logging()
    test_logger = setup_test_logging()

    assert "TRAINING" in training_logger.name or training_logger.name == "TRAINING"
    assert "API" in api_logger.name or api_logger.name == "API"
    assert "SPARK" in spark_logger.name or spark_logger.name == "SPARK"
    assert "TEST" in test_logger.name or test_logger.name == "TEST"


def test_get_logger_auto_setup():
    """Тест автоматической настройки логгера."""
    logger = get_logger("auto_test")

    # Должен работать даже без явной настройки
    logger.info("Auto setup test")
    assert logger.name == "auto_test"


def test_logging_config_generation():
    """Тест генерации конфигурации логирования."""
    config = get_logging_config(
        level="WARNING", log_file="test.log", component_name="CONFIG_TEST"
    )

    assert config["version"] == 1
    assert "formatters" in config
    assert "handlers" in config
    assert "loggers" in config
    assert "file" in config["handlers"]
    assert "console" in config["handlers"]


def test_external_libraries_suppression():
    """Тест подавления логов внешних библиотек."""
    config = get_logging_config(level="INFO")

    # Проверяем что внешние библиотеки настроены на WARNING
    assert config["loggers"]["urllib3"]["level"] == "WARNING"
    assert config["loggers"]["pyspark"]["level"] == "WARNING"
    assert config["loggers"]["py4j"]["level"] == "WARNING"


def test_format_consistency():
    """Тест консистентности форматов."""
    # Разные компоненты должны использовать совместимые форматы
    configs = [
        get_logging_config(component_name="TRAINING"),
        get_logging_config(component_name="API"),
        get_logging_config(component_name="SPARK"),
    ]

    for config in configs:
        # Все должны иметь одинаковую структуру
        assert "formatters" in config
        assert "default" in config["formatters"]
        assert "detailed" in config["formatters"]

        # Форматы должны содержать основные элементы
        default_format = config["formatters"]["default"]["format"]
        assert "%(levelname)s" in default_format
        assert "%(message)s" in default_format


def test_environment_detection():
    """Тест автоматического определения среды."""
    # Тестируем разные сценарии через mock переменных окружения

    # Development (по умолчанию)
    with patch.dict(os.environ, {}, clear=True):
        assert auto_detect_environment() == "development"

    # Airflow
    with patch.dict(os.environ, {"AIRFLOW_HOME": "/opt/airflow"}, clear=True):
        assert auto_detect_environment() == "airflow"

    # Docker
    with patch.dict(os.environ, {"DOCKER_CONTAINER": "true"}, clear=True):
        assert auto_detect_environment() == "docker"

    # Production
    with patch.dict(os.environ, {"PRODUCTION": "true"}, clear=True):
        assert auto_detect_environment() == "production"


def test_environment_specific_logging():
    """Тест настройки логирования для разных сред."""
    # Development
    dev_logger = setup_for_environment("development")
    assert dev_logger.level == logging.DEBUG

    # Production
    prod_logger = setup_for_environment("production")
    assert prod_logger.level == logging.INFO

    # Docker
    docker_logger = setup_for_environment("docker")
    assert docker_logger.level == logging.INFO

    # Airflow
    airflow_logger = setup_for_environment("airflow")
    assert airflow_logger.level == logging.INFO


def test_auto_logging_setup():
    """Тест автоматической настройки логирования."""
    # Мокаем определение среды как Docker
    with patch("scripts.logging_config.auto_detect_environment", return_value="docker"):
        logger = setup_auto_logging()
        assert logger.level == logging.INFO
        assert "DOCKER" in logger.name


def test_docker_optimized_logging():
    """Тест оптимизации логирования для Docker."""
    # В Docker не должно быть timestamp (Docker добавляет свои)
    docker_logger = setup_for_environment("docker")

    # Проверяем что логирование работает без ошибок
    docker_logger.info("Test Docker logging")
    docker_logger.warning("Test warning in Docker")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])

"""Централизованная конфигурация логирования для всех компонентов проекта."""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import os
import json
import contextvars
from contextlib import contextmanager

# Контекстный trace_id для корреляции событий
_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "trace_id", default=None
)


class TraceIdFilter(logging.Filter):
    """Фильтр, добавляющий trace_id в запись лога."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        record.trace_id = _trace_id_var.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """Простой JSON-форматтер без внешних зависимостей."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": getattr(record, "trace_id", None),
            "module": record.module,
            "lineno": record.lineno,
        }
        # Добавляем исключение если есть
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def set_trace_id(value: Optional[str]) -> None:
    """Установить текущий trace_id в контексте."""

    _trace_id_var.set(value)


def get_trace_id() -> Optional[str]:
    """Получить текущий trace_id из контекста."""

    return _trace_id_var.get()


def clear_trace_id() -> None:
    """Сбросить trace_id в контексте."""

    _trace_id_var.set(None)


@contextmanager
def trace_context(trace_id: str):
    """Контекст-менеджер для установки trace_id на время блока."""

    token = _trace_id_var.set(trace_id)
    try:
        yield
    finally:
        _trace_id_var.reset(token)


def get_logging_config(
    level: str = "INFO",
    log_file: Optional[str] = None,
    include_timestamp: bool = True,
    component_name: Optional[str] = None,
    log_format: str = "text",  # text|json
) -> Dict[str, Any]:
    """
    Создает конфигурацию логирования для компонента.

    Args:
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
        log_file: Путь к файлу логов (опционально)
        include_timestamp: Включать ли временные метки
        component_name: Имя компонента для форматирования

    Returns:
        Словарь конфигурации для logging.config.dictConfig
    """
    # Базовый формат
    if include_timestamp:
        base_format = "%(asctime)s [%(levelname)s]"
    else:
        base_format = "[%(levelname)s]"

    # Добавляем компонент если указан
    if component_name:
        base_format += f" {component_name}"

    # Добавляем имя модуля и сообщение
    base_format += " %(name)s: %(message)s"

    # Выбор форматтера
    formatter_name = "default" if log_format != "json" else "json"

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": base_format, "datefmt": "%Y-%m-%d %H:%M:%S"},
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {
                "()": "scripts.logging_config.JsonFormatter",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
        },
        "filters": {
            "trace": {"()": "scripts.logging_config.TraceIdFilter"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": formatter_name,
                "stream": "ext://sys.stdout",
                "filters": ["trace"],
            }
        },
        "root": {"level": level, "handlers": ["console"]},
        "loggers": {
            # Наши компоненты
            "kindle": {"level": level, "handlers": ["console"], "propagate": False},
            "scripts": {"level": level, "handlers": ["console"], "propagate": False},
            # Подавляем избыточные логи внешних библиотек
            "urllib3": {"level": "WARNING"},
            "pyspark": {"level": "WARNING"},
            "py4j": {"level": "WARNING"},
            "mlflow": {"level": "ERROR"},  # Убираем INFO/WARNING от MLflow
            "optuna": {"level": "ERROR"},  # Убираем INFO/WARNING от Optuna
            "git": {"level": "ERROR"},     # Убираем предупреждения git
        },
    }

    # Добавляем файловый хендлер если нужен
    if log_file:
        # Создаем директорию для логов если нужно
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "formatter": formatter_name if log_format == "json" else "detailed",
            "filename": str(log_path),
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
            "filters": ["trace"],
        }

        # Добавляем файловый хендлер к root и основным логгерам
        config["root"]["handlers"].append("file")
        config["loggers"]["kindle"]["handlers"].append("file")
        config["loggers"]["scripts"]["handlers"].append("file")

    return config


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    component_name: Optional[str] = None,
    force_setup: bool = False,
    include_timestamp: bool = True,
    log_format: str = "text",
) -> logging.Logger:
    """
    Настраивает логирование для компонента проекта.

    Args:
        level: Уровень логирования
        log_file: Путь к файлу логов
        component_name: Имя компонента
        force_setup: Принудительная перенастройка
        include_timestamp: Включать ли временные метки

    Returns:
        Настроенный логгер
    """
    # Получаем конфигурацию
    config = get_logging_config(
        level=level,
        log_file=log_file,
        component_name=component_name,
        include_timestamp=include_timestamp,
        log_format=log_format,
    )

    # Применяем конфигурацию только если нужно
    root_logger = logging.getLogger()
    if not root_logger.handlers or force_setup:
        # Очищаем существующие хендлеры при принудительной настройке
        if force_setup:
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

        logging.config.dictConfig(config)

    # Возвращаем логгер для компонента
    logger_name = component_name or "kindle"
    logger = logging.getLogger(logger_name)

    # Принудительно устанавливаем уровень логгера
    numeric_level = getattr(logging, level.upper())
    logger.setLevel(numeric_level)

    # Также устанавливаем уровень для хендлеров
    for handler in logger.handlers:
        handler.setLevel(numeric_level)

    logger.info(
        "Логирование настроено для компонента: %s, уровень: %s, формат: %s",
        logger_name,
        level,
        log_format,
    )
    return logger


def setup_training_logging(log_dir: str = "logs") -> logging.Logger:
    """Настройка логирования для обучения модели."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/training_{timestamp}.log"
    return setup_logging(level="INFO", log_file=log_file, component_name="TRAINING")


def setup_api_logging(log_dir: str = "logs") -> logging.Logger:
    """Настройка логирования для API сервиса."""
    log_file = f"{log_dir}/api_service.log"
    return setup_logging(level="INFO", log_file=log_file, component_name="API")


def setup_spark_logging(log_dir: str = "logs") -> logging.Logger:
    """Настройка логирования для Spark процессора."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/spark_processing_{timestamp}.log"
    return setup_logging(level="INFO", log_file=log_file, component_name="SPARK")


def setup_test_logging() -> logging.Logger:
    """Настройка логирования для тестов."""
    return setup_logging(level="INFO", component_name="TEST")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Получить логгер с автоматической настройкой если необходимо.

    Args:
        name: Имя логгера (если None, используется имя модуля)

    Returns:
        Настроенный логгер
    """
    # Если логирование не настроено, настраиваем с базовыми параметрами
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        setup_logging()

    return logging.getLogger(name or "kindle")


# Предустановленные конфигурации для разных сред
DEVELOPMENT_CONFIG = {
    "level": "DEBUG",
    "include_timestamp": True,
    "log_file": None,  # Только консоль
}

PRODUCTION_CONFIG = {
    "level": "INFO",
    "include_timestamp": True,
    "log_file": "logs/production.log",
}

DOCKER_CONFIG = {
    "level": "INFO",
    "include_timestamp": False,  # Docker добавляет свои timestamp
    "log_file": None,  # В Docker логи идут в stdout/stderr
}

AIRFLOW_CONFIG = {
    "level": "INFO",
    "include_timestamp": False,  # Airflow управляет timestamp
    "log_file": None,  # Airflow перенаправляет логи
}


def setup_for_environment(env: str = "development") -> logging.Logger:
    """
    Настройка логирования для конкретной среды.

    Args:
        env: Название среды (development, production, docker, airflow)

    Returns:
        Настроенный логгер
    """
    configs = {
        "development": DEVELOPMENT_CONFIG,
        "production": PRODUCTION_CONFIG,
        "docker": DOCKER_CONFIG,
        "airflow": AIRFLOW_CONFIG,
    }

    config = configs.get(env, DEVELOPMENT_CONFIG)

    # Переопределения через переменные окружения
    env_level = os.getenv("LOG_LEVEL") or config["level"]
    env_format = (os.getenv("LOG_FORMAT") or "text").lower()  # text|json
    include_ts = (
        os.getenv("LOG_INCLUDE_TIMESTAMP") or str(config["include_timestamp"]).lower()
    )
    include_ts_bool = include_ts in ("1", "true", "yes", "y")

    return setup_logging(
        level=env_level,
        log_file=config.get("log_file"),
        component_name=f"KINDLE-{env.upper()}",
        include_timestamp=include_ts_bool,
        log_format=env_format,
    )


def auto_detect_environment() -> str:
    """
    Автоматическое определение среды выполнения.

    Returns:
        Название среды
    """
    import os

    # Проверяем переменные окружения
    if os.getenv("AIRFLOW_HOME"):
        return "airflow"
    elif os.getenv("DOCKER_CONTAINER") or os.path.exists("/.dockerenv"):
        return "docker"
    elif os.getenv("PROD") or os.getenv("PRODUCTION"):
        return "production"
    else:
        return "development"


def setup_auto_logging() -> logging.Logger:
    """
    Автоматическая настройка логирования на основе среды.

    Returns:
        Настроенный логгер
    """
    env = auto_detect_environment()
    return setup_for_environment(env)


DOCKER_CONFIG = {
    "level": "INFO",
    "include_timestamp": True,
    "log_file": None,  # В Docker обычно логи идут в stdout
}

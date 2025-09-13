from pathlib import Path
import logging
import os
from types import SimpleNamespace
from scripts.models.kinds import ModelKind

# Опциональная загрузка .env (безопасно: если python-dotenv не установлен, пропускаем)
try:  # noqa: WPS501
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    # Нет python-dotenv — просто пропускаем загрузку из .env
    pass

# Директории (Path для удобства работы с путями)
RAW_DATA_DIR: Path = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
PROCESSED_DATA_DIR: Path = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
MODEL_DIR: Path = Path(os.getenv("MODEL_DIR", "model"))

# Данные
KAGGLE_DATASET: str = os.getenv("KAGGLE_DATASET", "bharadwaj6/kindle-reviews")
CSV_NAME: str = os.getenv("CSV_NAME", "kindle_reviews.csv")
JSON_NAME: str = os.getenv("JSON_NAME", "kindle_reviews.json")

# Флаги
FORCE_DOWNLOAD: bool = os.getenv("FORCE_DOWNLOAD", "false").lower() in {
    "1",
    "true",
    "yes",
}
FORCE_PROCESS: bool = os.getenv("FORCE_PROCESS", "false").lower() in {
    "1",
    "true",
    "yes",
}
FORCE_TRAIN: bool = os.getenv("FORCE_TRAIN", "false").lower() in {"1", "true", "yes"}
SEED: int = int(os.getenv("SEED", "42"))

# Параметры обработки
PER_CLASS_LIMIT: int = int(os.getenv("PER_CLASS_LIMIT", "20000"))
HASHING_TF_FEATURES: int = int(os.getenv("HASHING_TF_FEATURES", "4096"))
SHUFFLE_PARTITIONS: int = int(os.getenv("SHUFFLE_PARTITIONS", "64"))
MIN_DF: float | int = int(os.getenv("MIN_DF", "2"))
MIN_TF: int = int(os.getenv("MIN_TF", "1"))

# Параметры обучения
# Явный список моделей через Enum (минимизируем опечатки и упрощаем валидацию)
SELECTED_MODEL_KINDS: list[ModelKind] = [
    ModelKind.logreg,
    ModelKind.rf,
    ModelKind.hist_gb,
    ModelKind.mlp,
]
OPTUNA_N_TRIALS: int = int(os.getenv("OPTUNA_N_TRIALS", "40"))
OPTUNA_STORAGE: str = os.getenv("OPTUNA_STORAGE", "sqlite:///optuna_study.db")
STUDY_BASE_NAME: str = os.getenv("STUDY_BASE_NAME", "kindle_optuna")

# Доп. параметры для обратной совместимости и ускорения тестов
N_FOLDS: int = int(os.getenv("N_FOLDS", "1"))  # k-fold валидация, по умолчанию 1
TRAIN_DEVICE: str = os.getenv("TRAIN_DEVICE", "cuda")  # устройство для NN
EARLY_STOP_PATIENCE: int = int(os.getenv("EARLY_STOP_PATIENCE", "10"))
OPTUNA_TIMEOUT_SEC: int = int(os.getenv("OPTUNA_TIMEOUT_SEC", "300"))

# Используем централизованную систему логирования
from .logging_config import get_logger

log = get_logger("kindle")

# ---- Объект settings для обратной совместимости (scripts.config ожидает его) ----
# Преобразуем Path в строки и перечислим модели в строковом формате через запятую
_selected_models_csv = ",".join(m.value for m in SELECTED_MODEL_KINDS)

settings = SimpleNamespace(
    # флаги
    force_download=FORCE_DOWNLOAD,
    force_process=FORCE_PROCESS,
    force_train=FORCE_TRAIN,
    # датасет и имена файлов
    kaggle_dataset=KAGGLE_DATASET,
    csv_name=CSV_NAME,
    json_name=JSON_NAME,
    # директории (как строки)
    model_dir=str(MODEL_DIR),
    raw_data_dir=str(RAW_DATA_DIR),
    processed_data_dir=str(PROCESSED_DATA_DIR),
    # параметры обработки
    per_class_limit=PER_CLASS_LIMIT,
    hashing_tf_features=HASHING_TF_FEATURES,
    shuffle_partitions=SHUFFLE_PARTITIONS,
    min_df=MIN_DF,
    min_tf=MIN_TF,
    # обучение / optuna
    selected_models=_selected_models_csv,
    optuna_n_trials=OPTUNA_N_TRIALS,
    optuna_storage=OPTUNA_STORAGE,
)

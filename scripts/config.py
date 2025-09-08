"""Конфигурационные параметры для скачивания и обработки датасета Kindle Reviews."""

# Если True — принудительно скачивать или обрабатывать данные
FORCE_DOWNLOAD = True
FORCE_PROCESS = True
FORCE_TRAIN = True


# Параметры датасета и файлы внутри архива
KAGGLE_DATASET = "bharadwaj6/kindle-reviews"
CSV_NAME = "kindle_reviews.csv"
JSON_NAME = "kindle_reviews.json"
MODEL_DIR = "model"


# Папка назначения (используем прямой стиль, скрипт будет работать с pathlib)
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
ZIP_FILENAME = RAW_DATA_DIR + "/kindle-reviews.zip"
CSV_PATH = RAW_DATA_DIR + "/" + CSV_NAME

# Параметры обработки
# Максимальное число отзывов на каждый класс звёзд (1-5)
PER_CLASS_LIMIT = 20_000
HASHING_TF_FEATURES = 4096
SHUFFLE_PARTITIONS = 64

# Пороговые параметры векторизации (для CountVectorizer)
# Если MIN_DF >= 1 — трактуется как абсолютное число документов; если 0 < MIN_DF < 1 — как доля документов
MIN_DF = 2
MIN_TF = 1

# Параметры для обучения модели
# Если модель уже есть в папке MODEL_DIR и FORCE_TRAIN=False, тренировка пропускается
SELECTED_MODELS = [
    "logreg",
    "rf",
    "hist_gb",
    "mlp",
    "distilbert",
]  # можно добавить "mlp", "distilbert" при необходимости
OPTUNA_N_TRIALS = 40  # число trial'ов по умолчанию

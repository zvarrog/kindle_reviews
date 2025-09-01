"""Конфигурационные параметры для скачивания и обработки датасета Kindle Reviews."""

# Если True — принудительно скачивать или обрабатывать данные
FORCE_DOWNLOAD = False
FORCE_PROCESS = False

# Параметры датасета и файлы внутри архива
KAGGLE_DATASET = "bharadwaj6/kindle-reviews"
CSV_NAME = "kindle_reviews.csv"
JSON_NAME = "kindle_reviews.json"

# Папка назначения (используем прямой стиль, скрипт будет работать с pathlib)
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
ZIP_FILENAME = RAW_DATA_DIR + "/kindle-reviews.zip"
CSV_PATH = RAW_DATA_DIR + "/" + CSV_NAME

# Параметры обработки
# Максимальное число отзывов на каждый класс звёзд (1-5)
PER_CLASS_LIMIT = 10_000
HASHING_TF_FEATURES = 4096
SHUFFLE_PARTITIONS = 64

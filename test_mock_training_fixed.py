"""
Мок-тест для быстрого тестирования pipeline обучения с легкими параметрами.
Создает небольшой датасет и обучает простую модель для проверки работоспособности.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import os


def create_mock_dataset(n_samples=1000, n_classes=3):
    """Создает мок-датасет для тестирования"""
    np.random.seed(42)

    # Создаем случайные отзывы с разными оценками
    reviews = []
    ratings = []

    positive_words = [
        "great",
        "excellent",
        "amazing",
        "wonderful",
        "fantastic",
        "love",
        "perfect",
        "best",
    ]
    negative_words = [
        "terrible",
        "awful",
        "hate",
        "worst",
        "bad",
        "horrible",
        "disappointing",
    ]
    neutral_words = ["okay", "average", "decent", "normal", "fine", "acceptable"]

    word_sets = [negative_words, neutral_words, positive_words]

    for i in range(n_samples):
        rating = np.random.randint(1, n_classes + 1)
        word_set = word_sets[rating - 1]

        # Создаем текст отзыва
        n_words = np.random.randint(5, 20)
        review_words = np.random.choice(
            word_set + ["book", "kindle", "device", "reading"], n_words
        )
        review_text = " ".join(review_words)

        reviews.append(review_text)
        ratings.append(rating)

    # Создаем DataFrame
    df = pd.DataFrame(
        {
            "reviewText": reviews,
            "overall": ratings,
            "text_len": [len(text) for text in reviews],
            "word_count": [len(text.split()) for text in reviews],
            "kindle_freq": np.random.random(n_samples),
            "sentiment": np.random.uniform(-1, 1, n_samples),
            "user_avg_len": np.random.uniform(50, 200, n_samples),
            "user_review_count": np.random.randint(1, 50, n_samples),
            "item_avg_len": np.random.uniform(50, 200, n_samples),
            "item_review_count": np.random.randint(1, 100, n_samples),
        }
    )

    return df


def create_mock_parquet_files(base_dir: Path, n_samples=1000):
    """Создает мок parquet файлы для тестирования"""
    base_dir.mkdir(parents=True, exist_ok=True)

    # Создаем общий датасет
    df = create_mock_dataset(n_samples)

    # Разделяем на train/val/test
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))

    train_df = df[:train_size]
    val_df = df[train_size : train_size + val_size]
    test_df = df[train_size + val_size :]

    # Сохраняем в parquet
    train_df.to_parquet(base_dir / "train.parquet")
    val_df.to_parquet(base_dir / "val.parquet")
    test_df.to_parquet(base_dir / "test.parquet")

    print(
        f"Созданы мок-файлы: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    return train_df, val_df, test_df


def run_mock_training():
    """Запускает мок-обучение с легкими параметрами"""

    # Сначала устанавливаем переменные окружения
    mock_env = {
        "FORCE_TRAIN": "true",
        "OPTUNA_N_TRIALS": "3",
        "TFIDF_MAX_FEATURES_MAX": "120",
        "TFIDF_MAX_FEATURES_MIN": "60",
        "FORCE_SVD_THRESHOLD_MB": "10",
        "MEMORY_WARNING_MB": "100",
        "OPTUNA_TIMEOUT_SEC": "60",
        "SELECTED_MODELS": "logreg",  # Только логистическая регрессия
    }

    # Сохраняем и устанавливаем новые переменные
    original_env = {}
    for key, value in mock_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    # Создаем временную директорию для мок-данных
    temp_dir = Path(tempfile.mkdtemp(prefix="mock_kindle_"))
    mock_processed_dir = temp_dir / "processed"
    mock_model_dir = temp_dir / "model"

    try:
        print(f"Создаю мок-датасет в {temp_dir}")

        # Создаем мок-данные
        create_mock_parquet_files(mock_processed_dir, n_samples=500)

        # Добавляем пути к данным
        os.environ["PROCESSED_DATA_DIR"] = str(mock_processed_dir)
        os.environ["MODEL_DIR"] = str(mock_model_dir)

        print("Запускаю мок-обучение с легкими параметрами...")
        print("Настройки: trials=3, max_features=100, model=logreg")

        # Импортируем и запускаем обучение ПОСЛЕ установки переменных
        from scripts.train import run as train_run

        train_run()

        # Проверяем результаты
        if mock_model_dir.exists():
            model_files = list(mock_model_dir.glob("*"))
            print(f"Создано файлов моделей: {len(model_files)}")
            for f in model_files:
                print(f"  - {f.name}")
            return True
        else:
            print("Модель не была создана")
            return False

    except Exception as e:
        print(f"Ошибка при мок-обучении: {e}")
        return False
    finally:
        # Восстанавливаем оригинальные переменные окружения
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        # Удаляем временную директорию
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"Удалена временная директория {temp_dir}")


if __name__ == "__main__":
    success = run_mock_training()
    if success:
        print("✅ Мок-тест прошел успешно!")
    else:
        print("❌ Мок-тест не удался")

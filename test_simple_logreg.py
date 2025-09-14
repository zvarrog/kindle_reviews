"""
Простой мок-тест только для логистической регрессии
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import os


def create_simple_mock_data(n_samples=200):
    """Создает очень простой датасет"""
    np.random.seed(42)

    # Простые тексты
    texts_1 = ["bad terrible awful"] * (n_samples // 3)
    texts_2 = ["okay normal average"] * (n_samples // 3)
    texts_3 = ["great excellent amazing"] * (n_samples - 2 * (n_samples // 3))

    texts = texts_1 + texts_2 + texts_3
    ratings = (
        [1] * (n_samples // 3)
        + [2] * (n_samples // 3)
        + [3] * (n_samples - 2 * (n_samples // 3))
    )

    # Случайно перемешиваем
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    ratings = [ratings[i] for i in indices]

    df = pd.DataFrame(
        {
            "reviewText": texts,
            "overall": ratings,
            "text_len": [len(text) for text in texts],
            "word_count": [len(text.split()) for text in texts],
            "kindle_freq": [0.1] * n_samples,
            "sentiment": [r - 2 for r in ratings],  # -1, 0, 1
            "user_avg_len": [50.0] * n_samples,
            "user_review_count": [5] * n_samples,
            "item_avg_len": [50.0] * n_samples,
            "item_review_count": [10] * n_samples,
        }
    )

    return df


def test_simple_training():
    """Тестирует простое обучение только LogReg"""

    # Устанавливаем переменные окружения
    env_vars = {
        "FORCE_TRAIN": "true",
        "OPTUNA_N_TRIALS": "2",
        "TFIDF_MAX_FEATURES_MAX": "100",
        "TFIDF_MAX_FEATURES_MIN": "50",
        "MEMORY_WARNING_MB": "100",
        "OPTUNA_TIMEOUT_SEC": "30",
    }

    original_env = {}
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    # Создаем временные директории
    temp_dir = Path(tempfile.mkdtemp(prefix="simple_mock_"))
    processed_dir = temp_dir / "processed"
    model_dir = temp_dir / "model"
    processed_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    try:
        print(f"Создаю простой датасет в {temp_dir}")

        # Создаем данные
        df = create_simple_mock_data(200)

        # Разделяем
        train_size = int(0.7 * len(df))
        val_size = int(0.15 * len(df))

        train_df = df[:train_size]
        val_df = df[train_size : train_size + val_size]
        test_df = df[train_size + val_size :]

        # Сохраняем
        train_df.to_parquet(processed_dir / "train.parquet")
        val_df.to_parquet(processed_dir / "val.parquet")
        test_df.to_parquet(processed_dir / "test.parquet")

        print(f"Данные: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        # Устанавливаем пути
        os.environ["PROCESSED_DATA_DIR"] = str(processed_dir)
        os.environ["MODEL_DIR"] = str(model_dir)

        # Импортируем и переопределяем модели
        from scripts.models.kinds import ModelKind
        from scripts import train

        # Принудительно ограничиваем только LogReg
        train.MODEL_CHOICES = [ModelKind.logreg]

        print("Запускаю обучение только LogReg...")
        train.run()

        # Проверяем результат
        model_files = list(model_dir.glob("*"))
        print(f"Создано файлов: {len(model_files)}")
        for f in model_files:
            print(f"  - {f.name}")

        return len(model_files) > 0

    except Exception as e:
        print(f"Ошибка: {e}")
        return False
    finally:
        # Восстанавливаем переменные
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        # Удаляем временные файлы
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"Удалена временная директория {temp_dir}")


if __name__ == "__main__":
    success = test_simple_training()
    if success:
        print("✅ Простой тест успешен!")
    else:
        print("❌ Простой тест провален")

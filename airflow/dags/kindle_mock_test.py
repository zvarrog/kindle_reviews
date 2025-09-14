"""
Простой тестовый DAG для проверки работоспособности обучения в Airflow
с минимальными данными и параметрами.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

# Настройки по умолчанию для DAG
default_args = {
    'owner': 'kindle_test',
    'depends_on_past': False,
    'start_date': datetime(2025, 9, 13),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Создание DAG
dag = DAG(
    'kindle_mock_test',
    default_args=default_args,
    description='Тестовый DAG для обучения с мок-данными',
    schedule_interval=None,  # Запуск только вручную
    catchup=False,
    tags=['test', 'mock', 'kindle']
)

def create_mock_data(**context):
    """Создает небольшой мок-датасет для тестирования"""
    from scripts.settings import PROCESSED_DATA_DIR, log
    import os
    
    # Создаем мок-данные
    np.random.seed(42)
    n_samples = 300  # Очень мало данных для быстрого теста
    
    reviews = []
    ratings = []
    
    words_by_rating = {
        1: ["terrible", "awful", "hate", "worst", "bad"],
        2: ["okay", "average", "decent", "normal", "fine"],
        3: ["great", "excellent", "amazing", "wonderful", "love"]
    }
    
    for i in range(n_samples):
        rating = np.random.randint(1, 4)  # 1, 2, или 3
        words = words_by_rating[rating]
        
        # Создаем простой текст
        n_words = np.random.randint(3, 8)
        review_words = np.random.choice(words + ["book", "kindle"], n_words)
        review_text = " ".join(review_words)
        
        reviews.append(review_text)
        ratings.append(rating)
    
    # Создаем DataFrame с минимальными признаками
    df = pd.DataFrame({
        'reviewText': reviews,
        'overall': ratings,
        'text_len': [len(text) for text in reviews],
        'word_count': [len(text.split()) for text in reviews],
        'kindle_freq': np.random.random(n_samples) * 0.1,  # Низкие значения
        'sentiment': [r - 2 for r in ratings],  # -1, 0, 1
        'user_avg_len': np.random.uniform(20, 50, n_samples),
        'user_review_count': np.random.randint(1, 10, n_samples),
        'item_avg_len': np.random.uniform(20, 50, n_samples),
        'item_review_count': np.random.randint(1, 20, n_samples),
    })
    
    # Разделяем на train/val/test
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    # Создаем директорию если не существует
    processed_dir = Path(PROCESSED_DATA_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем данные
    train_df.to_parquet(processed_dir / "train.parquet")
    val_df.to_parquet(processed_dir / "val.parquet")
    test_df.to_parquet(processed_dir / "test.parquet")
    
    log.info(f"Создан мок-датасет: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return f"Создано: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test образцов"

def run_light_training(**context):
    """Запускает обучение с очень легкими параметрами"""
    import os
    from scripts.train import run as train_run
    from scripts.settings import log
    
    # Устанавливаем очень легкие параметры через переменные окружения
        os.environ['FORCE_TRAIN'] = 'true'  # Принудительно переобучаем
    os.environ['OPTUNA_N_TRIALS'] = '2'  # Всего 2 попытки
    os.environ['TFIDF_MAX_FEATURES_MAX'] = '50'  # Максимум 50 признаков
    os.environ['TFIDF_MAX_FEATURES_MIN'] = '20'  # Минимум 20 признаков
    os.environ['MEMORY_WARNING_MB'] = '50'  # Низкий лимит памяти
    os.environ['OPTUNA_TIMEOUT_SEC'] = '120'  # 2 минуты максимум
    
    log.info("Запуск легкого обучения с параметрами:")
    log.info("- trials: 2")
    log.info("- max_features: 20-50")
    log.info("- timeout: 120 сек")
    
    try:
            train_run()
            return "Обучение завершено успешно"
    except Exception as e:
        log.error(f"Ошибка при обучении: {e}")
        raise

# Определяем задачи
create_data_task = PythonOperator(
    task_id='create_mock_data',
    python_callable=create_mock_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='run_light_training',
    python_callable=run_light_training,
    dag=dag,
)

check_results_task = BashOperator(
    task_id='check_results',
    bash_command='ls -la /opt/airflow/model/ || echo "Нет файлов модели"',
    dag=dag,
)

# Устанавливаем зависимости
create_data_task >> train_task >> check_results_task
# Скачивание датасета Kindle Reviews

Этот проект содержит скрипт для скачивания датасета `bharadwaj6/kindle-reviews` с Kaggle в папку `data/raw`.

Требуемые шаги:

1. Установите зависимости:

````markdown
# Скачивание датасета Kindle Reviews

Этот проект содержит скрипт для скачивания датасета `bharadwaj6/kindle-reviews` с Kaggle в папку `data/raw`.

Требуемые шаги:

1. Установите зависимости:

```powershell
pip install --user -r requirements.txt
```

2. Настройте Kaggle credentials:
## API для инференса

Простой сервис на FastAPI для предсказаний по тексту (без перезагрузки на лету):

1. Установите зависимости:

```powershell
pip install -r requirements.txt
```

2. Запустите сервис:

```powershell
uvicorn scripts.api_service:app --host 0.0.0.0 --port 8000
```

3. Примеры запросов:

- Проверка:

```powershell
curl http://localhost:8000/health
```

- Предсказание (только текст):

```powershell
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"texts": ["Loved this Kindle!", "Terrible experience"]}'
```

Сервис автоматически подхватывает `model/best_model.joblib` и вычисляет простые числовые признаки из текста (длина, число слов, частота слова "kindle"). Для обновления модели перезапустите сервис.


- Поместите `kaggle.json` в `%USERPROFILE%\\.kaggle\\kaggle.json` (Windows), или
- Установите переменные окружения `KAGGLE_USERNAME` и `KAGGLE_KEY`.

3. Запустите скрипт:

```powershell
python .\scripts\download_kindle_reviews.py --dest data\\raw
```

Примечание: Не добавляйте `kaggle.json` в репозиторий.


## Обучение модели

1. Установите дополнительные зависимости (если ещё не установлены):

```powershell
pip install --user -r requirements.txt
```

2. Запустите тренинг (скрипт прочитает parquet из `data/processed` и сохранит лучшую модель в `model/`):

```powershell
python .\scripts\train.py
```

Флаги и поведение:

- `scripts/config.py` содержит константу `FORCE_TRAIN` — если выставлена в `False` и в папке `model/` уже есть `best_model.joblib`, тренинг пропустится.
- Результаты подготавливаются в MLflow эксперименте `kindle_classical_models`.

````

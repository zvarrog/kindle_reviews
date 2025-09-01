# Скачивание датасета Kindle Reviews

Этот проект содержит скрипт для скачивания датасета `bharadwaj6/kindle-reviews` с Kaggle в папку `data/raw`.

Требуемые шаги:

1. Установите зависимости:

```powershell
pip install --user -r requirements.txt
```

2. Настройте Kaggle credentials:

- Поместите `kaggle.json` в `%USERPROFILE%\\.kaggle\\kaggle.json` (Windows), или
- Установите переменные окружения `KAGGLE_USERNAME` и `KAGGLE_KEY`.

3. Запустите скрипт:

```powershell
python .\scripts\download_kindle_reviews.py --dest data\\raw
```

Примечание: Не добавляйте `kaggle.json` в репозиторий.

"""E2E/интеграционные смоки.

- Проверка API вживую (lifespan) через TestClient
- Проверка drift_monitor генерации отчёта
"""

from pathlib import Path
import json
import sys

from fastapi.testclient import TestClient


def test_api_smoke_lifespan():
    # Локальный запуск приложения (без Docker) с lifespan
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.api_service import app

    with TestClient(app) as client:
        # health
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json().get("status") == "ok"

        # predict с простым текстом, без агрегатов
        r = client.post(
            "/predict", json={"texts": ["kindle is great", "bad experience"]}
        )
        assert r.status_code == 200
        data = r.json()
        assert "labels" in data and isinstance(data["labels"], list)


def test_drift_monitor_report(tmp_path: Path):
    # Создаём копию тестовых данных и запускаем drift_monitor на них
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.drift_monitor import main as drift_main
    from scripts.settings import PROCESSED_DATA_DIR, MODEL_DIR

    test_parquet = Path(PROCESSED_DATA_DIR) / "test.parquet"
    assert test_parquet.exists(), "Ожидается наличие test.parquet в data/processed"

    # Меняем cwd, чтобы отчёт писался в модельную директорию проекта
    # Подменяем аргументы командной строки, чтобы не конфликтовать с pytest
    old_argv = sys.argv
    try:
        sys.argv = [
            "drift_monitor",
            "--new-path",
            str(test_parquet),
        ]
        drift_main()
        report_path = Path(MODEL_DIR) / "drift_report.json"
        assert report_path.exists(), "Ожидается drift_report.json после drift_monitor"
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
    finally:
        sys.argv = old_argv

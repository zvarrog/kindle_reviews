"""
Smoke-тесты API: базовая работоспособность эндпоинтов.
"""

from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Обеспечиваем импорт локального пакета scripts
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.api_service import app


def test_health_ok():
    # Важно: использовать контекст, чтобы сработал lifespan/startup
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") == "ok"
        # Ожидаем наличие модели в артефактах
        assert data.get("model_exists") is True


def test_predict_minimal_texts():
    payload = {"texts": ["Great book", "Not impressed"]}
    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert "labels" in body
        assert len(body["labels"]) == 2


def test_batch_predict_minimal():
    payload = {
        "data": [
            {"reviewText": "Loved it!"},
            {"reviewText": "Terrible experience"},
        ]
    }
    with TestClient(app) as client:
        r = client.post("/batch_predict", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert "predictions" in body
        assert len(body["predictions"]) == 2

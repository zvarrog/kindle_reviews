"""
Snapshot-тест метаданных модели и контракта признаков.
"""

import json
from pathlib import Path

MODEL_META_PATH = Path("model/best_model_meta.json")


def test_model_meta_snapshot_structure():
    assert MODEL_META_PATH.exists(), "Отсутствует model/best_model_meta.json"
    meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))

    # Ключи верхнего уровня
    for key in [
        "best_model",
        "best_params",
        "test_metrics",
        "sizes",
    ]:
        assert key in meta, f"Нет ключа {key} в best_model_meta.json"

    # Допустимые диапазоны метрик
    tm = meta["test_metrics"]
    assert 0.0 <= tm.get("accuracy", 0.0) <= 1.0
    assert 0.0 <= tm.get("f1_macro", 0.0) <= 1.0


def test_feature_contract_snapshot_via_api():
    # Через API получаем сводку по контракту
    from fastapi.testclient import TestClient
    from pathlib import Path
    import sys

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from scripts.api_service import app

    with TestClient(app) as client:
        r = client.get("/metadata")
    assert r.status_code == 200
    body = r.json()
    assert "feature_contract" in body
    fc = body["feature_contract"]

    # Проверяем согласованность структуры
    assert "required_text_columns" in fc
    assert "expected_numeric_columns" in fc
    assert isinstance(fc["required_text_columns"], list)
    assert isinstance(fc["expected_numeric_columns"], list)

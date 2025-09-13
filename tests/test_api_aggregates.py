"""
Тест восстановления агрегатов user/item в API.
"""

import pytest
import requests
import json

API_URL = "http://localhost:8000/predict"

# Пример данных с агрегатами
EXAMPLE = {
    "texts": ["Отличная книга, рекомендую!", "Скучно, не дочитал."],
    "numeric_features": {
        "user_avg_len": [500, 100],
        "user_review_count": [10, 2],
        "item_avg_len": [600, 80],
        "item_review_count": [15, 1],
        "text_len": [20, 10],
        "word_count": [4, 2],
        "kindle_freq": [0, 0],
        "sentiment": [0.8, -0.5],
    },
}


def test_api_aggregates():
    """
    Проверяет, что API корректно принимает и использует агрегаты user/item.
    """
    resp = requests.post(API_URL, json=EXAMPLE)
    assert resp.status_code == 200
    data = resp.json()
    assert "labels" in data
    assert len(data["labels"]) == 2
    # Проверяем, что не возникает предупреждений по признакам
    assert not data.get("warnings")
    # TODO: можно добавить проверку на влияние агрегатов на результат

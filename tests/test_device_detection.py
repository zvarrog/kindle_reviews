"""Тесты определения доступности GPU и корректности выбора устройства.

Лёгкие проверки (без запуска обучения), чтобы не дергать основной pipeline.

Поведение:
 - Если установлен REQUIRE_GPU=1 и CUDA недоступна – тест падает (явно сигнализируем).
 - Проверяем работу helper `_select_device` из `scripts.train` при разных условиях.
 - Если CUDA недоступна, соответствующие проверки помечаются как xfail/skip.
"""

import os
import sys
import pytest

# Гарантируем, что корень проекта в sys.path (для других тестов, ожидающих 'config')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


@pytest.mark.skipif(torch is None, reason="torch не установлен")
def test_cuda_basic_detection():
    """Базовая проверка: torch импортируется, cuda.is_available() вызывается.

    Если выставлен REQUIRE_GPU=1 и CUDA нет — падаем.
    Иначе просто утверждаем, что значение булево.
    """
    available = torch.cuda.is_available()
    if os.environ.get("REQUIRE_GPU") == "1" and not available:
        pytest.fail("Требуется GPU (REQUIRE_GPU=1), но torch.cuda.is_available()=False")
    assert isinstance(available, bool)


@pytest.mark.skipif(torch is None, reason="torch не установлен")
def test_select_device_prefers_requested_cuda(monkeypatch):
    """Проверяем, что _select_device возвращает ожидаемое.

    Логика:
      - При MODEL_DEVICE=cuda и доступной CUDA -> 'cuda'
      - При MODEL_DEVICE=cuda и отсутствии CUDA -> 'cpu'
    """
    try:
        from scripts import train  # импортирует _select_device
    except ModuleNotFoundError:
        # Создаём временный модуль config в sys.modules, затем повторяем импорт
        import types

        cfg = types.ModuleType("config")
        cfg.PROCESSED_DATA_DIR = "./data/processed"
        cfg.MODEL_DIR = "./model"
        cfg.FORCE_TRAIN = False
        sys.modules["config"] = cfg
        from scripts import train

    monkeypatch.setenv("MODEL_DEVICE", "cuda")
    got = train._select_device(None)
    if torch.cuda.is_available():
        assert got == "cuda"
    else:
        assert got == "cpu"


@pytest.mark.skipif(torch is None, reason="torch не установлен")
def test_select_device_auto(monkeypatch):
    """Проверяет авто-режим (без MODEL_DEVICE)."""
    try:
        from scripts import train
    except ModuleNotFoundError:
        import types

        cfg = types.ModuleType("config")
        cfg.PROCESSED_DATA_DIR = "./data/processed"
        cfg.MODEL_DIR = "./model"
        cfg.FORCE_TRAIN = False
        sys.modules["config"] = cfg
        from scripts import train

    if "MODEL_DEVICE" in os.environ:
        monkeypatch.delenv("MODEL_DEVICE", raising=False)
    got = train._select_device(None)
    if torch.cuda.is_available():
        assert got == "cuda"
    else:
        assert got == "cpu"


@pytest.mark.skipif(torch is None, reason="torch не установлен")
def test_enforce_gpu_fail_fast(monkeypatch):
    """Если хотим жёстко валидировать наличие GPU в CI.

    Установи переменную окружения REQUIRE_GPU_STRICT=1 чтобы включить.
    Без неё тест просто пропускается.
    """
    if os.environ.get("REQUIRE_GPU_STRICT") != "1":
        pytest.skip("Строгая проверка GPU отключена (REQUIRE_GPU_STRICT!=1)")
    assert torch.cuda.is_available(), "GPU обязательна, но недоступна"

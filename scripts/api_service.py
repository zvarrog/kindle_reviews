import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from pydantic import BaseModel

# Настраиваем логирование для API (автоматическое определение среды)
from .logging_config import (
    setup_auto_logging,
    set_trace_id,
    clear_trace_id,
    get_trace_id,
)

log = setup_auto_logging()

# Фоллбек: если baseline не загружен, используем список числовых фич из тренера
try:  # noqa: WPS501
    from scripts.train import NUMERIC_COLS as _TRAIN_NUMERIC_COLS  # type: ignore
except Exception:  # noqa: BLE001
    _TRAIN_NUMERIC_COLS = []

from .settings import MODEL_DIR
from .feature_contract import FeatureContract

BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
META_PATH = MODEL_DIR / "best_model_meta.json"
BASELINE_NUMERIC_PATH = MODEL_DIR / "baseline_numeric_stats.json"


class PredictRequest(BaseModel):
    texts: List[str]
    # Опциональные числовые признаки для более точного предсказания
    numeric_features: Optional[Dict[str, List[float]]] = None


class PredictResponse(BaseModel):
    labels: List[int]
    probs: Optional[List[List[float]]] = None
    warnings: Optional[Dict[str, List[str]]] = None


class BatchPredictRequest(BaseModel):
    data: List[
        Dict[str, Any]
    ]  # Список объектов с полями reviewText и числовыми признаками


class BatchPredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    # предупреждения по каждому объекту: item_{i} -> {category: [messages]}
    warnings: Optional[Dict[str, Dict[str, List[str]]]] = None


class MetadataResponse(BaseModel):
    model_info: Dict[str, Any]
    feature_contract: Dict[str, Any]
    health: Dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация при старте
    _load_artifacts()
    yield
    # Здесь можно добавить освобождение ресурсов при остановке


app = FastAPI(title="Kindle Reviews API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Middleware: добавляет/прокидывает X-Request-ID и устанавливает trace_id."""
    req_id = request.headers.get("X-Request-ID")
    if not req_id:
        # Простой вариант без зависимостей: используем id объекта и время
        req_id = f"req-{id(request)}"
    set_trace_id(req_id)
    log.info("Запрос: %s %s, X-Request-ID=%s", request.method, request.url.path, req_id)
    try:
        response = await call_next(request)
    finally:
        # В логе после обработки
        log.info(
            "Ответ: %s %s -> %s, X-Request-ID=%s",
            request.method,
            request.url.path,
            getattr(request.state, "status_code", "?"),
            get_trace_id(),
        )
        clear_trace_id()
    # Проставляем заголовок в ответ
    response.headers["X-Request-ID"] = req_id
    return response


def _load_artifacts():
    log.info("Загрузка артефактов модели...")
    if not BEST_MODEL_PATH.exists():
        log.error("Модель не найдена: %s", BEST_MODEL_PATH)
        raise FileNotFoundError(f"Модель не найдена: {BEST_MODEL_PATH}")

    app.state.PIPELINE = joblib.load(BEST_MODEL_PATH)
    log.info("Модель загружена: %s", BEST_MODEL_PATH)

    app.state.META = (
        json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    )
    app.state.NUMERIC_DEFAULTS = (
        json.loads(BASELINE_NUMERIC_PATH.read_text(encoding="utf-8"))
        if BASELINE_NUMERIC_PATH.exists()
        else {}
    )
    # Загружаем feature contract
    app.state.FEATURE_CONTRACT = FeatureContract.from_model_artifacts(MODEL_DIR)
    log.info("Артефакты модели успешно загружены")


def _build_dataframe(
    texts: List[str], numeric_features: Optional[Dict[str, List[float]]] = None
) -> pd.DataFrame:
    """Строит DataFrame для предсказания из текстов и опциональных числовых признаков."""
    df = pd.DataFrame({"reviewText": texts})

    # Получаем список ожидаемых числовых колонок
    feature_contract = getattr(app.state, "FEATURE_CONTRACT", None)
    if feature_contract:
        numeric_cols = feature_contract.expected_numeric_columns
    elif getattr(app.state, "NUMERIC_DEFAULTS", None):
        numeric_cols = list(app.state.NUMERIC_DEFAULTS.keys())
    elif _TRAIN_NUMERIC_COLS:
        numeric_cols = list(_TRAIN_NUMERIC_COLS)
    else:
        numeric_cols = []

    # Если переданы числовые признаки, используем их
    if numeric_features:
        for col, values in numeric_features.items():
            if col in numeric_cols and len(values) == len(texts):
                df[col] = values

    # Посчитаем лёгкие признаки из текста, если они нужны и не переданы
    need_text_len = "text_len" in numeric_cols and "text_len" not in df.columns
    need_word_count = "word_count" in numeric_cols and "word_count" not in df.columns
    need_kindle_freq = "kindle_freq" in numeric_cols and "kindle_freq" not in df.columns

    if need_text_len or need_word_count or need_kindle_freq:
        s = df["reviewText"].fillna("")
        if need_text_len:
            df["text_len"] = s.str.len().astype(float)
        if need_word_count:
            df["word_count"] = s.str.split().str.len().fillna(0).astype(float)
        if need_kindle_freq:
            df["kindle_freq"] = s.str.lower().str.count("kindle").astype(float)

    # Остальные требуемые числовые колонки заполним базовыми значениями или нулями
    baseline_stats = getattr(app.state, "NUMERIC_DEFAULTS", {})
    for col in numeric_cols:
        if col not in df.columns:
            default_val = baseline_stats.get(col, {}).get("mean", 0.0)
            df[col] = default_val

    return df


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_exists": BEST_MODEL_PATH.exists(),
        "best_model": getattr(app.state, "META", {}).get("best_model"),
    }


@app.get("/metadata", response_model=MetadataResponse)
def get_metadata():
    """Возвращает метаданные модели и информацию о признаках."""
    meta = getattr(app.state, "META", {})
    feature_contract = getattr(app.state, "FEATURE_CONTRACT", None)

    model_info = {
        "best_model": meta.get("best_model", "unknown"),
        "best_params": meta.get("best_params", {}),
        "test_metrics": meta.get("test_metrics", {}),
        "training_duration_sec": meta.get("duration_sec", None),
        "dataset_sizes": meta.get("sizes", {}),
    }

    feature_info = feature_contract.get_feature_info() if feature_contract else {}

    health_info = {
        "model_loaded": getattr(app.state, "PIPELINE", None) is not None,
        "baseline_stats_loaded": bool(getattr(app.state, "NUMERIC_DEFAULTS", {})),
        "feature_contract_loaded": feature_contract is not None,
    }

    return MetadataResponse(
        model_info=model_info, feature_contract=feature_info, health=health_info
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Предсказание для списка текстов с опциональными числовыми признаками."""
    pipeline = getattr(app.state, "PIPELINE", None)
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")
    if not req.texts:
        raise HTTPException(status_code=400, detail="Список texts пуст")

    # Валидация признаков
    warnings = {}
    feature_contract = getattr(app.state, "FEATURE_CONTRACT", None)
    if feature_contract and req.numeric_features:
        validation_data = {"reviewText": req.texts}
        validation_data.update(req.numeric_features)
        warnings = feature_contract.validate_input_data(validation_data)

    # Определим, нужен ли только текст, или текст + числовые
    name = pipeline.__class__.__name__.lower()
    if "distil" in name:
        preds = pipeline.predict(pd.Series(req.texts))
        probs = None
        if hasattr(pipeline, "predict_proba"):
            try:
                probs = pipeline.predict_proba(pd.Series(req.texts)).tolist()
            except (AttributeError, ValueError, TypeError):
                probs = None
        return PredictResponse(
            labels=[int(x) for x in preds],
            probs=probs,
            warnings=warnings if warnings else None,
        )

    # Иначе считаем, что это sklearn Pipeline
    df = _build_dataframe(req.texts, req.numeric_features)
    preds = pipeline.predict(df)
    probs = None
    if hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba(df).tolist()
        except (AttributeError, ValueError, TypeError):
            probs = None
    return PredictResponse(
        labels=[int(x) for x in preds],
        probs=probs,
        warnings=warnings if warnings else None,
    )


@app.post("/batch_predict", response_model=BatchPredictResponse)
def batch_predict(req: BatchPredictRequest):
    """Пакетное предсказание для списка объектов с полными данными."""
    pipeline = getattr(app.state, "PIPELINE", None)
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")
    if not req.data:
        raise HTTPException(status_code=400, detail="Список data пуст")

    # Валидация данных
    all_warnings = {}
    feature_contract = getattr(app.state, "FEATURE_CONTRACT", None)

    if feature_contract:
        for i, item in enumerate(req.data):
            item_warnings = feature_contract.validate_input_data(item)
            if item_warnings:
                all_warnings[f"item_{i}"] = item_warnings

    # Строим DataFrame из всех объектов
    texts = []
    numeric_features = {}

    for item in req.data:
        texts.append(item.get("reviewText", ""))
        # Собираем числовые признаки
        for key, value in item.items():
            if key != "reviewText" and isinstance(value, (int, float)):
                if key not in numeric_features:
                    numeric_features[key] = []
                numeric_features[key].append(float(value))

    # Выравниваем длины списков числовых признаков
    for key, values in numeric_features.items():
        while len(values) < len(texts):
            values.append(0.0)

    # Предсказание
    name = pipeline.__class__.__name__.lower()
    if "distil" in name:
        preds = pipeline.predict(pd.Series(texts))
        probs = None
        if hasattr(pipeline, "predict_proba"):
            try:
                probs_array = pipeline.predict_proba(pd.Series(texts))
                probs = [probs_array[i].tolist() for i in range(len(probs_array))]
            except (AttributeError, ValueError, TypeError):
                probs = None
    else:
        df = _build_dataframe(texts, numeric_features)
        preds = pipeline.predict(df)
        probs = None
        if hasattr(pipeline, "predict_proba"):
            try:
                probs_array = pipeline.predict_proba(df)
                probs = [probs_array[i].tolist() for i in range(len(probs_array))]
            except (AttributeError, ValueError, TypeError):
                probs = None

    # Формируем результат
    predictions = []
    for i in range(len(texts)):
        pred_item = {
            "index": i,
            "prediction": int(preds[i]),
        }
        if probs:
            pred_item["probabilities"] = probs[i]
        predictions.append(pred_item)

    return BatchPredictResponse(
        predictions=predictions, warnings=all_warnings if all_warnings else None
    )


# Локальный запуск: uvicorn scripts.api_service:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

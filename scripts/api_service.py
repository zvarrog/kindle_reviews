import os
import json
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel


MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
META_PATH = MODEL_DIR / "best_model_meta.json"
BASELINE_NUMERIC_PATH = MODEL_DIR / "baseline_numeric_stats.json"


class PredictRequest(BaseModel):
    texts: List[str]


class PredictResponse(BaseModel):
    labels: List[int]
    probs: Optional[List[List[float]]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация при старте
    _load_artifacts()
    yield
    # Здесь можно добавить освобождение ресурсов при остановке


app = FastAPI(title="Kindle Reviews API", version="1.0.0", lifespan=lifespan)


def _load_artifacts():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Модель не найдена: {BEST_MODEL_PATH}")
    app.state.PIPELINE = joblib.load(BEST_MODEL_PATH)
    app.state.META = (
        json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    )
    app.state.NUMERIC_DEFAULTS = (
        json.loads(BASELINE_NUMERIC_PATH.read_text(encoding="utf-8"))
        if BASELINE_NUMERIC_PATH.exists()
        else {}
    )


def _build_dataframe(texts: List[str]) -> pd.DataFrame:
    # Минимальные столбцы, которые ожидает тренер: 'reviewText' + числовые признаки
    df = pd.DataFrame({"reviewText": texts})
    # Извлечём список numeric_cols из baseline (ключи baseline_numeric_stats.json)
    numeric_cols: List[str] = (
        list(app.state.NUMERIC_DEFAULTS.keys()) if getattr(app.state, "NUMERIC_DEFAULTS", None) else []
    )

    # Посчитаем несколько лёгких признаков из текста, если они нужны
    need_text_len = "text_len" in numeric_cols
    need_word_count = "word_count" in numeric_cols
    need_kindle_freq = "kindle_freq" in numeric_cols

    if need_text_len or need_word_count or need_kindle_freq:
        # Векторизуем расчёт по Series
        s = df["reviewText"].fillna("")
        if need_text_len:
            df["text_len"] = s.str.len().astype(float)
        if need_word_count:
            df["word_count"] = s.str.split().str.len().fillna(0).astype(float)
        if need_kindle_freq:
            df["kindle_freq"] = s.str.lower().str.count("kindle").astype(float)

    # Остальные требуемые числовые колонки добавим нулями (если ещё не созданы)
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_exists": BEST_MODEL_PATH.exists(),
        "best_model": getattr(app.state, "META", {}).get("best_model"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    pipeline = getattr(app.state, "PIPELINE", None)
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")
    if not req.texts:
        raise HTTPException(status_code=400, detail="Список texts пуст")

    # Определим, нужен ли только текст, или текст + числовые
    # Если пайплайн — это наш DistilBertClassifier (имеет метод predict, принимает Series/List),
    # распознаем по наличию атрибутов/имени класса.
    name = pipeline.__class__.__name__.lower()
    if "distil" in name:
        preds = pipeline.predict(pd.Series(req.texts))
        probs = None
        if hasattr(pipeline, "predict_proba"):
            try:
                probs = pipeline.predict_proba(pd.Series(req.texts)).tolist()
            except (AttributeError, ValueError, TypeError):
                probs = None
        return PredictResponse(labels=[int(x) for x in preds], probs=probs)

    # Иначе считаем, что это sklearn Pipeline, ожидающий DataFrame с 'reviewText' и числовыми
    df = _build_dataframe(req.texts)
    preds = pipeline.predict(df)
    probs = None
    if hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba(df).tolist()
        except (AttributeError, ValueError, TypeError):
            probs = None
    return PredictResponse(labels=[int(x) for x in preds], probs=probs)


# Локальный запуск: uvicorn scripts.api_service:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
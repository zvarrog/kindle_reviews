"""Полноценный training pipeline классических моделей с Optuna + MLflow.

Что добавлено относительно базовой версии:
 - Корректное разделение: train.parquet / val.parquet / test.parquet (валидация только на val)
 - TF-IDF вместо Hashing (контролируемое max_features) + опциональный TruncatedSVD (понижение размерности)
 - Стандартизация числовых признаков (StandardScaler)
 - Выбор метрики оптимизации: macro F1 (лучше для несбалансированных классов)
 - Логирование в MLflow: параметры, метрики (train/val/test), артефакты (confusion matrix, classification report)
 - Optuna c SQLite storage (возобновляемость) + MedianPruner
 - Retrain лучшего пайплайна на train+val и финальная оценка на test
 - Метаданные: размеры датасетов, время выполнения, версии библиотек
 - Детальная структура кода без временных демо (для production impression)

Файл не содержит тестового «вспомогательного» кода; всё экспериментальное должно располагаться в tests/.
"""

from pathlib import Path
import logging
import json
import hashlib
import os
import time
import random
import joblib
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import mlflow
import optuna

from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    FORCE_TRAIN,
    SELECTED_MODELS,
    OPTUNA_N_TRIALS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ===== Optional heavy deps (кэшируем один раз) =====
try:  # noqa: WPS501
    import torch as _TORCH  # type: ignore
except Exception:  # noqa: BLE001
    _TORCH = None  # type: ignore

try:  # noqa: WPS501
    from transformers import AutoTokenizer as _AutoTokenizer, AutoModel as _AutoModel  # type: ignore
except Exception:  # noqa: BLE001
    _AutoTokenizer = None  # type: ignore
    _AutoModel = None  # type: ignore


# ===== Device helper =====
def _select_device(requested: Optional[str] = None) -> str:
    """Выбор устройства с логированием причин fallback.

    Логика:
      1. Если передано явно (аргумент или ENV MODEL_DEVICE), пробуем его.
      2. Иначе cuda если доступна, иначе cpu.
    Возвращает строку ('cuda'|'cpu').
    """
    try:  # локальный импорт torch только при наличии
        import torch  # noqa: WPS433
    except Exception:  # noqa: BLE001
        log.warning("torch не установлен — использую cpu")
        return "cpu"

    env_req = os.environ.get("MODEL_DEVICE")
    if requested is None and env_req:
        requested = env_req.strip().lower()

    if requested:
        if requested.startswith("cuda"):
            if torch.cuda.is_available():
                return "cuda"
            log.warning(
                "Запрошено устройство '%s', но torch.cuda.is_available()=False. Переход на cpu.",
                requested,
            )
            return "cpu"
        if requested in {"cpu", "mps"}:
            if requested == "mps":
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"  # (на macOS)
                log.warning("Запрошено mps, но оно недоступно — cpu")
                return "cpu"
            return "cpu"
        log.warning("Неизвестное устройство '%s' — cpu", requested)
        return "cpu"

    # Автовыбор
    if torch.cuda.is_available():
        return "cuda"
    # (не добавляем MPS для Windows)
    return "cpu"


def _gpu_env_summary() -> Dict[str, Optional[str]]:
    """Возвращает информацию об окружающей среде GPU (без жёстких зависимостей)."""
    info: Dict[str, Optional[str]] = {
        "torch_installed": None,
        "cuda_available": None,
        "cuda_device_count": None,
        "cuda_device0_name": None,
        "selected_device": None,
        "env_MODEL_DEVICE": os.environ.get("MODEL_DEVICE"),
        "env_REQUIRE_GPU": os.environ.get("REQUIRE_GPU"),
    }
    try:  # noqa: WPS501
        import torch  # noqa: WPS433

        info["torch_installed"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_count"] = str(torch.cuda.device_count())
            try:  # отдельный перехват имени (иногда падает)
                info["cuda_device0_name"] = torch.cuda.get_device_name(0)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        info["torch_installed"] = "no"
    return info


def _resolve_training_device() -> str:
    """Определяем устройство один раз с учётом ENV и REQUIRE_GPU.

    ENV:
      MODEL_DEVICE = cpu|cuda (приоритет)
      REQUIRE_GPU = 1  -> если GPU недоступна, выбрасываем RuntimeError.
    """
    requested = os.environ.get("MODEL_DEVICE")
    device = _select_device(requested)
    summary = _gpu_env_summary()
    summary["selected_device"] = device
    log.info("GPU summary: %s", summary)
    if os.environ.get("REQUIRE_GPU") == "1" and device != "cuda":
        raise RuntimeError(
            "REQUIRE_GPU=1, но CUDA недоступна или не выбрана. summary=" + str(summary)
        )
    return device


# Определяем устройство один раз (глобально для всех моделей)
try:  # noqa: WPS501
    TRAIN_DEVICE = _resolve_training_device()
except Exception as _dev_err:  # noqa: BLE001
    log.warning("Не удалось определить GPU устройство (%s) — fallback cpu", _dev_err)
    TRAIN_DEVICE = "cpu"


# ===== Constants =====
EXPERIMENT_NAME = "kindle_classical_models"
OPTUNA_STORAGE = "sqlite:///optuna_study.db"
STUDY_BASE_NAME = "classical_text_models"
SEED = 42
EARLY_STOP_PATIENCE = 8
N_FOLDS: int = int(os.environ.get("CV_N_FOLDS", 1))  # если 1 – holdout
OPTUNA_TIMEOUT_SEC = 3600  # ограничение общего времени оптимизации
MEM_WARN_MB = float(
    os.environ.get("DENSE_MEM_WARN_MB", 1024)
)  # предупреждение при конвертации > N MB

NUMERIC_COLS = [
    "text_len",
    "word_count",
    "kindle_freq",
    "sentiment",
    "user_avg_len",
    "user_review_count",
    "item_avg_len",
    "item_review_count",
]

# Dynamic model list & study name
MODEL_CHOICES = SELECTED_MODELS
SEARCH_SPACE_VERSION = "v2"
_model_sig = hashlib.md5(
    (",".join(MODEL_CHOICES) + "|" + SEARCH_SPACE_VERSION).encode("utf-8")
).hexdigest()[:8]
OPTUNA_STUDY_NAME = f"{STUDY_BASE_NAME}_{_model_sig}"


class SimpleMLP(BaseEstimator, ClassifierMixin):
    """Простейшая MLP (sklearn-совместимая) поверх плотных признаков.

    Ожидает на входе numpy массив (поэтому применима после DenseTransformer или когда preprocessor выдаёт dense).
    Использует PyTorch только если доступен.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        epochs: int = 5,
        lr: float = 1e-3,
        batch_size: int = 256,
        seed: int = SEED,
        device: Optional[str] = None,
    ):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self._fitted = False
        self._classes_: Optional[np.ndarray] = None
        self._model = None
        self._device = device  # 'cuda' | 'cpu' | None(auto)
        # Мягкая ранняя проверка зависимостей
        if _TORCH is None:
            log.warning(
                "SimpleMLP: torch не установлен — модель будет недоступна при fit()"
            )

    def fit(self, X, y):  # type: ignore[override]
        if _TORCH is None:
            raise ImportError("Для SimpleMLP требуется пакет torch. Установите torch.")
        from torch import nn  # type: ignore

        _TORCH.Generator().manual_seed(self.seed)
        device_str = _select_device(self._device)
        device = _TORCH.device(device_str)
        try:
            if device_str == "cpu":
                log.info(
                    "SimpleMLP: выбран cpu (cuda.is_available()=%s, env MODEL_DEVICE=%s)",
                    getattr(_TORCH.cuda, "is_available", lambda: False)(),
                    os.environ.get("MODEL_DEVICE"),
                )
            else:
                log.info("SimpleMLP использует устройство: %s", device)
        except Exception:  # noqa: BLE001
            log.info("SimpleMLP использует устройство: %s", device)
        # Label encoding (важно: исходные метки могут быть 1..5, а не 0..N-1)
        unique_labels = np.unique(y)
        self._classes_ = unique_labels  # сохраняем порядок возрастания
        label2idx = {lab: i for i, lab in enumerate(unique_labels)}
        y_idx = np.vectorize(label2idx.get)(y).astype(int)

        X_ = _TORCH.tensor(X.astype(np.float32), device=device)
        y_ = _TORCH.tensor(y_idx, device=device)
        in_dim = X_.shape[1]
        n_classes = len(unique_labels)
        model = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, n_classes),
        )
        model.to(device)
        opt = _TORCH.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        model.train()
        for _ in range(self.epochs):
            for i in range(0, len(X_), self.batch_size):
                xb = X_[i : i + self.batch_size]
                yb = y_[i : i + self.batch_size]
                opt.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
        self._model = model
        self._device_actual = device
        self._fitted = True
        return self

    def predict(self, X):  # type: ignore[override]
        self._ensure_fitted()
        if _TORCH is None:
            raise RuntimeError("torch недоступен во время predict")
        with _TORCH.no_grad():
            device = getattr(self, "_device_actual", _TORCH.device("cpu"))
            t = _TORCH.tensor(X.astype(np.float32), device=device)
            logits = self._model(t)
            pred_idx = logits.argmax(dim=1).cpu().numpy()
        # обратное преобразование к исходным меткам
        return self._classes_[pred_idx]

    def _ensure_fitted(self):  # noqa: D401
        if not self._fitted:
            raise RuntimeError("SimpleMLP не обучена")


class DistilBertClassifier(BaseEstimator, ClassifierMixin):
    """Лёгкая обёртка DistilBERT: замороженный encoder + линейная голова.

    Вход — тексты (list/ndarray/Series). Опционально добавляем биграммы к тексту.
    """

    def __init__(
        self,
        epochs: int = 2,
        lr: float = 2e-5,
        max_len: int = 160,
        batch_size: int = 16,
        seed: int = SEED,
        device: Optional[str] = None,
        use_bigrams: bool = False,
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.max_len = max_len
        self.batch_size = batch_size
        self.seed = seed
        self._fitted = False
        self._tokenizer: Optional[object] = None
        self._base_model: Optional[object] = None
        self._head: Optional[object] = None
        self._device = device
        self._device_actual: Optional[object] = None
        self._classes_: Optional[np.ndarray] = None
        self.use_bigrams = use_bigrams
        if _TORCH is None or _AutoTokenizer is None or _AutoModel is None:
            log.warning(
                "DistilBertClassifier: torch/transformers не установлены — модель будет пропущена"
            )

    @staticmethod
    def _augment_texts(texts):
        def _augment_bigrams(s: str) -> str:
            ws = [w for w in s.split() if w and w not in ENGLISH_STOP_WORDS]
            bigrams = [f"{ws[i]}_{ws[i+1]}" for i in range(len(ws) - 1)]
            return s + (" " + " ".join(bigrams[:20]) if bigrams else "")

        return np.array([_augment_bigrams(t) for t in texts])

    def fit(self, X, y):  # type: ignore[override]
        texts = X if isinstance(X, (list, np.ndarray, pd.Series)) else X.values
        if self.use_bigrams:
            texts = self._augment_texts(texts)
        if _TORCH is None or _AutoTokenizer is None or _AutoModel is None:
            raise ImportError(
                "Для DistilBertClassifier нужны пакеты torch и transformers. Установите их."
            )
        from torch import nn  # type: ignore

        _TORCH.manual_seed(self.seed)
        tokenizer = _AutoTokenizer.from_pretrained("distilbert-base-uncased")
        base_model = _AutoModel.from_pretrained("distilbert-base-uncased")
        device_str = _select_device(self._device)
        device = _TORCH.device(device_str)
        try:
            if device_str == "cpu":
                log.info(
                    "DistilBERT: выбран cpu (cuda.is_available()=%s, env MODEL_DEVICE=%s)",
                    _TORCH.cuda.is_available(),
                    os.environ.get("MODEL_DEVICE"),
                )
            else:
                log.info("DistilBERT использует устройство: %s", device)
        except Exception:  # noqa: BLE001
            log.info("DistilBERT использует устройство: %s", device)

        for p in base_model.parameters():
            p.requires_grad = False

        unique_labels = np.unique(y)
        self._classes_ = unique_labels
        label2idx = {lab: i for i, lab in enumerate(unique_labels)}

        hidden = base_model.config.hidden_size
        n_classes = len(unique_labels)
        head = nn.Linear(hidden, n_classes).to(device)
        base_model.to(device)
        optimizer = _TORCH.optim.Adam(head.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        base_model.eval()
        head.train()

        def batch_iter():
            for i in range(0, len(texts), self.batch_size):
                yield texts[i : i + self.batch_size], y[i : i + self.batch_size]

        for _ in range(self.epochs):
            for bt, by_raw in batch_iter():
                by = np.vectorize(label2idx.get)(by_raw).astype(int)
                enc = tokenizer(
                    list(bt),
                    truncation=True,
                    padding=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                with _TORCH.no_grad():
                    out = base_model(**enc)
                    cls = out.last_hidden_state[:, 0]
                logits = head(cls)
                loss = loss_fn(
                    logits, _TORCH.tensor(by, dtype=_TORCH.long, device=device)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._tokenizer = tokenizer
        self._base_model = base_model
        self._head = head.eval()
        self._device_actual = device
        self._fitted = True
        return self

    def predict(self, X):  # type: ignore[override]
        texts = X if isinstance(X, (list, np.ndarray, pd.Series)) else X.values
        if self.use_bigrams:
            texts = self._augment_texts(texts)
        self._ensure()
        if _TORCH is None:
            raise RuntimeError("torch недоступен во время predict")
        preds: List[int] = []
        device = getattr(self, "_device_actual", _TORCH.device("cpu"))
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self._tokenizer(
                list(batch),
                truncation=True,
                padding=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with _TORCH.no_grad():
                out = self._base_model(**enc)
                cls = out.last_hidden_state[:, 0]
                logits = self._head(cls)
                preds.extend(logits.argmax(dim=1).cpu().numpy())
        preds = np.array(preds)
        return self._classes_[preds]

    def _ensure(self) -> None:
        if not self._fitted:
            raise RuntimeError("DistilBertClassifier не обучен")


def set_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class DenseTransformer(TransformerMixin, BaseEstimator):
    """Превращает scipy sparse в dense numpy (для HistGradientBoosting) + контроль памяти."""

    def fit(self, _X, _y=None):
        return self

    def transform(self, X):  # noqa: D401
        if hasattr(X, "toarray"):
            arr = X.toarray()
        else:
            arr = X
        size_mb = arr.nbytes / (1024 * 1024)
        if size_mb > MEM_WARN_MB:
            log.warning(
                "DenseTransformer: размер dense=%.1f MB > %.0f MB (риск памяти)",
                size_mb,
                MEM_WARN_MB,
            )
        return arr


def load_splits() -> (
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]
):
    """Загружает сплиты; если отсутствуют val/test — формирует стратифицированно."""
    from sklearn.model_selection import train_test_split

    base = Path(PROCESSED_DATA_DIR)
    train_p = base / "train.parquet"
    val_p = base / "val.parquet"
    test_p = base / "test.parquet"
    if not train_p.exists():
        raise FileNotFoundError(f"Нет train.parquet: {train_p}")
    df_train = pd.read_parquet(train_p)
    if not {"reviewText", "overall"}.issubset(df_train.columns):
        raise KeyError("Ожидаются колонки 'reviewText' и 'overall'")

    frames = {"train": df_train}
    if val_p.exists():
        frames["val"] = pd.read_parquet(val_p)
    if test_p.exists():
        frames["test"] = pd.read_parquet(test_p)

    if "val" not in frames or "test" not in frames:
        # Объединяем всё и создаём 70/15/15 стратифицированно
        full = df_train.copy()
        if "val" in frames:
            full = pd.concat([full, frames["val"]], ignore_index=True)
        if "test" in frames:
            full = pd.concat([full, frames["test"]], ignore_index=True)
        full = full.dropna(subset=["reviewText", "overall"])
        y_full = full["overall"].astype(int)
        temp_X, X_test, temp_y, y_test = train_test_split(
            full, y_full, test_size=0.15, stratify=y_full, random_state=SEED
        )
        X_train, X_val, y_train, y_val = train_test_split(
            temp_X, temp_y, test_size=0.17647, stratify=temp_y, random_state=SEED
        )  # 0.17647 * 0.85 ≈ 0.15
    else:
        X_train = frames["train"]
        X_val = frames["val"]
        X_test = frames["test"]
        y_train = X_train["overall"].astype(int)
        y_val = X_val["overall"].astype(int)
        y_test = X_test["overall"].astype(int)

    def clean(df: pd.DataFrame):
        keep = ["reviewText", "overall"] + [c for c in NUMERIC_COLS if c in df.columns]
        return df[keep].dropna(subset=["reviewText", "overall"])

    X_train = clean(X_train)
    X_val = clean(X_val)
    X_test = clean(X_test)
    y_train = X_train["overall"].astype(int).values
    y_val = X_val["overall"].astype(int).values
    y_test = X_test["overall"].astype(int).values
    X_train = X_train.drop(columns=["overall"])
    X_val = X_val.drop(columns=["overall"])
    X_test = X_test.drop(columns=["overall"])
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_pipeline(trial: optuna.Trial, model_name: str) -> Pipeline:
    """Строит sklearn Pipeline для trial.

    Состав:
      - ColumnTransformer: text -> TF-IDF(+опц. SVD), numeric -> Impute+Scale
      - (опц.) DenseTransformer для hist_gb
      - Финальная модель
    """
    # DistilBERT отдельный (без TF-IDF)
    if model_name == "distilbert":
        epochs = trial.suggest_int("db_epochs", 1, 2)
        lr = trial.suggest_float("db_lr", 1e-5, 5e-5, log=True)
        max_len = trial.suggest_int("db_max_len", 96, 192, step=32)
        use_bi = trial.suggest_categorical("db_use_bigrams", [False, True])
        clf = DistilBertClassifier(
            epochs=epochs,
            lr=lr,
            max_len=max_len,
            device=TRAIN_DEVICE,
            use_bigrams=use_bi,
        )
        return Pipeline([("distilbert", clf)])

    # Общая лёгкая предобработка TF-IDF: удаляем стоп-слова, опционально стеммим
    # Для стемминга используем SnowballStemmer (нересурсоёмко). Подключим при необходимости.
    def _maybe_stemming_analyzer():
        use_stemming = trial.suggest_categorical("use_stemming", [False, True])
        if not use_stemming:
            return "word"  # стандартный анализатор TfidfVectorizer
        try:
            from nltk.stem import SnowballStemmer  # noqa: WPS433
            import re  # noqa: WPS433

            # используем список стоп-слов из sklearn, чтобы не тянуть nltk.corpus
            from sklearn.feature_extraction.text import (
                ENGLISH_STOP_WORDS,
            )  # noqa: WPS433

            stemmer = SnowballStemmer("english")

            def analyzer(text: str):
                # базовая очистка спецсимволов до токенизации
                text_clean = re.sub(
                    r"[\u200b\ufeff\u00A0]", " ", text
                )  # невидимые пробелы
                text_clean = re.sub(r"\s+", " ", text_clean).strip()
                # токены как в стандартном Tfidf (простая разбивка по не-буквам)
                tokens = re.findall(r"[A-Za-z]+", text_clean.lower())
                tokens = [
                    t for t in tokens if len(t) > 1 and t not in ENGLISH_STOP_WORDS
                ]
                return [stemmer.stem(t) for t in tokens]

            return analyzer
        except Exception:
            # если nltk не установлен — тихо пропускаем стемминг
            return "word"

    if model_name == "logreg":  # без SVD, расширяем max_features, float32
        text_max_features = trial.suggest_int(
            "tfidf_max_features", 5000, 15000, step=1000
        )
        tfidf = TfidfVectorizer(
            max_features=text_max_features,
            ngram_range=(1, 2),
            dtype=np.float32,
            stop_words="english",
            analyzer=_maybe_stemming_analyzer(),
        )
        use_svd = False
        text_steps = [("tfidf", tfidf)]
    else:
        text_max_features = trial.suggest_int(
            "tfidf_max_features", 5000, 15000, step=1000
        )
        use_svd = trial.suggest_categorical("use_svd", [False, True])
        if use_svd:
            svd_components = trial.suggest_int("svd_components", 100, 300, step=50)
        tfidf = TfidfVectorizer(
            max_features=text_max_features,
            ngram_range=(1, 2),
            dtype=np.float32,
            stop_words="english",
            analyzer=_maybe_stemming_analyzer(),
        )
        text_steps = [("tfidf", tfidf)]
        if use_svd:
            text_steps.append(
                ("svd", TruncatedSVD(n_components=svd_components, random_state=SEED))
            )
    text_pipeline = Pipeline(text_steps)

    numeric_available = [
        c
        for c in NUMERIC_COLS
        if c in trial.user_attrs.get("numeric_cols", NUMERIC_COLS)
    ]
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_pipeline, "reviewText"),
            ("num", numeric_pipeline, numeric_available),
        ],
        sparse_threshold=0.3,
    )
    steps = [("pre", preprocessor)]

    if model_name == "logreg":
        C = trial.suggest_float("logreg_C", 1e-4, 1e2, log=True)
        solver = trial.suggest_categorical(
            "logreg_solver", ["lbfgs", "liblinear", "saga"]
        )  # noqa: E501
        # допустимые penalty зависят от solver
        if solver == "lbfgs":
            penalty = trial.suggest_categorical(
                "logreg_penalty", ["l2"]
            )  # lbfgs поддерживает только l2
        elif solver == "liblinear":
            penalty = trial.suggest_categorical("logreg_penalty", ["l1", "l2"])  # OvR
        else:  # saga
            penalty = trial.suggest_categorical(
                "logreg_penalty", ["l1", "l2"]
            )  # multinomial/ovr
        clf = LogisticRegression(
            max_iter=2500,
            C=C,
            class_weight="balanced",
            solver=solver,
            penalty=penalty,
        )
    elif model_name == "rf":
        n_estimators = trial.suggest_int("rf_n_estimators", 100, 300, step=50)
        max_depth = trial.suggest_int("rf_max_depth", 6, 18, step=2)
        min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 10)
        bootstrap = trial.suggest_categorical("rf_bootstrap", [True, False])
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=SEED,
        )
    elif model_name == "mlp":
        hidden = trial.suggest_int("mlp_hidden", 64, 256, step=64)
        epochs = trial.suggest_int("mlp_epochs", 3, 8)
        lr = trial.suggest_float("mlp_lr", 1e-4, 5e-3, log=True)
        # обязателен dense
        steps.append(("to_dense", DenseTransformer()))
        clf = SimpleMLP(hidden_dim=hidden, epochs=epochs, lr=lr, device=TRAIN_DEVICE)
    elif model_name == "distilbert":
        pass  # уже обработан выше
    else:  # hist_gb
        lr = trial.suggest_float("hist_gb_lr", 0.01, 0.3, log=True)
        max_iter = trial.suggest_int("hist_gb_max_iter", 60, 160, step=20)
        l2 = trial.suggest_float("hist_gb_l2_regularization", 0.0, 1.0)
        min_leaf = trial.suggest_int("hist_gb_min_samples_leaf", 10, 60, step=10)
        clf = HistGradientBoostingClassifier(
            learning_rate=lr,
            max_iter=max_iter,
            l2_regularization=l2,
            min_samples_leaf=min_leaf,
            random_state=SEED,
        )
        steps.append(("to_dense", DenseTransformer()))

    steps.append(("model", clf))
    return Pipeline(steps)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


def log_confusion_matrix(y_true, y_pred, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def objective(trial: optuna.Trial, model_name: str, X_train, y_train, X_val, y_val):
    trial.set_user_attr(
        "numeric_cols", [c for c in NUMERIC_COLS if c in X_train.columns]
    )
    # Если distilbert – обучаем напрямую без ColumnTransformer: используем только текстовую колонку
    if model_name == "distilbert":
        # Текстовая модель; поддержим CV при N_FOLDS>1
        mlflow.log_param("model", model_name)
        try:
            if N_FOLDS > 1:
                from sklearn.model_selection import StratifiedKFold

                texts = X_train["reviewText"].values
                skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
                f1_scores = []
                for tr_idx, va_idx in skf.split(texts, y_train):
                    X_tr, X_va = texts[tr_idx], texts[va_idx]
                    y_tr, y_va = y_train[tr_idx], y_train[va_idx]
                    pipe = build_pipeline(trial, model_name)
                    pipe.fit(X_tr, y_tr)
                    preds_fold = pipe.predict(X_va)
                    f1_scores.append(f1_score(y_va, preds_fold, average="macro"))
                mean_f1 = float(np.mean(f1_scores))
                mlflow.log_metric("cv_f1_macro", mean_f1)
                return mean_f1
            else:
                clf_pipe = build_pipeline(trial, model_name)
                clf_pipe.fit(X_train["reviewText"], y_train)
                preds = clf_pipe.predict(X_val["reviewText"])
                metrics = compute_metrics(y_val, preds)
                mlflow.log_metrics(metrics)
                return metrics["f1_macro"]
        except ImportError as e:
            mlflow.log_param("skipped", f"missing_deps: {e}")
            return 0.0

    # KFold CV если указано через переменную окружения
    if N_FOLDS > 1:
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        f1_scores = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_va = y_train[tr_idx], y_train[val_idx]
            pipe = build_pipeline(trial, model_name)
            pipe.fit(X_tr, y_tr)
            preds_fold = pipe.predict(X_va)
            f1_scores.append(f1_score(y_va, preds_fold, average="macro"))
        mean_f1 = float(np.mean(f1_scores))
        mlflow.log_param("model", model_name)
        mlflow.log_metric("cv_f1_macro", mean_f1)
        return mean_f1
    else:
        pipe = build_pipeline(trial, model_name)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        metrics = compute_metrics(y_val, preds)
        mlflow.log_param("model", model_name)
        mlflow.log_metrics(metrics)
        return metrics["f1_macro"]


def _early_stop_callback(patience: int):
    # санитайзер: не позволяем patience < 1
    if patience is None or patience < 1:
        log.warning("patience<1 в early stop, принудительно устанавливаю в 1")
        patience = 1
    best = {"value": -1, "since": 0}

    def cb(study: optuna.Study, trial: optuna.Trial):
        if trial.value is None:
            return
        if trial.value > best["value"] + 1e-9:
            best["value"] = trial.value
            best["since"] = 0
        else:
            best["since"] += 1
        if best["since"] >= patience:
            log.info("Early stop: нет улучшений %d trial'ов", patience)
            study.stop()

    return cb


def _extract_feature_importances(
    pipeline: Pipeline, use_svd: bool
) -> List[Dict[str, float]]:
    res: List[Dict[str, float]] = []
    try:
        if "pre" not in pipeline.named_steps or "model" not in pipeline.named_steps:
            return res
        model = pipeline.named_steps["model"]
        pre: ColumnTransformer = pipeline.named_steps["pre"]
        text_pipe: Pipeline = pre.named_transformers_["text"]
        tfidf: TfidfVectorizer = text_pipe.named_steps["tfidf"]
        vocab_inv = (
            {idx: tok for tok, idx in tfidf.vocabulary_.items()}
            if hasattr(tfidf, "vocabulary_")
            else {}
        )
        text_dim = len(vocab_inv) if vocab_inv else 0
        numeric_cols = pre.transformers_[1][2]
        feature_names: List[str] = []
        if not use_svd and vocab_inv:
            feature_names.extend(
                [vocab_inv.get(i, f"tok_{i}") for i in range(text_dim)]
            )
        elif use_svd and "svd" in text_pipe.named_steps:
            svd_comp = text_pipe.named_steps["svd"].n_components
            feature_names.extend([f"svd_{i}" for i in range(svd_comp)])
        feature_names.extend(list(numeric_cols))

        if hasattr(model, "coef_"):
            coefs = np.mean(np.abs(model.coef_), axis=0)
        elif hasattr(model, "feature_importances_"):
            coefs = model.feature_importances_
        else:
            return res
        top_idx = np.argsort(coefs)[::-1][:50]
        for i in top_idx:
            if i < len(feature_names):
                res.append({"feature": feature_names[i], "importance": float(coefs[i])})
    except Exception as e:  # noqa: BLE001
        log.warning("Не удалось извлечь важности: %s", e)
    return res


def run():
    set_seeds()
    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best_model.joblib"

    if best_path.exists() and not FORCE_TRAIN:
        log.info("Модель уже существует и FORCE_TRAIN=False — пропуск")
        return

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
    log.info(
        "Размеры: train=%d, val=%d, test=%d", len(X_train), len(X_val), len(X_test)
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    start_time = time.time()

    with mlflow.start_run(run_name="classical_pipeline"):
        # общие параметры
        mlflow.log_params(
            {
                "seed": SEED,
                "numeric_cols": ",".join(
                    [c for c in NUMERIC_COLS if c in X_train.columns]
                ),
                "text_clean_stage": "spark_process",
                "cv_n_folds": N_FOLDS,
            }
        )

        # версии библиотек
        try:
            import sklearn  # type: ignore
            import pandas as _pd  # noqa: F401

            mlflow.log_params(
                {
                    "version_sklearn": sklearn.__version__,
                    "version_optuna": optuna.__version__,
                    "version_mlflow": mlflow.__version__,
                    "version_pandas": _pd.__version__,
                }
            )
        except Exception:  # noqa: BLE001
            pass

        # baseline статистики числовых признаков
        baseline_stats: Dict[str, Dict[str, float]] = {}
        for c in [c for c in NUMERIC_COLS if c in X_train.columns]:
            s = X_train[c]
            baseline_stats[c] = {"mean": float(s.mean()), "std": float(s.std() or 0.0)}
        bs_path = model_dir / "baseline_numeric_stats.json"
        with open(bs_path, "w", encoding="utf-8") as f:
            json.dump(baseline_stats, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(str(bs_path))

        # Перебираем модели и оптимизируем каждую в своей study
        per_model_results: Dict[str, Dict[str, object]] = {}
        for model_name in MODEL_CHOICES:
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
            study_name = f"{STUDY_BASE_NAME}_{model_name}_{_model_sig}"
            with mlflow.start_run(nested=True, run_name=f"model={model_name}"):
                study = optuna.create_study(
                    direction="maximize",
                    pruner=pruner,
                    storage=OPTUNA_STORAGE,
                    study_name=study_name,
                    load_if_exists=True,
                )
                # Прозрачность рестартов: логируем имя study и текущее число завершённых trial'ов
                mlflow.log_param("study_name", study_name)
                try:
                    mlflow.log_param(
                        "existing_trials",
                        len([t for t in study.trials if t.value is not None]),
                    )
                except Exception:
                    pass

                def opt_obj(trial):
                    with mlflow.start_run(nested=True):
                        return objective(
                            trial, model_name, X_train, y_train, X_val, y_val
                        )

                try:
                    study.optimize(
                        opt_obj,
                        n_trials=OPTUNA_N_TRIALS,
                        timeout=OPTUNA_TIMEOUT_SEC,
                        callbacks=[_early_stop_callback(EARLY_STOP_PATIENCE)],
                        show_progress_bar=False,
                    )
                except KeyboardInterrupt:
                    log.warning(
                        "Оптимизация прервана пользователем — используем текущий best"
                    )

                if not study.best_trial or study.best_trial.value is None:
                    log.warning(f"{model_name}: нет успешных trial'ов — пропуск")
                else:
                    per_model_results[model_name] = {
                        "best_value": study.best_value,
                        "best_params": study.best_trial.params,
                        "study_name": study_name,
                    }

        if not per_model_results:
            log.error("Нет ни одного успешного результата оптимизации — выход")
            return

        # Выбираем лучшую модель по best_value
        best_model = max(per_model_results.items(), key=lambda x: x[1]["best_value"])[0]
        best_info = per_model_results[best_model]
        log.info(
            f"Лучшая модель: {best_model} (val_f1_macro={best_info['best_value']:.4f})"
        )
        mlflow.log_param("best_model", best_model)
        mlflow.log_params({f"best_{k}": v for k, v in best_info["best_params"].items()})
        mlflow.log_metric("best_val_f1_macro", best_info["best_value"])

        # Retrain на train+val
        X_full = pd.concat([X_train, X_val], axis=0)
        y_full = np.concatenate([y_train, y_val])
        best_params = best_info["best_params"]

        class DummyTrial:
            def __init__(self, params):
                self.params = params
                self._attrs = {
                    "numeric_cols": [c for c in NUMERIC_COLS if c in X_full.columns]
                }

            def suggest_int(self, name, _low, _high, step=1):
                return self.params[name]

            def suggest_float(
                self,
                name,
                _low,
                _high,
                step=None,
                log=False,
                log_scale=False,
            ):
                # Игнорируем дополнительные параметры (step/log/log_scale),
                # так как возвращаем сохранённое значение из best_params
                return self.params[name]

            def suggest_categorical(self, name, _choices):
                return self.params[name]

            def set_user_attr(self, k, v):
                self._attrs[k] = v

            @property
            def user_attrs(self):
                return self._attrs

        if best_model == "distilbert":
            epochs = best_params.get("db_epochs", 2)
            lr = best_params.get("db_lr", 2e-5)
            max_len = best_params.get("db_max_len", 160)
            use_bi = best_params.get("db_use_bigrams", False)
            final_pipeline = DistilBertClassifier(
                epochs=epochs,
                lr=lr,
                max_len=max_len,
                device=TRAIN_DEVICE,
                use_bigrams=use_bi,
            )
            final_pipeline.fit(X_full["reviewText"], y_full)
        else:
            dummy = DummyTrial(best_params)
            final_pipeline = build_pipeline(dummy, best_model)
            final_pipeline.fit(X_full, y_full)
        joblib.dump(final_pipeline, best_path)
        mlflow.log_artifact(str(best_path))

        # Test evaluation
        if best_model == "distilbert":
            test_preds = final_pipeline.predict(X_test["reviewText"])
        else:
            test_preds = final_pipeline.predict(X_test)
        test_metrics = compute_metrics(y_test, test_preds)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # Артефакты: confusion matrix + classification report
        cm_path = model_dir / "confusion_matrix_test.png"
        log_confusion_matrix(y_test, test_preds, cm_path)
        mlflow.log_artifact(str(cm_path))
        cr_txt = classification_report(y_test, test_preds)
        cr_path = model_dir / "classification_report_test.txt"
        cr_path.write_text(cr_txt, encoding="utf-8")
        mlflow.log_artifact(str(cr_path))

        # Ошибки классификации (первые 200)
        mis_idx = np.where(test_preds != y_test)[0]
        if len(mis_idx):
            mis_samples = X_test.iloc[mis_idx].copy()
            mis_samples["true"] = y_test[mis_idx]
            mis_samples["pred"] = test_preds[mis_idx]
            mis_path = model_dir / "misclassified_samples_test.csv"
            mis_samples.head(200).to_csv(mis_path, index=False)
            mlflow.log_artifact(str(mis_path))

        duration = time.time() - start_time
        mlflow.log_metric("training_duration_sec", duration)

        meta = {
            "best_model": best_model,
            "best_params": best_params,
            "best_val_f1_macro": best_info["best_value"],
            "test_metrics": test_metrics,
            "sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
            "duration_sec": duration,
        }
        with open(model_dir / "best_model_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        log.info(f"Завершено. Лучшая модель сохранена: {best_path}")


if __name__ == "__main__":
    run()

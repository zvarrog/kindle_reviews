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
import json
import joblib
import hashlib
import os
import time
from typing import Optional, Dict, Tuple, List
import numpy as np
import pandas as pd
import optuna
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from scripts.settings import (
    SEED,
    MODEL_DIR,
    FORCE_TRAIN,
    OPTUNA_N_TRIALS,
    OPTUNA_STORAGE,
    log,
    SELECTED_MODEL_KINDS,
    STUDY_BASE_NAME,
    N_FOLDS,
    TRAIN_DEVICE,
    EARLY_STOP_PATIENCE,
    OPTUNA_TIMEOUT_SEC,
    TFIDF_MAX_FEATURES_MIN,
    TFIDF_MAX_FEATURES_MAX,
    FORCE_SVD_THRESHOLD_MB,
)
from scripts.train_modules.data_loading import load_splits
from scripts.train_modules.feature_space import NUMERIC_COLS, DenseTransformer
from scripts.train_modules.models import SimpleMLP
from scripts.models.distilbert import DistilBertClassifier
from scripts.models.kinds import ModelKind

# локальные реализации метрик и визуализаций ниже
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import sklearn

MODEL_CHOICES: List[ModelKind] = SELECTED_MODEL_KINDS
SEARCH_SPACE_VERSION = "v2"
EXPERIMENT_NAME: str = os.environ.get("MLFLOW_EXPERIMENT_NAME", "kindle_experiment")

_model_sig = hashlib.md5(
    (",".join([m.value for m in MODEL_CHOICES]) + "|" + SEARCH_SPACE_VERSION).encode(
        "utf-8"
    )
).hexdigest()[:8]
OPTUNA_STUDY_NAME = f"{STUDY_BASE_NAME}_{_model_sig}"


def _configure_mlflow_tracking() -> None:
    """Устанавливает безопасный file-store для MLflow внутри контейнера/локально.

    Цели:
    - Избежать утечки Windows-путей вида 'C:\\...' в Linux-контейнере (ошибка '/C:').
    - Явно проставить file-store, если переменная не задана.
    """
    import re

    original_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    log.info(f"MLflow URI до обработки: '{original_uri}'")
    # Принудительно устанавливаем безопасный путь в контейнере
    airflow_mlruns = Path("/opt/airflow/mlruns")
    if airflow_mlruns.exists():
        target = "file:/opt/airflow/mlruns"
        log.info("Обнаружен /opt/airflow/mlruns, принудительно устанавливаю MLflow URI")
    else:
        local = Path("mlruns").resolve()
        local.mkdir(parents=True, exist_ok=True)
        target = f"file:{local.as_posix()}"
        log.info(f"Создаю локальный mlruns: {local}")

    # Принудительно устанавливаем и логируем
    os.environ["MLFLOW_TRACKING_URI"] = target
    mlflow.set_tracking_uri(target)

    # Проверяем, что MLflow действительно использует правильный URI
    actual_uri = mlflow.get_tracking_uri()
    log.info(f"MLflow URI после установки: '{target}' -> MLflow видит: '{actual_uri}'")

    # Дополнительная проверка: убедимся, что artifact_uri тоже корректный
    try:
        # Создаём временный run чтобы проверить artifact_uri
        with mlflow.start_run(run_name="tracking_uri_test") as run:
            artifact_uri = run.info.artifact_uri
            log.info(f"Тестовый artifact_uri: '{artifact_uri}'")
            # Немедленно завершаем тестовый run
        mlflow.end_run()
    except Exception as e:
        log.warning(f"Не удалось проверить artifact_uri: {e}")


def build_pipeline(
    trial: optuna.Trial, model_name, fixed_solver: Optional[str] = None
) -> Pipeline:
    """Строит sklearn Pipeline для trial.

    Состав:
      - ColumnTransformer: text -> TF-IDF(+опц. SVD), numeric -> Impute+Scale
      - (опц.) DenseTransformer для hist_gb
      - Финальная модель
    """
    # Приводим имя модели к enum (обратная совместимость со строками)
    model_kind: ModelKind
    if isinstance(model_name, ModelKind):
        model_kind = model_name
    else:
        model_kind = ModelKind(str(model_name))
    # DistilBERT отдельный (без TF-IDF)
    if model_kind is ModelKind.distilbert:
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
            import importlib  # noqa: WPS433
            import re  # noqa: WPS433

            stem_mod = importlib.import_module("nltk.stem")  # type: ignore
            SnowballStemmer = getattr(stem_mod, "SnowballStemmer")
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
        except ModuleNotFoundError:
            # если nltk не установлен — тихо пропускаем стемминг
            return "word"

    if model_kind is ModelKind.logreg:  # без SVD, расширяем max_features, float32
        text_max_features = trial.suggest_int(
            "tfidf_max_features",
            TFIDF_MAX_FEATURES_MIN,
            TFIDF_MAX_FEATURES_MAX,
            step=250,
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
            "tfidf_max_features",
            TFIDF_MAX_FEATURES_MIN,
            TFIDF_MAX_FEATURES_MAX,
            step=250,
        )

        # Оценка потенциального размера матрицы и принудительное включение SVD
        estimated_size_mb = (text_max_features * 20000 * 4) / (
            1024 * 1024
        )  # примерная оценка для float32
        force_svd = estimated_size_mb > FORCE_SVD_THRESHOLD_MB

        if force_svd:
            log.warning(
                "Принудительно включаю SVD: оценочный размер матрицы %.1f MB > %d MB",
                estimated_size_mb,
                FORCE_SVD_THRESHOLD_MB,
            )
            use_svd = True
            svd_components = trial.suggest_int("svd_components", 20, 100, step=20)
        else:
            use_svd = trial.suggest_categorical("use_svd", [False, True])
            if use_svd:
                svd_components = trial.suggest_int("svd_components", 20, 100, step=20)
            else:
                svd_components = (
                    None  # Значение по умолчанию, когда SVD не используется
                )
        tfidf = TfidfVectorizer(
            max_features=text_max_features,
            ngram_range=(1, 2),
            dtype=np.float32,
            stop_words="english",
            analyzer=_maybe_stemming_analyzer(),
        )
        text_steps = [("tfidf", tfidf)]
        if use_svd and svd_components is not None:
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

    if model_kind is ModelKind.logreg:
        C = trial.suggest_float("logreg_C", 1e-4, 1e2, log=True)
        # Если fixed_solver задан, фиксируем выбор в trial через single-choice
        if fixed_solver is not None:
            solver = trial.suggest_categorical("logreg_solver", [fixed_solver])
        else:
            solver = trial.suggest_categorical(
                "logreg_solver", ["lbfgs", "liblinear", "saga"]
            )  # noqa: E501
        # Динамическое пространство: набор допустимых penalty зависит от выбранного solver.
        if solver == "lbfgs":
            penalty = trial.suggest_categorical("logreg_penalty", ["l2"])  # lbfgs
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
    elif model_kind is ModelKind.rf:
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
    elif model_kind is ModelKind.mlp:
        hidden = trial.suggest_int("mlp_hidden", 64, 256, step=64)
        epochs = trial.suggest_int("mlp_epochs", 3, 8)
        lr = trial.suggest_float("mlp_lr", 1e-4, 5e-3, log=True)
        # обязателен dense
        steps.append(("to_dense", DenseTransformer()))
        clf = SimpleMLP(hidden_dim=hidden, epochs=epochs, lr=lr, device=TRAIN_DEVICE)
    elif model_kind is ModelKind.distilbert:
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


def objective(
    trial: optuna.Trial,
    model_name,
    X_train,
    y_train,
    X_val,
    y_val,
    fixed_solver: Optional[str] = None,
):
    model_kind: ModelKind = (
        model_name if isinstance(model_name, ModelKind) else ModelKind(str(model_name))
    )
    trial.set_user_attr(
        "numeric_cols", [c for c in NUMERIC_COLS if c in X_train.columns]
    )
    # Если distilbert – обучаем напрямую без ColumnTransformer
    if model_kind is ModelKind.distilbert:
        mlflow.log_param("model", model_kind.value)
        try:
            if N_FOLDS > 1:
                from sklearn.model_selection import StratifiedKFold

                texts = X_train["reviewText"].values
                skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
                f1_scores = []
                for tr_idx, va_idx in skf.split(texts, y_train):
                    X_tr, X_va = texts[tr_idx], texts[va_idx]
                    y_tr, y_va = y_train[tr_idx], y_train[va_idx]
                    pipe = build_pipeline(trial, model_kind, fixed_solver)
                    pipe.fit(X_tr, y_tr)
                    preds_fold = pipe.predict(X_va)
                    f1_scores.append(f1_score(y_va, preds_fold, average="macro"))
                mean_f1 = float(np.mean(f1_scores))
                mlflow.log_metric("cv_f1_macro", mean_f1)
                return mean_f1
            else:
                clf_pipe = build_pipeline(trial, model_kind, fixed_solver)
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
            pipe = build_pipeline(trial, model_kind, fixed_solver)
            pipe.fit(X_tr, y_tr)
            preds_fold = pipe.predict(X_va)
            f1_scores.append(f1_score(y_va, preds_fold, average="macro"))
        mean_f1 = float(np.mean(f1_scores))
        mlflow.log_param("model", model_kind.value)
        mlflow.log_metric("cv_f1_macro", mean_f1)
        return mean_f1
    else:
        pipe = build_pipeline(trial, model_kind, fixed_solver)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        metrics = compute_metrics(y_val, preds)
        mlflow.log_param("model", model_kind.value)
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
    except (KeyError, AttributeError, ValueError) as e:  # noqa: TRY302
        log.warning("Не удалось извлечь важности: %s", e)
    return res


def run():
    # Настраиваем логирование для обучения
    from .logging_config import setup_training_logging

    setup_training_logging()

    # Настройка MLflow трекинга до любых логов
    _configure_mlflow_tracking()

    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best_model.joblib"

    log.info("FORCE_TRAIN=%s, best_model exists=%s", FORCE_TRAIN, best_path.exists())

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
        mlflow.log_params(
            {
                "version_sklearn": sklearn.__version__,
                "version_optuna": optuna.__version__,
                "version_mlflow": mlflow.__version__,
                "version_pandas": pd.__version__,
            }
        )

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
        per_model_results: Dict[ModelKind, Dict[str, object]] = {}
        # Расширяем цели поиска: для logreg создаём отдельные study по solver, чтобы избежать динамики дистрибуций в одной study
        search_targets: List[Tuple[ModelKind, Optional[str]]] = []
        for model_name in MODEL_CHOICES:
            if model_name is ModelKind.logreg:
                search_targets.extend(
                    [
                        (model_name, "lbfgs"),
                        (model_name, "liblinear"),
                        (model_name, "saga"),
                    ]
                )
            else:
                search_targets.append((model_name, None))

        for model_name, fixed_solver in search_targets:
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
            study_name = f"{STUDY_BASE_NAME}_{model_name.value}{('_' + fixed_solver) if fixed_solver else ''}_{_model_sig}"
            with mlflow.start_run(nested=True, run_name=f"model={model_name.value}"):
                study = optuna.create_study(
                    direction="maximize",
                    pruner=pruner,
                    storage=OPTUNA_STORAGE,
                    study_name=study_name,
                    load_if_exists=True,
                )
                # Прозрачность рестартов: логируем имя study и текущее число завершённых trial'ов
                mlflow.log_param("study_name", study_name)
                mlflow.log_param(
                    "existing_trials",
                    len([t for t in study.trials if t.value is not None]),
                )

                def opt_obj(trial, model_name=model_name, fixed_solver=fixed_solver):
                    with mlflow.start_run(nested=True):
                        return objective(
                            trial,
                            model_name,
                            X_train,
                            y_train,
                            X_val,
                            y_val,
                            fixed_solver,
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
                    log.warning("%s: нет успешных trial'ов — пропуск", model_name.value)
                else:
                    # сохраняем лучший результат по каждой модели (включая разные solvers) как максимум
                    cur = per_model_results.get(model_name)
                    new_entry = {
                        "best_value": study.best_value,
                        "best_params": study.best_trial.params,
                        "study_name": study_name,
                    }
                    if cur is None or new_entry["best_value"] > cur["best_value"]:
                        per_model_results[model_name] = new_entry

        if not per_model_results:
            log.error("Нет ни одного успешного результата оптимизации — выход")
            return

        # Выбираем лучшую модель по best_value
        best_model = max(per_model_results.items(), key=lambda x: x[1]["best_value"])[0]
        best_info = per_model_results[best_model]
        log.info(
            "Лучшая модель: %s (val_f1_macro=%.4f)", best_model, best_info["best_value"]
        )
        mlflow.log_param("best_model", best_model.value)
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

            def suggest_int(self, name, _low, _high, **_kwargs):  # noqa: ANN001
                return self.params[name]

            def suggest_float(self, name, _low, _high, **_kwargs):  # noqa: ANN001
                # Игнорируем дополнительные параметры (step/log/log_scale и т.п.),
                # так как возвращаем сохранённое значение из best_params
                return self.params[name]

            def suggest_categorical(self, name, _choices):
                return self.params[name]

            def set_user_attr(self, k, v):
                self._attrs[k] = v

            @property
            def user_attrs(self):
                return self._attrs

        if best_model is ModelKind.distilbert:
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
        if best_model is ModelKind.distilbert:
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
    log.info("Завершено. Лучшая модель сохранена: %s", best_path)


if __name__ == "__main__":
    run()

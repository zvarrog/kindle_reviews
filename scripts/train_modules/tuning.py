"""
Модуль гиперпараметров, Optuna-логики и objective для обучения моделей.
"""

import optuna
import time
from scripts.settings import OPTUNA_N_TRIALS, OPTUNA_STORAGE, log, STUDY_BASE_NAME
from scripts.train_modules.models import ModelKind

OPTUNA_TIMEOUT_SEC = 3600
SEARCH_SPACE_VERSION = "v2"


# Пример функции objective (реализация зависит от структуры pipeline)
def objective(trial, model_name, X_train, y_train, X_val, y_val, fixed_solver=None):
    """
    Objective-функция для Optuna.
    """
    # ...реализация перенесётся из train.py...
    pass


# Пример функции для запуска Optuna study
def run_optuna_study(objective_fn, model_name, X_train, y_train, X_val, y_val):
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{STUDY_BASE_NAME}_{model_name.value}",
        storage=OPTUNA_STORAGE,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective_fn(trial, model_name, X_train, y_train, X_val, y_val),
        n_trials=OPTUNA_N_TRIALS,
        timeout=OPTUNA_TIMEOUT_SEC,
    )
    return study

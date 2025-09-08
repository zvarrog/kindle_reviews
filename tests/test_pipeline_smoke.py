import pandas as pd
from scripts.train import load_splits, build_pipeline
import optuna


def test_pipeline_smoke_build_fit():
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    # Минимальный trial-подобный объект
    class T:
        def suggest_categorical(self, *a, **k):
            return "logreg"

        def suggest_float(self, *a, **k):
            # Вернем среднее допустимого диапазона если есть
            low, high = a[1], a[2]
            return (low + high) / 2

        def suggest_int(self, *a, **k):
            low, high = a[1], a[2]
            return (low + high) // 2

        def should_prune(self):
            return False

    trial = T()
    pipe = build_pipeline(trial, "logreg")
    # Отберем маленький поднабор чтобы ускорить smoke
    sample_idx = y_train.sample(min(200, len(y_train)), random_state=42).index
    pipe.fit(X_train.loc[sample_idx], y_train.loc[sample_idx])
    preds = pipe.predict(X_val.head(20))
    assert len(preds) == min(20, len(y_val))

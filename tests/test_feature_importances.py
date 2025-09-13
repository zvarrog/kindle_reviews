import sys, os

sys.path.append("..\\..")
print(os.listdir(sys.path))

from scripts.train import (
    _extract_feature_importances,
    build_pipeline,
    load_splits,
    NUMERIC_COLS,
)


class DummyTrial:
    def __init__(self):
        self.params = {
            "tfidf_max_features": 5000,
            "use_svd": False,
            "model": "logreg",
            "logreg_C": 1.0,
        }
        self._attrs = {"numeric_cols": NUMERIC_COLS}

    def suggest_int(self, name, low, high, step=1):
        return self.params.get(name, low)

    def suggest_categorical(self, name, choices):
        return self.params.get(name, choices[0])

    def suggest_float(self, name, low, high, log=False):
        return self.params.get(name, low)

    @property
    def user_attrs(self):
        return self._attrs


def test_feature_importances_non_empty(tmp_path):
    X_train, X_val, *_ = load_splits()
    trial = DummyTrial()
    pipe = build_pipeline(trial, "logreg")
    pipe.fit(
        X_train.head(200), X_train.head(200).assign(dummy=1).index % 5 + 1
    )  # synthetic y
    fi = _extract_feature_importances(pipe, use_svd=False)
    assert fi, "Expect non-empty feature importances"

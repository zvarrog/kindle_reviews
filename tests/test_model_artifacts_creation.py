from pathlib import Path
import shutil
from scripts.train import run, MODEL_DIR


def test_run_creates_artifacts(tmp_path, monkeypatch):
    # Перенаправим MODEL_DIR на временную папку
    import scripts.train as train_mod

    monkeypatch.setattr(train_mod, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(train_mod, "FORCE_TRAIN", True)
    run()
    p = Path(tmp_path)
    assert (p / "best_model.joblib").exists()
    assert (p / "best_model_meta.json").exists()

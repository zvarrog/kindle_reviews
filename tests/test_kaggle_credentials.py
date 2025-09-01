from scripts.download_kindle_reviews import has_kaggle_credentials


def test_has_credentials_via_env(monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    # set env
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "key")
    assert has_kaggle_credentials() is True


def test_has_credentials_via_file(tmp_path, monkeypatch):
    # ensure env vars not set
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    kaggle_dir = home / ".kaggle"
    kaggle_dir.mkdir()
    (kaggle_dir / "kaggle.json").write_text('{"key":"value"}')
    assert has_kaggle_credentials(home=str(home)) is True


def test_no_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    assert has_kaggle_credentials(home=str(home)) is False

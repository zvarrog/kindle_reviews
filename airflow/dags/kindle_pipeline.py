from datetime import datetime

try:  # Локальная среда без установленного Airflow (статический анализ тестов)
    from airflow import DAG  # type: ignore
    from airflow.operators.python import PythonOperator  # type: ignore
except ImportError:  # noqa: BLE001

    class _Dummy:
        def __init__(self, *_, **__):
            self.task_id = "dummy"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class DAG(_Dummy):  # type: ignore
        def __init__(self, *_, **__):
            super().__init__()

    class PythonOperator(_Dummy):  # type: ignore
        def __init__(self, *_, **__):
            super().__init__()

        def __rshift__(self, other):
            return other

        def __lshift__(self, other):
            return other


_DOC = (
    "DAG kindle_reviews_pipeline.\n"
    "- Параметры управляются через dag.params (force_* и пути).\n"
    "- Значимые настройки (e.g. OPTUNA_STORAGE) читаются из Airflow Variables/Connections.\n"
    "- Путь артефактов прокидывается через XCom между задачами."
)
__doc__ = _DOC  # используем чтобы не было 'string statement has no effect'

default_args = {
    "start_date": datetime(2025, 9, 1),
}

with DAG(
    dag_id="kindle_reviews_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description=_DOC,
    params={
        # Флаги форсирования
        "force_download": False,
        "force_process": False,
        "force_train": False,
        # Пути (можно переопределить при запуске)
        "raw_data_dir": "data/raw",
        "processed_data_dir": "data/processed",
        "model_dir": "model",
    },
) as dag:

    # Единый словарь флагов — можно переопределять при Trigger DAG (dagrun.conf)
    # Например: airflow dags trigger kindle_reviews_pipeline -c '{"force_train": true}'
    def _flag(key: str, default: str) -> str:  # простая обёртка для dagrun.conf
        return str(
            {"true": "1", "false": "0", "1": "1", "0": "0"}.get(
                str(dag.params.get(key, default)).lower(), default
            )
        )

    def _with_env(**kwargs):
        import os

        # централизованно проставляем FORCE_* для вызываемых функций
        for k, v in kwargs.items():
            os.environ[k] = str(int(bool(v)))
        # Прокидываем пути для скриптов
        os.environ["RAW_DATA_DIR"] = dag.params.get("raw_data_dir")
        os.environ["PROCESSED_DATA_DIR"] = dag.params.get("processed_data_dir")
        os.environ["MODEL_DIR"] = dag.params.get("model_dir")

    def _get_flag(context, name: str, default: bool = False) -> bool:
        try:
            dr = context.get("dag_run")
            if dr and getattr(dr, "conf", None) is not None:
                return bool(dr.conf.get(name, default))
        except Exception:
            pass
        return default

    def _task_download(**context):
        from scripts import download

        _with_env(FORCE_DOWNLOAD=_get_flag(context, "force_download", False))
        # модульный запуск
        # download.main() отсутствует — импорт модуля выполнит работу
        # поэтому вызовем защитную функцию
        # Возвращаем путь к сырому CSV (для XCom)
        from scripts.download import CSV_PATH

        return str(CSV_PATH)

    def _task_process(**context):
        from scripts import spark_process

        _with_env(FORCE_PROCESS=_get_flag(context, "force_process", False))
        # Возвращаем пути к parquet (для XCom)
        from scripts.spark_process import TRAIN_PATH, VAL_PATH, TEST_PATH

        return {"train": str(TRAIN_PATH), "val": str(VAL_PATH), "test": str(TEST_PATH)}

    def _task_train(**context):
        import os

        # Предпочитаем Variable OPTUNA_STORAGE, иначе env/дефолт
        try:
            from airflow.models import Variable  # type: ignore

            optuna_storage = Variable.get(
                "OPTUNA_STORAGE",
                default_var=os.getenv(
                    "OPTUNA_STORAGE",
                    "postgresql+psycopg2://admin:admin@postgres:5432/optuna",
                ),
            )
        except Exception:
            optuna_storage = os.getenv(
                "OPTUNA_STORAGE",
                "postgresql+psycopg2://admin:admin@postgres:5432/optuna",
            )
        os.environ["OPTUNA_STORAGE"] = optuna_storage
        _with_env(FORCE_TRAIN=_get_flag(context, "force_train", False))
        from scripts.train import run

        run()

    download = PythonOperator(
        task_id="downloading",
        python_callable=_task_download,
    )

    process = PythonOperator(
        task_id="spark_processing",
        python_callable=_task_process,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=_task_train,
    )

    download >> process >> train

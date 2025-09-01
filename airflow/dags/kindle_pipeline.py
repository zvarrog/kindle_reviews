from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime

default_args = {
    "start_date": datetime(2025, 9, 1),
}

with DAG(
    dag_id="kindle_reviews_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    download = BashOperator(
        task_id="downloading",
        bash_command="cd /opt/airflow && python /opt/airflow/scripts/download.py",
    )

    process = BashOperator(
        task_id="spark_processing",
        bash_command="cd /opt/airflow && python /opt/airflow/scripts/spark_process.py",
    )

    download >> process

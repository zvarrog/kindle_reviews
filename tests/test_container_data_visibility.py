"""
Тест для проверки видимости данных в контейнере Airflow.
Проверяет, какие файлы и папки видит контейнер в примонтированных директориях.
"""
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

def check_directory_contents(path, description):
    """Проверяет содержимое директории и выводит подробную информацию"""
    log.info(f"=== Проверка {description} ===")
    log.info(f"Путь: {path}")
    
    if not Path(path).exists():
        log.error(f"❌ Директория НЕ СУЩЕСТВУЕТ: {path}")
        return
    
    log.info(f"✅ Директория существует: {path}")
    
    try:
        items = list(Path(path).iterdir())
        if not items:
            log.warning(f"⚠️  Директория ПУСТАЯ: {path}")
            return
            
        log.info(f"📁 Найдено {len(items)} элементов:")
        for item in sorted(items):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                log.info(f"  📄 {item.name} ({size_mb:.2f} MB)")
            elif item.is_dir():
                try:
                    sub_items = len(list(item.iterdir()))
                    log.info(f"  📁 {item.name}/ ({sub_items} элементов)")
                except PermissionError:
                    log.info(f"  📁 {item.name}/ (нет доступа)")
    except Exception as e:
        log.error(f"❌ Ошибка при чтении директории: {e}")

def main():
    log.info("🔍 ДИАГНОСТИКА МОНТИРОВАНИЯ ДАННЫХ В КОНТЕЙНЕРЕ AIRFLOW")
    log.info("=" * 60)
    
    # Проверяем основные директории
    check_directory_contents("/opt/airflow", "корневая директория Airflow")
    check_directory_contents("/opt/airflow/data", "директория данных")
    check_directory_contents("/opt/airflow/data/raw", "директория сырых данных")
    check_directory_contents("/opt/airflow/data/processed", "директория обработанных данных")
    check_directory_contents("/opt/airflow/scripts", "директория скриптов")
    check_directory_contents("/opt/airflow/dags", "директория DAG")
    
    # Проверяем конкретные файлы
    log.info("=== Проверка конкретных файлов ===")
    
    files_to_check = [
        "/opt/airflow/data/raw/kindle_reviews.csv",
        "/opt/airflow/scripts/config.py",
        "/opt/airflow/scripts/download.py", 
        "/opt/airflow/scripts/spark_process.py",
        "/home/airflow/.kaggle/kaggle.json"
    ]
    
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            if path.is_file():
                size_mb = path.stat().st_size / (1024 * 1024)
                log.info(f"✅ {file_path} ({size_mb:.2f} MB)")
            else:
                log.info(f"✅ {file_path} (директория)")
        else:
            log.error(f"❌ НЕ НАЙДЕН: {file_path}")
    
    # Проверяем переменные окружения
    log.info("=== Переменные окружения ===")
    env_vars = ["JAVA_HOME", "PYTHONPATH", "AIRFLOW__CORE__DAGS_FOLDER"]
    for var in env_vars:
        value = os.environ.get(var, "НЕ УСТАНОВЛЕНА")
        log.info(f"{var} = {value}")
    
    log.info("=" * 60)
    log.info("🏁 ДИАГНОСТИКА ЗАВЕРШЕНА")

if __name__ == "__main__":
    main()

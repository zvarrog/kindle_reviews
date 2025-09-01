"""
Тест для проверки локальных данных на хост-машине.
Проверяет, какие файлы есть в локальных папках перед монтированием в контейнер.
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

def check_local_directory(path, description):
    """Проверяет содержимое локальной директории"""
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
    log.info("🔍 ДИАГНОСТИКА ЛОКАЛЬНЫХ ДАННЫХ НА ХОСТ-МАШИНЕ")
    log.info("=" * 60)
    
    # Получаем путь к проекту (родительская папка от tests)
    project_root = Path(__file__).parent.parent
    log.info(f"Корень проекта: {project_root.absolute()}")
    
    # Проверяем локальные директории
    check_local_directory(project_root / "data", "директория данных")
    check_local_directory(project_root / "data" / "raw", "директория сырых данных")
    check_local_directory(project_root / "data" / "processed", "директория обработанных данных")
    check_local_directory(project_root / "scripts", "директория скриптов")
    check_local_directory(project_root / "airflow" / "dags", "директория DAG")
    
    # Проверяем конкретные файлы
    log.info("=== Проверка конкретных файлов ===")
    
    files_to_check = [
        project_root / "data" / "raw" / "kindle_reviews.csv",
        project_root / "scripts" / "config.py",
        project_root / "scripts" / "download.py",
        project_root / "scripts" / "spark_process.py",
        project_root / "docker-compose.yml",
        project_root / "requirements.txt"
    ]
    
    for file_path in files_to_check:
        if file_path.exists():
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                log.info(f"✅ {file_path.name} ({size_mb:.2f} MB)")
            else:
                log.info(f"✅ {file_path.name} (директория)")
        else:
            log.error(f"❌ НЕ НАЙДЕН: {file_path}")
    
    # Проверяем Kaggle credentials
    import os
    kaggle_path = Path(os.path.expanduser("~")) / ".kaggle" / "kaggle.json"
    log.info(f"=== Kaggle credentials ===")
    if kaggle_path.exists():
        log.info(f"✅ Kaggle credentials найдены: {kaggle_path}")
    else:
        log.error(f"❌ Kaggle credentials НЕ НАЙДЕНЫ: {kaggle_path}")
    
    log.info("=" * 60)
    log.info("🏁 ДИАГНОСТИКА ЗАВЕРШЕНА")

if __name__ == "__main__":
    main()

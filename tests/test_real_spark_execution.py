"""
Тест реального выполнения скрипта spark_process.py в контейнере.
Проверяет, действительно ли скрипт пропускает обработку.
"""
import subprocess
import sys
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

def test_real_spark_processing():
    """Реальный тест выполнения spark_process.py"""
    log.info("🚀 РЕАЛЬНЫЙ ТЕСТ ВЫПОЛНЕНИЯ SPARK_PROCESS.PY")
    log.info("=" * 60)
    
    start_time = datetime.now()
    log.info(f"⏰ Начало: {start_time}")
    
    try:
        # Запускаем spark_process.py и захватываем вывод
        result = subprocess.run([
            sys.executable, '/opt/airflow/scripts/spark_process.py'
        ], capture_output=True, text=True, timeout=300)  # 5 минут таймаут
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        log.info(f"⏰ Окончание: {end_time}")
        log.info(f"⏱️  Длительность: {duration:.2f} секунд")
        log.info(f"📤 Код возврата: {result.returncode}")
        
        # Анализируем вывод
        log.info("=" * 30 + " STDOUT " + "=" * 30)
        if result.stdout:
            stdout_lines = result.stdout.strip().split('\\n')
            for i, line in enumerate(stdout_lines[:50]):  # Показываем первые 50 строк
                log.info(f"  {i+1:2d}: {line}")
            if len(stdout_lines) > 50:
                log.info(f"  ... и еще {len(stdout_lines) - 50} строк")
        else:
            log.info("  (пустой)")
        
        log.info("=" * 30 + " STDERR " + "=" * 30)
        if result.stderr:
            stderr_lines = result.stderr.strip().split('\\n')
            for i, line in enumerate(stderr_lines[:20]):  # Показываем первые 20 строк ошибок
                log.error(f"  {i+1:2d}: {line}")
            if len(stderr_lines) > 20:
                log.error(f"  ... и еще {len(stderr_lines) - 20} строк ошибок")
        else:
            log.info("  (нет ошибок)")
        
        # Анализ результата
        log.info("=" * 60)
        if result.returncode == 0:
            log.info("✅ СКРИПТ ЗАВЕРШИЛСЯ УСПЕШНО")
            
            # Проверяем признаки пропуска обработки
            output_text = (result.stdout + result.stderr).lower()
            if any(keyword in output_text for keyword in [
                'уже существуют', 'already exist', 'пропуск', 'skip'
            ]):
                log.info("✅ ОБРАБОТКА БЫЛА ПРОПУЩЕНА (как и ожидалось)")
            elif duration < 10:
                log.info("✅ БЫСТРОЕ ВЫПОЛНЕНИЕ - вероятно, обработка была пропущена")
            else:
                log.warning("⚠️  ДОЛГОЕ ВЫПОЛНЕНИЕ - возможно, данные обрабатывались заново")
                
        else:
            log.error(f"❌ СКРИПТ ЗАВЕРШИЛСЯ С ОШИБКОЙ (код {result.returncode})")
            
    except subprocess.TimeoutExpired:
        log.error("❌ ТАЙМАУТ: Скрипт выполнялся слишком долго (>5 минут)")
        log.error("   Это может указывать на то, что данные обрабатываются заново")
    except Exception as e:
        log.error(f"❌ ОШИБКА ВЫПОЛНЕНИЯ: {e}")
    
    log.info("🏁 ТЕСТ ЗАВЕРШЕН")

if __name__ == "__main__":
    test_real_spark_processing()

-- Создание отдельной БД для метаданных Airflow (разделяем с Optuna)
DO
$$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_database WHERE datname = 'airflow_meta'
   ) THEN
      PERFORM dblink_exec('dbname=' || current_database(), 'CREATE DATABASE airflow_meta');
   END IF;
END
$$;

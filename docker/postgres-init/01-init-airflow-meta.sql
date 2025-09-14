-- Создание отдельной БД для метаданных Airflow (упрощенная версия без dblink)
-- Этот скрипт создаст базу данных только если её еще нет

-- Включаем расширение dblink если доступно, иначе используем простое создание
CREATE EXTENSION IF NOT EXISTS dblink;

-- Создаем базу данных airflow_meta если её нет
SELECT 'CREATE DATABASE airflow_meta'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow_meta')\gexec

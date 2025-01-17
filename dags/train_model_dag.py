import yaml
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
from src.data.make_dataset import main as generate_data
from src.features.build_features import build_features
from src.models.train_model import train_model

def load_config(config_path="/opt/airflow/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# Задаём конфигурацию
config = load_config()

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'depends_on_past': False
}

with DAG(
    dag_id='train_model_dag',
    default_args=default_args,
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    # Задача для генерации данных
    generate_data_task = PythonOperator(
        task_id='generate_data',
        python_callable=generate_data
    )

    # Задача для построения признаков
    def build_features_task():
        input_path = config['paths']['processed_data']
        output_path = config['paths']['features_data']
        build_features(input_filepath=input_path, output_filepath=output_path)

    feature_task = PythonOperator(
        task_id='build_features',
        python_callable=build_features_task
    )

    # Задача для обучения модели
    def train_model_task():
        train_model()

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model_task
    )

    # Определяем порядок выполнения задач
    generate_data_task >> feature_task >> train_task
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime, timedelta
import os
import pandas as pd
import pickle

from dags.train_model_dag import load_config

config = load_config()
DATA_PATH = '/opt/data'
MODEL_PATH = '/opt/airflow/models/model.pkl'
PREDICTIONS_PATH = '/opt/data/predictions.csv'
NEW_DATA_FILE = os.path.join(DATA_PATH, 'new_data.csv')

default_args = {
    'owner': 'airflow',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
        'inference_dag',
        default_args=default_args,
        description='DAG for batch inference',
        schedule_interval=None,
        start_date=datetime(2024, 11, 10),
        catchup=False,
) as dag:
    def load_model():
        with open(MODEL_PATH, 'rb') as model_file:
            model = pickle.load(model_file)
        return model


    def perform_inference():
        model = load_model()

        # Загружаем данные
        data = pd.read_csv(NEW_DATA_FILE)
        X = data.drop(columns='label', errors='ignore')

        predictions = model.predict(X)

        output_df = pd.DataFrame({'predictions': predictions})
        output_df.to_csv(PREDICTIONS_PATH, index=False)
        print(f"Predictions saved to {PREDICTIONS_PATH}")

        os.remove(NEW_DATA_FILE)
        print(f"Deleted {NEW_DATA_FILE} after inference")


    wait_for_file = FileSensor(
        task_id='wait_for_new_data_file',
        filepath=NEW_DATA_FILE,
        fs_conn_id='fs_default',
        poke_interval=30,  # Проверяет наличие файла каждые 30 секунд
        timeout=600  # Таймаут через 10 минут
    )

    inference_task = PythonOperator(
        task_id='perform_inference',
        python_callable=perform_inference
    )

    wait_for_file >> inference_task
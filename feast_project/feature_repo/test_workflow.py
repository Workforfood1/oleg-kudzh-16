import pickle
import subprocess
from datetime import datetime

import pandas as pd

from feast import FeatureStore
from feast.data_source import PushMode

from train_model import train_model, save_model


def run_demo():
    store = FeatureStore(repo_path=".")
    print("\n--- Fetch training data ---")
    training_data = fetch_training_data(store)

    print("\n--- Train model ---")
    model = train_model(training_data)
    save_model(model, filepath="model.pkl")

    print("\n--- Test model ---")
    test_model(filepath="model.pkl")

def fetch_training_data(store: FeatureStore):
    entity_df = pd.DataFrame.from_dict(
        {
            "driver_id": [1001, 1002, 1003],
            "event_timestamp": [
                datetime(2021, 4, 12, 10, 59, 42),
                datetime(2021, 4, 12, 8, 12, 10),
                datetime(2021, 4, 12, 16, 40, 26),
            ],
        }
    )
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "driver_hourly_stats:conv_rate",
            "driver_hourly_stats:acc_rate",
            "driver_hourly_stats:avg_daily_trips",
        ],
    ).to_df()
    print(training_df.head())
    return training_df


def fetch_historical_features_entity_df(store: FeatureStore, for_batch_scoring: bool):
    # Note: see https://docs.feast.dev/getting-started/concepts/feature-retrieval for more details on how to retrieve
    # for all entities in the offline store instead
    entity_df = pd.DataFrame.from_dict(
        {
            # entity's join key -> entity values
            "driver_id": [1001, 1002, 1003],
            # "event_timestamp" (reserved key) -> timestamps
            "event_timestamp": [
                datetime(2021, 4, 12, 10, 59, 42),
                datetime(2021, 4, 12, 8, 12, 10),
                datetime(2021, 4, 12, 16, 40, 26),
            ],
            # (optional) label name -> label values. Feast does not process these
            "label_driver_reported_satisfaction": [1, 5, 3],
            # values we're using for an on-demand transformation
            "val_to_add": [1, 2, 3],
            "val_to_add_2": [10, 20, 30],
        }
    )
    # For batch scoring, we want the latest timestamps
    if for_batch_scoring:
        entity_df["event_timestamp"] = pd.to_datetime("now", utc=True)

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "driver_hourly_stats:conv_rate",
            "driver_hourly_stats:acc_rate",
            "driver_hourly_stats:avg_daily_trips",
            "transformed_conv_rate:conv_rate_plus_val1",
            "transformed_conv_rate:conv_rate_plus_val2",
        ],
    ).to_df()
    print(training_df.head())


def fetch_online_features(store, source: str = ""):
    entity_rows = [
        # {join_key: entity_value}
        {
            "driver_id": 1001,
            "val_to_add": 1000,
            "val_to_add_2": 2000,
        },
        {
            "driver_id": 1002,
            "val_to_add": 1001,
            "val_to_add_2": 2002,
        },
    ]
    if source == "feature_service":
        features_to_fetch = store.get_feature_service("driver_activity_v1")
    elif source == "push":
        features_to_fetch = store.get_feature_service("driver_activity_v3")
    else:
        features_to_fetch = [
            "driver_hourly_stats:acc_rate",
            "transformed_conv_rate:conv_rate_plus_val1",
            "transformed_conv_rate:conv_rate_plus_val2",
        ]
    returned_features = store.get_online_features(
        features=features_to_fetch,
        entity_rows=entity_rows,
    ).to_dict()
    for key, value in sorted(returned_features.items()):
        print(key, " : ", value)

def test_model(filepath="model.pkl"):
    import pandas as pd

    # Загружаем модель
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")

    # Загружаем список признаков
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    # Создаём тестовые данные
    test_data = pd.DataFrame.from_dict(
        {
            "conv_rate": [0.5, 0.7, 0.2],
            "acc_rate": [0.9, 0.8, 0.95],
            "avg_daily_trips": [30, 40, 50],
        }
    )

    # Убедимся, что тестовые данные соответствуют признакам
    X_test = test_data[feature_columns]

    # Выполняем предсказание
    predictions = model.predict(X_test)
    print(f"Predictions: {predictions}")




if __name__ == "__main__":
    run_demo()
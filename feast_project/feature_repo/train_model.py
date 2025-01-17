def train_model(training_data):
    from sklearn.ensemble import RandomForestRegressor

    # Отделяем признаки и метки
    X = training_data[["conv_rate", "acc_rate"]]
    y = training_data["avg_daily_trips"]

    # Обучаем модель
    model = RandomForestRegressor()
    model.fit(X, y)

    # Сохраняем список признаков
    with open("feature_columns.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)

    # Возвращаем обученную модель
    return model


import pickle

def save_model(model, filepath="model.pkl"):
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")
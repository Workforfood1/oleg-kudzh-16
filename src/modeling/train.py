import pickle
import json
from loguru import logger
import pandas as pd
import typer
import sklearn
from src.params.pipe_params import read_pipeline_params


app = typer.Typer()


@app.command()
def main(params_path: str):
    params = read_pipeline_params(params_path)
    train = pd.read_csv(params.data_params.train_data_path)
    X_train = train.drop("target", axis=1)
    y_train = train["target"].values.reshape(-1, 1)

    test = pd.read_csv(params.data_params.test_data_path)
    X_test = test.drop("target", axis=1)
    y_test = test["target"].values.reshape(-1, 1)



    with open(params.train_params.model_path, 'rb') as model_file:
        model = pickle.load(model_file)


    model.fit(X_train, y_train)
    logger.info(f"Model trained {model}")

    y_test_pred = model.predict_proba(X_test)[:, 1]
    roc_auc = sklearn.metrics.roc_auc_score(y_test, y_test_pred)
    # conf_matrix = sklearn.metrics.confusion_matrix(y_test, y_test_pred)
    logger.info(f"Got ROC-AUC {roc_auc:.3f}")
    # logger.info(f"Got Confusion Matrix {conf_matrix}")

    metrics = {"roc-auc": roc_auc}#, "conf-matrix": conf_matrix}

    with open(params.train_params.model_path, 'wb') as fin:
        pickle.dump(model, fin)
    logger.info(f"Saved model to path {params.train_params.model_path}")

    with open(params.train_params.metrics_path, 'w') as fin:
        json.dump(metrics, fin)
    logger.info(f"Saved metrics to path {params.train_params.metrics_path}")


if __name__ == "__main__":
    app()

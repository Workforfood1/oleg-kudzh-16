import pickle
import typer
from sklearn.linear_model import LogisticRegression
from loguru import logger
from src.params.pipe_params import read_pipeline_params


app = typer.Typer()


@app.command()
def main(params_path: str):
    params = read_pipeline_params(params_path)
    model = LogisticRegression(max_iter=params.train_params.n_estimators,
                               random_state=params.random_state)

    with open(params.train_params.model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    logger.info(f"Regression created in {params.train_params.model_path}")


if __name__ == "__main__":
    app()

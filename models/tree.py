import pickle
import typer
from sklearn.tree import DecisionTreeClassifier
from loguru import logger
from src.params.pipe_params import read_pipeline_params


app = typer.Typer()


@app.command()
def main(params_path: str):
    params = read_pipeline_params(params_path)
    tree = DecisionTreeClassifier(max_depth=params.train_params.n_estimators)

    with open(params.train_params.model_path, 'wb') as model_file:
        pickle.dump(tree, model_file)
    logger.info(f"Tree created in {params.train_params.model_path}")


if __name__ == "__main__":
    app()

import typer
from loguru import logger
from src.params.pipe_params import read_pipeline_params
import src.dataset
import src.modeling.train
from models import decision_tree, logistic_regression, random_forest

app = typer.Typer()


@app.command()
def main(params_path: str):
    params = read_pipeline_params(params_path)

    try:
        src.dataset.main(params_path)
    except Exception as e:
        logger.error(e)
    else:
        logger.info("Dataset successfully created")

    try:
        match params.train_params.model_path:
            case "./models/tree.pkl":
                decision_tree.main(params_path)
            case "./models/logistic_regression.pkl":
                logistic_regression.main(params_path)
            case "./models/random_forest.pkl":
                random_forest.main(params_path)
    except Exception as e:
        logger.error(e)
    else:
        logger.info("Neural Network created successfully")

    try:
        src.modeling.train.main(params_path)
    except Exception as e:
        logger.error(e)
    else:
        logger.info("Neural network successfully trained")

    return 0

if __name__ == "__main__":
    app()

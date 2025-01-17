import os
import pytest
from sympy import andre

import src.dataset
from src.params.data_params import DataParams
from src.params.pipe_params import read_pipeline_params
import src.modeling.train
import src.pipeline


def test_dataset(config_path: str):
    config = read_pipeline_params(config_path)
    src.dataset.main(config_path)
    assert os.path.exists(config.data_params.test_data_path) is True

def test_training(config_path: str):
    config = read_pipeline_params(config_path)
    src.modeling.train.main(config_path)
    assert os.path.exists(config.train_params.metrics_path) is True

def test_pipeline(config_path: str):
    config = read_pipeline_params(config_path)
    src.pipeline.main(config_path)
    assert (os.path.exists(config.train_params.metrics_path) is True and
            os.path.exists(config.data_params.test_data_path) is True)

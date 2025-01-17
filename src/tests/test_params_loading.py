import pytest
import os
from src.params.pipe_params import read_pipeline_params


def test_load_dataset(config_path: str):
    config = read_pipeline_params(config_path)
    assert config.train_params.n_estimators == 500

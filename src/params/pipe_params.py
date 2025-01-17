from dataclasses import dataclass
import yaml
from marshmallow_dataclass import class_schema
from src.params.train_params import TrainParams
from src.params.data_params import DataParams


@dataclass()
class PipelineParams:
    train_params: TrainParams
    data_params: DataParams
    random_state: int


PipelineParamsSchema = class_schema(PipelineParams)


def read_pipeline_params(path: str) -> PipelineParams:
    with open(path, 'r') as f:
        schema = PipelineParamsSchema()
        return schema.load(yaml.safe_load(f))

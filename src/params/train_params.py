from dataclasses import dataclass, field
from marshmallow import validate


@dataclass()
class TrainParams:
    model_path: str
    metrics_path: str
    n_estimators: int = field(default=50, metadata={"validate":validate.Range(min=0)})

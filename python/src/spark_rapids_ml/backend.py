from typing import Any, Union, Dict

import pandas as pd
from pyspark.ml.param import Params
from pyspark.sql import DataFrame
from pyspark_project.backend import Backend

from spark_rapids_ml.core import _CumlEstimator
from spark_rapids_ml.registry import lookup_estimator

from .classification import LogisticRegression, LogisticRegressionModel

class SparkRapidsMLBackend(Backend):
    def __init__(self):
        super().__init__()
        # Mapping pyspark estimators to the torch estimators.
        self._estimators: Dict[str, _CumlEstimator] = dict()

    def _get_torch_estimator(self, pyspark_estimator: Params) -> _CumlEstimator:
        """Get the torch estimator for a given pyspark estimator."""
        torch_estimator_class = lookup_estimator(pyspark_estimator)
        name = torch_estimator_class.__name__
        if name not in self._estimators:
            self._estimators[name] = torch_estimator_class()
        return self._estimators[name]

    def supported_parameters(self, estimator: Params) -> Params:
        return self._get_torch_estimator(estimator)

    def fit(self, estimator: Params, dataset: Union[DataFrame, pd.DataFrame]) -> Any:
        torch_estimator = self._get_torch_estimator(estimator)
        return torch_estimator.fit(estimator, dataset)

    def transform(self, model: Params, dataset: Union[DataFrame, pd.DataFrame]) -> Any:
        torch_estimator = self._get_torch_estimator(model)
        return torch_estimator.transform(model, dataset)

    def get_backend_model_filename(self, model: Params) -> str:
        pass

    def save_backend_model(self, model: Params, path: str) -> None:
        pass

    def load_backend_model(self, model: Params, path: str) -> Any:
        pass
from typing import Type, Dict, Optional

from pyspark.ml.param import Params

from spark_rapids_ml.core import _CumlEstimator

_ESTIMATORS: Dict[str, Type[_CumlEstimator]] = dict()


def register_estimator(pyspark_estimator: str, pyspark_model: Optional[str] = None):
    def wrapper(estimator: Type[_CumlEstimator]):
        assert pyspark_estimator is not None
        assert pyspark_estimator not in _ESTIMATORS, f"duplicate name: {pyspark_estimator}"

        _ESTIMATORS[pyspark_estimator] = estimator

        if pyspark_model is not None:
            assert pyspark_model not in _ESTIMATORS, f"duplicate name: {pyspark_model}"
            _ESTIMATORS[pyspark_model] = estimator

    return wrapper


def lookup_estimator(estimator: Params) -> Type[_CumlEstimator]:
    name = estimator.__class__.__name__
    if name not in _ESTIMATORS:
        raise RuntimeError(f"no such estimator: {name}")
    else:
        return _ESTIMATORS[name]

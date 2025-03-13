from pyspark.ml.classification import (LogisticRegression,
                                       LogisticRegressionModel)
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

from typing import Dict, Any

from pyspark import keyword_only
from pyspark.ml.wrapper import JavaEstimator, JavaModel


class DummyEstimator(JavaEstimator["DummyModel"]):
    _input_kwargs: Dict[str, Any]

    @keyword_only
    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__()
        self._java_obj = self._new_java_obj(
            "com.example.ml.DummyEstimator", self.uid
        )
        self._set(**self._input_kwargs)

    def _create_model(self, java_model: "JavaObject") -> "DummyModel":
        return DummyModel(java_model)


class DummyModel(JavaModel):
    pass

spark = (SparkSession.builder.remote("sc://localhost")
         .getOrCreate())


df = spark.createDataFrame([
        (Vectors.dense([1.0, 2.0]), 1),
        (Vectors.dense([2.0, -1.0]), 1),
        (Vectors.dense([-3.0, -2.0]), 0),
        (Vectors.dense([-1.0, -2.0]), 0),
        ], schema=['features', 'label'])
est = DummyEstimator()
model = est.fit(df)
model.transform(df).show()

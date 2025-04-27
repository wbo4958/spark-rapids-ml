from typing import Union, Any

from pyspark.ml import Estimator
from pyspark.ml.param import Params
from pyspark.ml.param.shared import HasParallelism, HasCollectSubModels
from pyspark.ml.tuning import _CrossValidatorParams
from pyspark.sql import DataFrame
from pyspark.sql.connect import proto as proto
from pyspark.sql.connect.plan import LogicalPlan
from pyspark.ml.tuning import CrossValidator as SparkCrossValidator

import spark_rapids_ml.proto as rapids_pb


class CrossValidatorPlan(LogicalPlan):

    def __init__(self, cv_relation: rapids_pb.CrossValidatorRelation):
        super().__init__(None)
        self._cv_relation = cv_relation

    def plan(self, session: "SparkConnectClient") -> proto.Relation:
        plan = self._create_proto_relation()
        plan.extension.Pack(self._cv_relation)
        return plan


def extractParams(instance: "Params") -> str:
    params = {}
    # TODO: support vector/matrix
    for k, v in instance._paramMap.items():
        if instance.isSet(k) and isinstance(v, int | float | str | bool):
            params[k.name] = v

    import json
    return json.dumps(params)


class CrossValidator(SparkCrossValidator):

    def _fit(self, dataset: DataFrame) -> Any:
        estimator = self.getEstimator()
        evaluator = self.getEvaluator()
        cv_rel = rapids_pb.CrossValidatorRelation(
            estimator=rapids_pb.MlOperator(
                name=type(estimator).__name__,
                uid=estimator.uid,
                type=rapids_pb.MlOperator.OperatorType.OPERATOR_TYPE_ESTIMATOR,
                params=extractParams(estimator),
            ),
            evaluator=rapids_pb.MlOperator(
                name=type(evaluator).__name__,
                uid=evaluator.uid,
                type=rapids_pb.MlOperator.OperatorType.OPERATOR_TYPE_EVALUATOR,
                params=extractParams(evaluator),
            ),
            dataset=dataset._plan.to_proto(dataset.sparkSession.client).SerializeToString(),
            params=extractParams(self),
        )
        from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame
        df = ConnectDataFrame(CrossValidatorPlan(cv_relation=cv_rel), dataset.sparkSession)
        x = df.collect()
        print(f"------------------------- x is {x}")

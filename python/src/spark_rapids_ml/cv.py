from typing import Union, Any

from pyspark.ml import Estimator
from pyspark.ml.param.shared import HasParallelism, HasCollectSubModels
from pyspark.ml.tuning import _CrossValidatorParams
from pyspark.sql import DataFrame
from pyspark.sql.connect import proto as proto
from pyspark.sql.connect.plan import LogicalPlan

import spark_rapids_ml.proto as rapids_pb


class CrossValidatorPlan(LogicalPlan):

    def __init__(self, cv_relation: rapids_pb.CrossValidatorRelation):
        super().__init__(None)
        self._cv_relation = cv_relation

    def plan(self, session: "SparkConnectClient") -> proto.Relation:
        plan = self._create_proto_relation()
        plan.extension.Pack(self._cv_relation)
        return plan


class CrossValidator(
    Estimator,
    _CrossValidatorParams,
    HasParallelism,
    HasCollectSubModels,
):

    def _fit(self, dataset: DataFrame) -> Any:
        import pyspark.sql.connect.proto as pb2
        from pyspark.ml.connect.serialize import serialize_ml_params, deserialize

        cv_rel = rapids_pb.CrossValidatorRelation(
            estimator=rapids_pb.MlOperator(
                name="LogisticRegression",
                uid=self.uid,
                type=rapids_pb.MlOperator.OperatorType.OPERATOR_TYPE_ESTIMATOR,
            ),
            evaluator=rapids_pb.MlOperator(
                name="MultiClassEvaluator",
                uid=self.uid,
                type=rapids_pb.MlOperator.OperatorType.OPERATOR_TYPE_EVALUATOR,
            ),
            dataset = dataset._plan.to_proto(dataset.sparkSession.client).SerializeToString()
        )
        from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame
        df = ConnectDataFrame(CrossValidatorPlan(cv_relation=cv_rel), dataset.sparkSession)
        x = df.collect()
        print(f"------------------------- x is {x}")
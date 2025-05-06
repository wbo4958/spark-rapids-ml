import json

from pyspark.ml.param import Params
from pyspark.sql import DataFrame
from pyspark.sql.connect import proto as proto
from pyspark.sql.connect.plan import LogicalPlan
from pyspark.ml.tuning import CrossValidator as SparkCrossValidator, CrossValidatorModel

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

    def _fit(self, dataset: DataFrame) -> "CrossValidatorModel":
        estimator = self.getEstimator()
        evaluator = self.getEvaluator()
        est_param_list = []
        for param_group in self.getEstimatorParamMaps():
            est_param_items = []
            for p, v in param_group.items():
                tmp_map = {"parent": p.parent, "name": p.name, "value": v}
                est_param_items.append(tmp_map)
            est_param_list.append(est_param_items)
        est_param_map_json = json.dumps(est_param_list)

        estimator_name = type(estimator).__name__
        cv_rel = rapids_pb.CrossValidatorRelation(
            uid=self.uid,
            estimator=rapids_pb.MlOperator(
                name=estimator_name,
                uid=estimator.uid,
                type=rapids_pb.MlOperator.OperatorType.OPERATOR_TYPE_ESTIMATOR,
                params=extractParams(estimator),
            ),
            estimator_param_maps=est_param_map_json,
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
        row = df.collect()

        best_model = None
        model_id = row[0].best_model_id
        # TODO support other estimators
        if estimator_name == "LogisticRegression":
            from pyspark.ml.classification import LogisticRegressionModel
            best_model = LogisticRegressionModel(model_id)

        return CrossValidatorModel(best_model)

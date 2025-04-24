package com.nvidia.rapids.ml

import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.rapids.{Fit, PythonEstimatorRunner, RapidsLogisticRegressionModel, RapidsUtils, TrainedModel}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

class RapidsCrossValidator(override val uid: String) extends CrossValidator with RapidsEstimator {

  def this() = this(Identifiable.randomUID("cv"))

  override def fit(dataset: Dataset[_]): CrossValidatorModel = {
    val trainedModel = trainOnPython(dataset)

    val cpuModel = copyValues(trainedModel.model.asInstanceOf[LogisticRegressionModel])
    val isMultinomial = cpuModel.numClasses != 2
    val rlr = new RapidsLogisticRegressionModel(uid, cpuModel, trainedModel.modelAttributes, isMultinomial)
    rlr.setFeaturesCol("test_feature")
    RapidsUtils.createCrossValidatorModel(this.uid, rlr)
  }

  /**
   * The estimator name
   *
   * @return
   */
  override def name: String = "CrossValidator"

  override def trainOnPython(dataset: Dataset[_]): TrainedModel = {
    logger.info(s"Training $name ...")

    def getName(name: String): String = {
      Utils.transform(name).getOrElse(name)
    }

    val estimatorName = getName(getEstimator.getClass.getName)
    // TODO estimator could be a PipeLine which contains multiple stages.
    val cvParams = RapidsUtils.getJson(Map(
      "estimator" -> RapidsUtils.getUserDefinedParams(getEstimator,
        extra = Map(
          "estimator_name" -> estimatorName,
          "uid" -> getEstimator.uid)),
      "evaluator" -> RapidsUtils.getUserDefinedParams(getEvaluator,
        extra = Map(
          "evaluator_name" -> getName(getEvaluator.getClass.getName),
          "uid" -> getEvaluator.uid)),
      "estimatorParaMaps" -> RapidsUtils.getEstimatorParamMapsJson(getEstimatorParamMaps),
      "cv" -> RapidsUtils.getUserDefinedParams(this,
        List("estimator", "evaluator", "estimatorParamMaps"))
    ))
    val runner = new PythonEstimatorRunner(
      Fit(name, cvParams),
      dataset.toDF)

    val trainedModel = Arm.withResource(runner) { _ =>
      runner.runInPython(useDaemon = false)
    }

    logger.info(s"Finished $name training.")
    trainedModel
  }
}

object RapidsCrossValidator {


}

package org.apache.spark.ml.rapids

import com.nvidia.rapids.ml.RapidsEstimator
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, Dataset}

class RapidsLogisticRegressionModel(val coef: String,
                                    val intercepts: String,
                                    val numClass: String,
                                    val nCols: Int,
                                    val dtype: String,
                                    val nIters: Int,
                                    val objective: String)
  extends LogisticRegressionModel(uid = "asd", coefficients = Vectors.dense(Array(0.1, 0.2)), intercept = 0.3)
  with RapidsEstimator {

  override def copy(extra: ParamMap): RapidsLogisticRegressionModel = {
    val newModel = copyValues(new RapidsLogisticRegressionModel(coef, intercepts, numClass,
      nCols, dtype, nIters, objective), extra)
    newModel.setSummary(trainingSummary).setParent(parent)
    newModel
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    println("in RapidsLogisticRegressionModel transform")
    val params = RapidsUtils.getUserDefinedParams(this)

    val runner = new PythonRunnerModel(
      Transform(estimatorName, params, this),
      dataset.toDF)

    withResource(runner) { _ =>
      runner.runInPython(useDaemon = false)
    }

    dataset.toDF()

  }

  /**
   * The estimator name
   *
   * @return
   */
  override def estimatorName: String = "LogisticRegressionModel"
}

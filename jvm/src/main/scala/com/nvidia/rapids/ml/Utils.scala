package com.nvidia.rapids.ml

object Utils {

  def transform(name: String): Option[String] = {
    name match {
      case "org.apache.spark.ml.classification.LogisticRegression" =>
        Some("com.nvidia.rapids.ml.RapidsLogisticRegression")
      case "org.apache.spark.ml.classification.LogisticRegressionModel" =>
        Some("org.apache.spark.ml.rapids.RapidsLogisticRegressionModel")
      case _ => None
    }
  }

}

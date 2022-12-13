package com.nvidia.spark.ml


import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.{Estimator, MLDriverPlugin, Model}
import org.apache.spark.sql.Dataset

import com.nvidia.spark.ml.feature.{PCA => CuPCA}

class CumlDriverPlugin extends MLDriverPlugin {

  override def fit(estimator: Estimator[_], dataset: Dataset[_]): Model[_] = {
    estimator match {
      case pca: PCA => new CuPCA(pca.uid)
        .setInputCol(pca.getInputCol)
        .setOutputCol(pca.getOutputCol)
        .setK(pca.getK)
        .fit(dataset)
      case x => throw new UnsupportedOperationException("Not implemented yet")
    }
  }
}


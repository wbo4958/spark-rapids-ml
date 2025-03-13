/**
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.ml

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

class DummyModel(override val uid: String) extends Model[DummyModel] {

  def this() = this("abcd")

  override def copy(extra: ParamMap): DummyModel = this

  override def transform(dataset: Dataset[_]): DataFrame = {
    println("--------------------------------------------------------- DummyModel ")
    dataset.toDF()
  }

  override def transformSchema(schema: StructType): StructType = schema
}


/**
 * RapidsLogisticRegression is a JVM wrapper of LogisticRegression in spark-rapids-ml python package.
 *
 * The training process is going to launch a Python Process where to run spark-rapids-ml
 * LogisticRegression and return the corresponding model
 *
 * @param uid unique ID of the estimator
 */
class DummyEstimator(override val uid: String) extends Estimator[DummyModel] {

  def this() = this("abcd")

  override def fit(dataset: Dataset[_]): DummyModel = {
    //    dataset.rdd.mapPartitions { iter =>
    //      iter
    //    }.collect()
    println("--------------------------------------------------------- DummyEstimator ")
    new DummyModel(this.uid)
  }

  override def copy(extra: ParamMap): Estimator[DummyModel] = this

  override def transformSchema(schema: StructType): StructType = schema
}



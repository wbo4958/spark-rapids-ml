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

package com.nvidia.rapids.ml

import org.apache.commons.logging.LogFactory
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.Row
import org.apache.spark.sql.connect.planner.SparkConnectPlanner
import org.apache.spark.sql.connect.plugin.RelationPlugin
import org.apache.spark.connect.{proto => sparkProto}
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.evaluation.{Evaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.rapids.RapidsUtils
import org.apache.spark.sql.rapids.Utils
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.json4s._
import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import java.util.Optional
import scala.jdk.CollectionConverters.SeqHasAsJava

class RapidsRelationPlugin extends RelationPlugin {
  protected val logger = LogFactory.getLog("Spark-Rapids-ML RapidsRelationPlugin")

  override def transform(bytes: Array[Byte], sparkConnectPlanner: SparkConnectPlanner): Optional[LogicalPlan] = {
    logger.info("In RapidsRelationPlugin")

    val rel = com.google.protobuf.Any.parseFrom(bytes)
    val sparkSession = sparkConnectPlanner.session

    // CrossValidation
    if (rel.is(classOf[proto.CrossValidatorRelation])) {
      val cvProto = rel.unpack(classOf[proto.CrossValidatorRelation])
      val dataLogicalPlan = sparkProto.Plan.parseFrom(cvProto.getDataset.toByteArray)
      val dataset = Utils.ofRows(sparkSession,
        sparkConnectPlanner.transformRelation(dataLogicalPlan.getRoot))

      val estProto = cvProto.getEstimator
      println(s"------------------------------- name: ${estProto.getName}")
      var estimator: Option[Estimator[_]] = None
      if (estProto.getName == "LogisticRegression") {
        estimator = Some(new RapidsLogisticRegression(uid = estProto.getUid))
        val estParams = estProto.getParams
        RapidsUtils.setParams(estimator.get, estParams)
      }
      val evalProto = cvProto.getEvaluator
      var evaluator: Option[Evaluator] = None
      if (evalProto.getName == "MulticlassClassificationEvaluator") {
        evaluator = Some(new MulticlassClassificationEvaluator(uid = evalProto.getUid))
        val evalParams = evalProto.getParams
        RapidsUtils.setParams(evaluator.get, evalParams)
      }

      val cv = new RapidsCrossValidator(uid = "xx")
      RapidsUtils.setParams(cv, cvProto.getParams)
      cv.setEstimator(estimator.get).setEvaluator(evaluator.get)

      dataset.show()
      val resultDf = sparkSession.createDataFrame(
        List(Row("123456_model_id")).asJava,
        StructType(Seq(StructField("model_id", StringType))))
      Optional.of(Utils.getLogicalPlan(resultDf))
    } else {
      Optional.empty()
    }
  }
}

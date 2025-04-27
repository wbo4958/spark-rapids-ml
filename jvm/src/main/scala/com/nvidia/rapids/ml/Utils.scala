///**
// * Copyright (c) 2025, NVIDIA CORPORATION.
// *
// * Licensed under the Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// * http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//
//package com.nvidia.rapids.ml
//
//import org.apache.spark.ml.Model
//import org.apache.spark.ml.classification.LogisticRegressionModel
//import org.apache.spark.ml.param.Params
//import org.apache.spark.ml.rapids.{RapidsLogisticRegressionModel, TrainedModel}
//
//object RapidsMlUtils {
//
//  def transform(name: String): Option[String] = {
//    name match {
//      case "org.apache.spark.ml.classification.LogisticRegression" =>
//        Some("com.nvidia.rapids.ml.RapidsLogisticRegression")
//      case "org.apache.spark.ml.classification.LogisticRegressionModel" =>
//        Some("org.apache.spark.ml.rapids.RapidsLogisticRegressionModel")
//      case _ => None
//    }
//  }
//
//  // Just the user defined parameters
//  def copyParams[T <: Params, S <: Params](src: S, to: T): T = {
//    src.extractParamMap().toSeq.foreach { p =>
//      val name = p.param.name
//      if (to.hasParam(name) && src.isSet(p.param)) {
//        to.set(to.getParam(name), p.value)
//      }
//    }
//    to
//  }
//
//  def createModel(name: String, uid: String, src: Params, trainedModel: TrainedModel): Model[_] = {
//    if (name.contains("LogisticRegression")) {
//      val cpuModel = copyParams(src, trainedModel.model.asInstanceOf[LogisticRegressionModel])
//      val isMultinomial = cpuModel.numClasses != 2
//      copyParams(src, new RapidsLogisticRegressionModel(uid, cpuModel, trainedModel.modelAttributes, isMultinomial))
//    } else {
//      throw new RuntimeException(s"$name Not supported")
//    }
//  }
//
//}

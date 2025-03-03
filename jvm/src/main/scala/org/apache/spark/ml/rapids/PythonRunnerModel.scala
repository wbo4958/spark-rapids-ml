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

package org.apache.spark.ml.rapids

import net.razorvine.pickle.Pickler
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.api.python.{PythonFunction, PythonRDD, PythonWorkerUtils, SimplePythonFunction}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.execution.python.PythonPlannerRunner

import java.util.Base64
import py4j.GatewayServer.GatewayServerBuilder

import java.io.{DataInputStream, DataOutputStream}
import java.security.SecureRandom
import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters._
import scala.sys.process.Process


case class Transform(name: String, params: String, model: RapidsLogisticRegressionModel)

/**
 * PythonRunner is a bridge to launch/manage Python process. And it sends the
 * estimator related message to python process and run.
 *
 * @param transform     the estimator information
 * @param dataset input dataset
 */
class PythonRunnerModel(transform: Transform,
                   dataset: DataFrame,
                   func: PythonFunction = PythonRunner.RAPIDS_PYTHON_FUNC)
  extends PythonPlannerRunner[Object](func) with AutoCloseable {

  private val datasetKey = PythonRunner.putNewObjectToPy4j(dataset)
  private val jscKey = PythonRunner.putNewObjectToPy4j(new JavaSparkContext(dataset.sparkSession.sparkContext))

  override protected val workerModule: String = "spark_rapids_ml.connect_plugin"

  override protected def writeToPython(dataOut: DataOutputStream, pickler: Pickler): Unit = {
    println(s"in writeToPython ${transform.name} in PythonRunnerModel")
    PythonRDD.writeUTF(PythonRunner.AUTH_TOKEN, dataOut)
    PythonRDD.writeUTF(transform.name, dataOut)
    PythonRDD.writeUTF(transform.params, dataOut)
    PythonRDD.writeUTF(jscKey, dataOut)
    PythonRDD.writeUTF(datasetKey, dataOut)


    PythonRDD.writeUTF(transform.model.coef, dataOut)
    PythonRDD.writeUTF(transform.model.intercepts, dataOut)
    PythonRDD.writeUTF(transform.model.numClass, dataOut)
    dataOut.writeInt(transform.model.nCols)
    PythonRDD.writeUTF(transform.model.dtype, dataOut)
    dataOut.writeInt(transform.model.nIters)
    PythonRDD.writeUTF(transform.model.objective, dataOut)
  }

  override protected def receiveFromPython(dataIn: DataInputStream): Object = {
    // Read the model target id in py4j server
//    val x = dataIn.readInt()
//    println(s"--------------- in receiveFromPython ${x}")
//    val dfTargetId = PythonWorkerUtils.readUTF(dataIn)
//    val o = PythonRunner.getObjectAndDeref(dfTargetId)
//    println("--------------- in receiveFromPython from PythonRunnerModel begin to show")
//    o.asInstanceOf[DataFrame].show()
//    println("--------------- done in receiveFromPython from PythonRunnerModel begin to show")
//    o
    val x = 10
    x.asInstanceOf[Object]
  }

  override def close(): Unit = {
    PythonRunner.deleteObject(jscKey)
    PythonRunner.deleteObject(datasetKey)
  }
}

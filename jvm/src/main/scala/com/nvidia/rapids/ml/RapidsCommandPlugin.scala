package com.nvidia.rapids.ml

import org.sparkproject.connect.protobuf.Any
import org.apache.commons.logging.LogFactory
//import org.apache.spark.sql.connect.ml.ConnectUtils
import org.apache.spark.sql.connect.planner.SparkConnectPlanner
import org.apache.spark.sql.connect.plugin.CommandPlugin


/**
 * We could use command to achieve it, but it's going to shade all google protos
 * into the package which is so annoying.
 */
class RapidsCommandPlugin extends CommandPlugin {

  protected val logger = LogFactory.getLog("Spark-Rapids-ML Plugin")

  override def process(bytes: Array[Byte], sparkConnectPlanner: SparkConnectPlanner): Boolean = {
    if (sparkConnectPlanner.executeHolderOpt.isEmpty) {
      logger.warn("Empty executeholder!")
      return false
    }
    //    sparkConnectPlanner
    val cmdProto = Any.parseFrom(bytes)
    logger.warn("--adf-asd-fads-fa-sdf-asdf-ad-fa-df!")

//    ConnectUtils.responseModelId(sparkConnectPlanner)
    true
  }
}

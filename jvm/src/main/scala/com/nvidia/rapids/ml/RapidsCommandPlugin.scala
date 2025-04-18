package com.nvidia.rapids.ml

import org.sparkproject.connect.protobuf.Any
import org.apache.commons.logging.LogFactory
import org.apache.spark.sql.connect.planner.SparkConnectPlanner
import org.apache.spark.sql.connect.plugin.CommandPlugin
import org.apache.spark.sql.connect.ml.rapids.ConnectUtils


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

    ConnectUtils.responseModelId(sparkConnectPlanner)
    true
  }
}

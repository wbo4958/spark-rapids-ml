package com.nvidia.spark.ml

import org.apache.spark.api.plugin.{DriverPlugin, ExecutorPlugin, SparkPlugin}

class CumlPlugin extends SparkPlugin {

  override def driverPlugin(): DriverPlugin = new CumlDriverPlugin()

  override def executorPlugin(): ExecutorPlugin = new ExecutorPlugin {

  }
}

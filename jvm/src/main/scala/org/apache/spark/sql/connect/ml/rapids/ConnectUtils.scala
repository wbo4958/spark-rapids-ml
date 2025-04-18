package org.apache.spark.sql.connect.ml.rapids

import org.apache.spark.connect.proto
import org.apache.spark.sql.connect.ml.MLHandler
import org.apache.spark.sql.connect.planner.SparkConnectPlanner
import org.apache.spark.connect.proto


object ConnectUtils {

  def responseModelId(planer: SparkConnectPlanner): Unit = {

    val lit = proto.Expression
      .newBuilder()
      .setLiteral(proto.Expression.Literal.newBuilder().setInteger(32).build())
    planer.executeHolderOpt.get.responseObserver.onNext(
        proto.ExecutePlanResponse
          .newBuilder()
          .setSessionId(planer.sessionId)
          .setServerSideSessionId(planer.sessionHolder.serverSessionId)
          .setExtension(org.sparkproject.connect.protobuf.Any.pack(lit.build()))
          .build()
    )
  }
}

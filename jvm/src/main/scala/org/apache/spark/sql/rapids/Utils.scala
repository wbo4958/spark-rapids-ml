package org.apache.spark.sql.rapids

import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.classic.{Dataset, SparkSession}

object Utils {

  def ofRows(session: SparkSession, logicalPlan: LogicalPlan): DataFrame = {
    Dataset.ofRows(session, logicalPlan)
  }

  def getLogicalPlan(df: Dataset[_]): LogicalPlan = df.logicalPlan

}

#! /bin/bash

#--conf spark.sql.files.maxPartitionBytes=4294967296 \

debug="
      --conf spark.driver.extraJavaOptions=\"-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=5005\"
      --conf spark.executor.heartbeatInterval=1800000
      --conf spark.storage.blockManagerSlaveTimeoutMs=1800000
      --conf spark.network.timeout=18000000
"

  #--conf spark.plugins=com.nvidia.spark.SQLPlugin,com.nvidia.spark.ml.CumlPlugin \
  #--conf spark.plugins=com.nvidia.spark.SQLPlugin \
debug=""
spark-submit $debug \
  --master local[1] \
  --conf spark.plugins=com.nvidia.spark.SQLPlugin,com.nvidia.spark.ml.CumlPlugin \
  --conf spark.rapids.memory.gpu.pooling.enabled=false \
  --conf spark.rapids.memory.gpu.allocSize=0 \
  --conf spark.executor.extraJavaOptions="-Duser.timezone=UTC" \
  --conf spark.driver.extraJavaOptions="-Duser.timezone=UTC" \
  --conf spark.sql.cache.serializer=com.nvidia.spark.ParquetCachedBatchSerializer \
  --conf spark.rapids.sql.explain=ALL \
  --conf spark.rapids.sql.enabled=true \
  --conf spark.sql.execution.arrow.maxRecordsPerBatch=1000000000 \
  --conf spark.sql.files.maxPartitionBytes=4294967296 \
  --executor-memory 30G \
  --driver-memory 30G \
  --class com.nvidia.spark.examples.pca.Main \
  --jars rapids-4-spark_2.12-22.10.0-cuda11.jar,/home/bobwang/work.d/ml/rapids-ml-scala/target/rapids-4-spark-ml_2.12-22.10.0-plugin.jar \
    /home/bobwang/work.d/nvspark/spark-rapids-examples/examples/ML+DL-Examples/Spark-cuML/pca/target/PCAExample-22.10.0-SNAPSHOT.jar /home/bobwang/data/sparkcuml/pca/25m-30-parquet


jps | grep SparkSubmit | awk '{print $1}' | xargs   kill -9

export PYSPARK_PYTHON=/home/bobwang/anaconda3/envs/rapids-25.02/bin/python

echo "$# ---"
if [ $# -gt 0 ]; then
  echo "---------------"
  export SPARK_SUBMIT_OPTS=-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=5005
fi

CONNECT_JAR=`readlink -f $SPARK_HOME/jars/spark-connect_2.13*`

RAPIDS_ML_jar=/home/bobwang/work.d/spark-rapids-ml/jvm/target/com.nvidia.rapids.ml-1.0-SNAPSHOT.jar
#RAPIDS_ML_jar=""

$SPARK_HOME/sbin/start-connect-server.sh \
  --master spark://$myip:7077 \
  --num-executors=1 \
  --conf spark.task.maxFailures=1 \
  --conf spark.stage.maxAttempts=1 \
  --conf spark.stage.maxConsecutiveAttempts=1\
  --conf spark.executor.memory=25G \
  --conf spark.executor.cores=1 \
  --conf spark.task.cpus=1 \
  --conf spark.executor.resource.gpu.amount=1 \
  --conf spark.task.resource.gpu.amount=1 \
  --jars $CONNECT_JAR,$RAPIDS_ML_jar

tail -f $SPARK_HOME/logs/spark-bobwang-org.apache.spark.sql.connect.service.SparkConnectServer-1-spark-bobby.out

jps | grep SparkSubmit | awk '{print $1}' | xargs   kill -9

export PYSPARK_PYTHON=/home/bobwang/anaconda3/envs/rapids-25.02/bin/python

echo "$# ---"
if [ $# -gt 0 ]; then
  echo "---------------"
  #export SPARK_SUBMIT_OPTS=-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=5005
fi
#  --conf spark.python.worker.reuse=false \
$SPARK_HOME/sbin/start-connect-server.sh \
  --master local[*] \
  --jars $SPARK_HOME/jars/spark-connect_2.13-4.1.0-SNAPSHOT.jar,target/com.nvidia.rapids.ml-1.0-SNAPSHOT.jar

tail -f /home/bobwang/work.d/spark/spark-master/spark-4.1.0-SNAPSHOT-bin-wbo4958-spark/logs/spark-bobwang-org.apache.spark.sql.connect.service.SparkConnectServer-1-spark-bobby.out

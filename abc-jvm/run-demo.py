from pyspark.ml.classification import (LogisticRegression,
                                       LogisticRegressionModel)
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

spark = (SparkSession.builder.remote("sc://localhost")
         .config("spark.connect.ml.backend.classes", "com.nvidia.rapids.ml.Plugin")
         .config("spark.rapids.ml.python.transform.enabled", "true")
         .getOrCreate())

df = spark.createDataFrame([
        (Vectors.dense([1.0, 2.0]), 1),
        (Vectors.dense([2.0, -1.0]), 1),
        (Vectors.dense([-3.0, -2.0]), 0),
        (Vectors.dense([-1.0, -2.0]), 0),
        ], schema=['features', 'label'])
lr = LogisticRegression(maxIter=19, tol=0.0023)
model = lr.fit(df)
print(f"======== model.getMaxIter(): {model.getMaxIter()}")
print(f"======== model.intercept: {model.intercept}")
print(f"======== model.coefficients: {model.coefficients}")
model.transform(df).show()

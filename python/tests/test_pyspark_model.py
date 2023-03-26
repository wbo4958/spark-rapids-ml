

from python.tests.sparksession import CleanSparkSession


def test_pca_model():
    with CleanSparkSession() as spark:
        from pyspark.ml.linalg import Vectors
        data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
                (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
                (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
        df = spark.createDataFrame(data, ["features"])
        from spark_rapids_ml.feature import PCA
        pca = PCA(k=2, inputCol="features")
        pca.setOutputCol("pca_features")
        model = pca.fit(df)

        from pyspark.ml.feature import PCAModel
        sc = spark.sparkContext

        from pyspark.ml.common import _py2java
        _java_pc = _py2java(sc, model.pc)
        _java_explainedVariance = _py2java(sc, model.explainedVariance)
        java_model = sc._jvm.org.apache.spark.ml.feature.PCAModel("xxx", _java_pc, _java_explainedVariance)

        final_model = PCAModel(java_model)
        final_model.setInputCol("features")
        final_model.set(final_model.k, 2)
        final_model.transform(df).show()


def test_ml_kmeans_model():
    from pyspark.mllib.clustering import KMeansModel as MllibKMeansModel
    from pyspark.mllib.common import _py2java
    from pyspark.mllib.linalg import _convert_to_vector
    from pyspark.ml.linalg import Vectors

    from spark_rapids_ml.clustering import KMeans as CumlKMeans

    with CleanSparkSession() as spark:
        data = [(Vectors.dense([0.0, 0.0]), 2.0), (Vectors.dense([1.0, 1.0]), 2.0),
                (Vectors.dense([9.0, 8.0]), 2.0), (Vectors.dense([8.0, 9.0]), 2.0)]
        df = spark.createDataFrame(data, ["features", "weighCol"])
        sc = spark.sparkContext

        centers = [[0.5, 0.5], [8.5, 8.5]]
        cuml_model = CumlKMeans().fit(df)

        centers = cuml_model.cluster_centers_

        # kmeans_mllib_model = MllibKMeansModel(centers)

        java_centers = _py2java(sc, [_convert_to_vector(c) for c in centers])
        java_model = sc._jvm.org.apache.spark.mllib.clustering.KMeansModel(java_centers)

        final_model = sc._jvm.org.apache.spark.ml.clustering.KMeansModel("xxx", java_model)

        from pyspark.ml.clustering import KMeansModel as MLKmeansModel
        kmeans_model = MLKmeansModel(final_model)
        kmeans_model.transform(df).show(100, False)
        kmeans_model.write().overwrite().save("/tmp/xxxx")



def test_lr_model():
    from spark_rapids_ml.regression import LinearRegression as CumlLinearRegression
    from pyspark.ml.regression import LinearRegressionModel as LRModel
    from pyspark.ml.linalg import Vectors
    from pyspark.ml.common import _py2java
    from pyspark.ml.linalg import _convert_to_vector

    with CleanSparkSession() as spark:
        df = spark.createDataFrame([
            (1.0, 2.0, Vectors.dense(1.0, 2.0)),
            (0.0, 2.0, Vectors.dense(3.2, 7.2))], ["label", "weight", "features"])
        lr = CumlLinearRegression(regParam=0.0, solver="normal")
        model = lr.fit(df)
        intercept = float(model.intercept)
        coef = _convert_to_vector(model.coefficients)
        print(coef)

        sc = spark.sparkContext
        assert sc._jvm is not None

        java_model = sc._jvm.org.apache.spark.ml.regression.LinearRegressionModel(
            "xyx",
            _py2java(sc, coef), intercept, 0.2)

        final_model = LRModel(java_model)
        final_model.transform(df).show()
        # final_model.write().overwrite().save("/tmp/zzzzzz")

        # model.write().overwrite().save("/tmp/xyz")
        # model.transform(df).show()

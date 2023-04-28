from typing import Optional

from spark_rapids_ml.safety import safe_patch

from .classification import RandomForestClassifier
from pyspark.ml.classification import RandomForestClassifier as SparkRFC


def gpu_acceleration(
        enabled: bool = False,
        num_workers: Optional[int] = None,
) -> None:

    def patched_fit(original, self, *args, **kwargs):
        if enabled:
            if isinstance(self, SparkRFC):
                rfc = RandomForestClassifier()
                if num_workers is not None:
                    rfc.num_workers = num_workers

                rfc.setFeaturesCol(self.getFeaturesCol())
                rfc.setLabelCol(self.getLabelCol())

                # self._copyValues(rfc)
                # TODO copy parameters from pyspark to spark_rapids_ml
                # fit_result = rfc.fit(args[0])
                print("Calling spark-rapids-ml RandomForestClassifier.fit()")
                fit_result = rfc.fit(*args, **kwargs)
                print("Finish the spark-rapids-ml training")
                return fit_result
        else:
            fit_result = original(self, *args, **kwargs)
            return fit_result

    safe_patch("pyspark.ml.classification.RandomForestClassifier", SparkRFC, "fit", patched_fit)


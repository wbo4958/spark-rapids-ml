from typing import Optional

from spark_rapids_ml.safety import safe_patch


def gpu_acceleration(
        enabled: bool = False,
        num_workers: Optional[int] = None,
) -> None:
    from pyspark.ml.base import Estimator

    from pyspark.ml.classification import RandomForestClassifier as SparkRFC

    def patched_fit(original, self, *args, **kwargs):
        if enabled and isinstance(SparkRFC, self):
            from spark_rapids_ml.classification import RandomForestClassifier
            rfc = RandomForestClassifier()
            if num_workers is not None:
                rfc.num_workers = num_workers

            # self._copyValues(rfc)
            # copy parameters from pyspark to spark_rapids_ml
            fit_result = rfc.fit(*args, **kwargs)
            return fit_result
        else:
            return original(self, *args, **kwargs)

    safe_patch("spark", Estimator, "fit", patched_fit)

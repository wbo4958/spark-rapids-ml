from typing import Optional

from spark_rapids_ml.safety import safe_patch

from .classification import RandomForestClassifier

def gpu_acceleration(
        enabled: bool = False,
        num_workers: Optional[int] = None,
) -> None:
    from pyspark.ml.base import Estimator

    from pyspark.ml.classification import RandomForestClassifier as SparkRFC

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
                print("++++++++++++++++++++++++++++++")

                fit_result = rfc.fit(*args, **kwargs)
                print("------------- the final model returned by cuml randomforest classifier")
                fit_result.transform(args[0]).show()
                print("-----------------------------")
                return fit_result
        else:
            fit_result = original(self, *args, **kwargs)
            return fit_result

    safe_patch("spark", Estimator, "fit", patched_fit)

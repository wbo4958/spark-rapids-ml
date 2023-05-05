from multiprocessing.pool import ThreadPool
from typing import Callable, List, Sequence, Tuple, cast

import numpy as np
from pyspark import inheritable_thread_target
from pyspark.ml import Estimator, Model, Transformer
from pyspark.ml._typing import ParamMap
from pyspark.ml.evaluation import Evaluator
from pyspark.ml.tuning import CrossValidator as PySparkCrossValidator
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.sql import DataFrame


def _parallelFitTasks(
    est: Estimator,
    train: DataFrame,
    eva: Evaluator,
    validation: DataFrame,
    epm: Sequence["ParamMap"],
    collectSubModel: bool,
) -> List[Callable[[], Tuple[int, float, Transformer]]]:
    """
    Creates a list of callables which can be called from different threads to fit and evaluate
    an estimator in parallel. Each callable returns an `(index, metric)` pair.

    Parameters
    ----------
    est : :py:class:`pyspark.ml.baseEstimator`
        he estimator to be fit.
    train : :py:class:`pyspark.sql.DataFrame`
        DataFrame, training data set, used for fitting.
    eva : :py:class:`pyspark.ml.evaluation.Evaluator`
        used to compute `metric`
    validation : :py:class:`pyspark.sql.DataFrame`
        DataFrame, validation data set, used for evaluation.
    epm : :py:class:`collections.abc.Sequence`
        Sequence of ParamMap, params maps to be used during fitting & evaluation.
    collectSubModel : bool
        Whether to collect sub model.

    Returns
    -------
    tuple
        (int, float, subModel), an index into `epm` and the associated metric value.
    """
    modelIter = est.fitMultiple(train, epm)

    def singleTask() -> Tuple[int, float, Transformer]:
        index, model = next(modelIter)
        # TODO: duplicate evaluator to take extra params from input
        #  Note: Supporting tuning params in evaluator need update method
        #  `MetaAlgorithmReadWrite.getAllNestedStages`, make it return
        #  all nested stages and evaluators
        #  TODO, copy the
        # df = model.transform(validation, epm[index])
        metric = model.evaluateByGpu(validation, epm[index])
        return index, metric, model if collectSubModel else None

    return [singleTask] * len(epm)


def _gen_avg_and_std_metrics_(
    metrics_all: List[List[float]],
) -> Tuple[List[float], List[float]]:
    avg_metrics = np.mean(metrics_all, axis=0)
    std_metrics = np.std(metrics_all, axis=0)
    return list(avg_metrics), list(std_metrics)


class CrossValidator(PySparkCrossValidator):
    """Gpu-versioned CrossValidator"""

    def _fit(self, dataset: DataFrame) -> "CrossValidatorModel":
        """Copied from PySparkCrossValidator"""
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        nFolds = self.getOrDefault(self.numFolds)
        metrics_all = [[0.0] * numModels for i in range(nFolds)]

        pool = ThreadPool(processes=min(self.getParallelism(), numModels))
        subModels = None
        collectSubModelsParam = self.getCollectSubModels()
        if collectSubModelsParam:
            subModels = [[None for j in range(numModels)] for i in range(nFolds)]

        datasets = self._kFold(dataset)
        for i in range(nFolds):
            validation = datasets[i][1].cache()
            train = datasets[i][0].cache()

            tasks = map(
                inheritable_thread_target,
                _parallelFitTasks(
                    est, train, eva, validation, epm, collectSubModelsParam
                ),
            )
            for j, metric, subModel in pool.imap_unordered(lambda f: f(), tasks):
                metrics_all[i][j] = metric
                if collectSubModelsParam:
                    assert subModels is not None
                    subModels[i][j] = subModel

            validation.unpersist()
            train.unpersist()

        metrics, std_metrics = _gen_avg_and_std_metrics_(metrics_all)

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
        bestModel = est.fit(dataset, epm[bestIndex])

        # TODO, compatiable with different pyspark version
        try:
            model = CrossValidatorModel(
                bestModel, metrics, cast(List[List[Model]], subModels), std_metrics
            )
        except:
            model = CrossValidatorModel(
                bestModel, metrics, cast(List[List[Model]], subModels)
            )

        return self._copyValues(model)

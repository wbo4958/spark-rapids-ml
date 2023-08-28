#
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from typing import List, Tuple

from pyspark import Row
from pyspark.ml.evaluation import BinaryClassificationEvaluator


class BinaryLabelCounter:
    """Python version of BinaryLabelCounter"""

    def __init__(self, weighted_num_positives=0.0, weighted_num_negatives=0.0):
        self.weighted_num_positives = weighted_num_positives
        self.weighted_num_negatives = weighted_num_negatives

    def add_negatives(self, count: int) -> "BinaryLabelCounter":
        self.weighted_num_negatives += count
        return self

    def add_positives(self, count: int) -> "BinaryLabelCounter":
        self.weighted_num_positives += count
        return self

    def merge(self, other: "BinaryLabelCounter") -> "BinaryLabelCounter":
        return BinaryLabelCounter(self.weighted_num_positives + other.weighted_num_positives,
                                  self.weighted_num_negatives + other.weighted_num_negatives)


class BinaryConfusionMatrix:
    """Python version of BinaryConfusionMatrix"""

    def __init__(self, count: BinaryLabelCounter, total_count: BinaryLabelCounter) -> None:
        self.count = count
        self.totalCount = total_count

    def weighted_true_positives(self):
        return self.count.weighted_num_positives

    def weighted_false_positives(self):
        return self.count.weighted_num_negatives

    def weighted_false_negatives(self):
        return self.totalCount.weighted_num_positives - self.count.weighted_num_positives

    def weighted_true_negatives(self):
        return self.totalCount.weighted_num_negatives - self.count.weighted_num_negatives

    def weighted_positives(self):
        return self.totalCount.weighted_num_positives

    def weighted_negatives(self):
        return self.totalCount.weighted_num_negatives

    def false_positive_rate(self):
        if self.weighted_negatives == 0.0:
            return 0.0
        else:
            return self.weighted_false_positives() / self.weighted_negatives()

    def recall(self):
        if self.weighted_positives == 0.0:
            return 0.0
        else:
            return self.weighted_true_positives() / self.weighted_positives()


# This class is aligning with Spark BinaryClassificationMetrics scala version.
class BinaryClassificationMetrics:
    """Metrics for binary classification case."""

    def __init__(self, curves: List[Tuple[float, float]]) -> None:
        self._curves = curves

    @classmethod
    def from_rows(cls, rows: List[Row]) -> "BinaryClassificationMetrics":
        # confusions for prediction 0.0
        confusions_0 = BinaryLabelCounter()
        # confusions for prediction 1.0
        confusions_1 = BinaryLabelCounter()

        for row in rows:
            counter = confusions_0 if row["rawPrediction"] == 0.0 else confusions_1
            if row["label"] > 0.5:
                counter.add_positives(row["count"])
            else:
                counter.add_negatives(row["count"])

        total = confusions_0.merge(confusions_1)
        confusion_matrix_1 = BinaryConfusionMatrix(confusions_1, total)
        confusion_matrix_all = BinaryConfusionMatrix(total, total)

        # follow scala way, sortByKey(ascending=False)
        curves = [
            (0.0, 0.0),
            (confusion_matrix_1.false_positive_rate(), confusion_matrix_1.recall()),
            (confusion_matrix_all.false_positive_rate(), confusion_matrix_all.recall()),
            (1.0, 1.0),
        ]

        return BinaryClassificationMetrics(curves)

    @property
    def area_under_roc(self) -> float:
        size = len(self._curves)
        assert size > 0
        local_area = 0.0

        def trapezoid(x, y):
            return (y[0] - x[0]) * (y[1] + x[1]) / 2.0

        for i in range(size):
            if i + 1 < size:
                local_area += trapezoid(self._curves[i], self._curves[i + 1])

        return local_area

    def evaluate(self, evaluator: BinaryClassificationEvaluator) -> float:
        metric_name = evaluator.getMetricName()
        if metric_name == "areaUnderROC":
            return self.area_under_roc
        elif metric_name == "areaUnderPR":
            raise RuntimeError("areaUnderPR is not supported yet")
        else:
            raise RuntimeError(f"Unsupported {metric_name}")

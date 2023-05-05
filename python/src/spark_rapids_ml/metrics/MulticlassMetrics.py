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

from typing import List

from pyspark import Row


class MulticlassMetrics:
    def __init__(self, num_class: int, rows: List[Row]) -> None:
        if num_class <= 2:
            raise RuntimeError(
                f"MulticlassMetrics requires at least 3 classes. Found {num_class}"
            )

        self._num_classes = 3
        self._tp_by_class = {}
        self._fp_by_class = {}
        self._label_count_by_class = {}
        self._label_count = 0

        for i in range(self._num_classes):
            self._tp_by_class[float(i)] = 0.0
            self._label_count_by_class[float(i)] = 0.0
            self._fp_by_class[float(i)] = 0.0

        for row in rows:
            self._label_count += row.total
            self._label_count_by_class[row.label] += row.total

            if row.label == row.prediction:
                self._tp_by_class[row.label] += row.total
            else:
                self._fp_by_class[row.prediction] += row.total

    def _precision(self, label: float) -> float:
        tp = self._tp_by_class[label]
        fp = self._fp_by_class[label]
        return 0.0 if (tp + fp == 0) else tp / (tp + fp)

    def _recall(self, label: float) -> float:
        return self._tp_by_class[label] / self._label_count_by_class[label]

    def _f_measure(self, label: float, beta: float = 1.0) -> float:
        p = self._precision(label)
        r = self._recall(label)
        beta_sqrd = beta * beta
        return 0.0 if (p + r == 0) else (1 + beta_sqrd) * p * r / (beta_sqrd * p + r)

    def weighted_fmeasure(self, beta: float = 1.0) -> float:
        sum = 0.0
        for k, v in self._label_count_by_class.items():
            sum += self._f_measure(k, beta) * v / self._label_count
        return sum

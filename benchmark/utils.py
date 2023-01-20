#
# Copyright (c) 2022, NVIDIA CORPORATION.
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
from time import time

from pyspark.sql import SparkSession
from typing import List, Any, Callable


class WithSparkSession(object):
    def __init__(self, confs: List[str]) -> None:
        builder = SparkSession.builder
        for conf in confs:
            key, value = conf.split("=")
            builder = builder.config(key, value)
        self.spark = builder.getOrCreate()

    def __enter__(self) -> SparkSession:
        return self.spark

    def __exit__(self, *args: Any) -> None:
        self.spark.stop()


def with_benchmark(phrase: str, action: Callable) -> Any:
    start = time()
    result = action()
    end = time()
    print('-' * 100)
    print('{} takes {} seconds'.format(phrase, round(end - start, 2)))
    print('-' * 100)
    return result

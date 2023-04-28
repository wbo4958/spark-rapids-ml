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
# The idea of gpu_acceleration is borrowed from MLFlow.

from typing import Optional

from . import spark
from .import_hooks import register_post_import_hook


def gpu_acceleration(
        enabled: bool = False,
        num_workers: Optional[int] = None,
) -> None:
    """Enable or disable pyspark estimators acceleration by spark_rapids_ml using nvidia GPUS

    :param enabled:  If True, Enable spark_rapids_ml to accelerate pyspark estimators. False, disable.

    :param num_workers: The total gpus to be used. spark_rapids_ml can infer the gpus and use all
                        of them in spark_rapids_ml by default, and users can still specify it manually.

    """

    register_post_import_hook(spark.gpu_acceleration, overwrite=True)

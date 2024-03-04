# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
__main__ for mslite_bench
"""

from mslite_bench.utils import ArgParser, InferLogger
from mslite_bench.common.model_info_enum import TaskType


if __name__ == '__main__':
    args = ArgParser.parse_arguments()
    logger = InferLogger(args.log_path)
    logger.info('[EASY INFER] Start model infer now!')

    if args.task_type == TaskType.FRAMEWORK_CMP.value:
        try:
            from mslite_bench.tools.cross_framework_accuracy import CrossFrameworkAccSummary
        except ImportError as e:
            logger.error(f'Failed to import CFA:{e}')
            raise
        logger.info('[EASY INFER] Start framework compare task!')
        CrossFrameworkAccSummary.accuracy_compare_func(args, logger)
    elif args.task_type in set(
            [TaskType.NPU_DYNAMIC_INFER.value, TaskType.MODEL_INFER.value]
    ):
        try:
            from mslite_bench.tools.easy_infer import EasyInfer
        except ImportError as e:
            logger.error(f'Failed to import easy infer:{e}')
            raise
        if args.task_type == TaskType.NPU_DYNAMIC_INFER.value:
            logger.info('[EASY INFER] Start mslite model dynamic infer task!')
            EasyInfer.ms_dynamic_input_infer(args, logger)
        else:
            logger.info(f'[EASY INFER] Start model infer: {args.model_file}!')
            EasyInfer.easy_infer(args, logger)
    else:
        raise NotImplementedError(f'Task Type {args.task_type} ')

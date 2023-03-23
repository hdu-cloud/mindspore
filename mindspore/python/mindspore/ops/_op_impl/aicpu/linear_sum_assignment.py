# Copyright 2022 Huawei Technologies Co., Ltd
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

"""LinearSumAssignment op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType


lsap_op_info = AiCPURegOp("LinearSumAssignment") \
    .fusion_type("OPAQUE") \
    .input(0, "cost_matrix", "required") \
    .input(1, "dimension_limit", "required") \
    .input(2, 'maximize', "required") \
    .output(0, "row_ind", "required") \
    .output(1, "col_ind", "required") \
    .attr("cust_aicpu", "str") \
    .dtype_format(DataType.F64_Default, DataType.I64_Default,
                  DataType.BOOL_Default, DataType.I64_Default, DataType.I64_Default,) \
    .dtype_format(DataType.F32_Default, DataType.I64_Default,
                  DataType.BOOL_Default, DataType.I64_Default, DataType.I64_Default,) \
    .get_op_info()


@op_info_register(lsap_op_info)
def _linear_sum_assignment_aicpu():
    """LinearSumAssignment AiCPU register"""
    return

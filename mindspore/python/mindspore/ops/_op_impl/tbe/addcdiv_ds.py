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

"""Addcdiv op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

addcdiv_ds_op_info = TBERegOp("Addcdiv") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("addcdiv.so") \
    .compute_cost(10) \
    .kernel_name("addcdiv") \
    .partial_flag(True) \
    .dynamic_shape(True) \
    .input(0, "input_data", False, "required", "all") \
    .input(1, "x1", False, "required", "all") \
    .input(2, "x2", False, "required", "all") \
    .input(3, "value", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(addcdiv_ds_op_info)
def _addcdiv_ds_tbe():
    """Addcdiv TBE register"""
    return

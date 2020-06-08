# Copyright 2020 Huawei Technologies Co., Ltd
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

"""ReverseSequence op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType
reverse_sequence_op_info = AiCPURegOp("ReverseSequence") \
    .fusion_type("OPAQUE") \
    .input(0, "x", "required") \
    .input(1, "seq_lengths", "required") \
    .output(0, "y", "required") \
    .attr("seq_dim", "int") \
    .attr("batch_dim", "int") \
    .dtype_format(DataType.BOOL_Default, DataType.I32_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.I8_Default, DataType.I32_Default, DataType.I8_Default) \
    .dtype_format(DataType.I16_Default, DataType.I32_Default, DataType.I16_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I32_Default, DataType.I64_Default) \
    .dtype_format(DataType.U8_Default, DataType.I32_Default, DataType.U8_Default) \
    .dtype_format(DataType.U16_Default, DataType.I32_Default, DataType.U16_Default) \
    .dtype_format(DataType.U32_Default, DataType.I32_Default, DataType.U32_Default) \
    .dtype_format(DataType.U64_Default, DataType.I32_Default, DataType.U64_Default) \
    .dtype_format(DataType.F16_Default, DataType.I32_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.I32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F64_Default, DataType.I32_Default, DataType.F64_Default) \
    .dtype_format(DataType.BOOL_NCHW, DataType.I32_NCHW, DataType.BOOL_NCHW) \
    .dtype_format(DataType.I8_NCHW, DataType.I32_NCHW, DataType.I8_NCHW) \
    .dtype_format(DataType.I16_NCHW, DataType.I32_NCHW, DataType.I16_NCHW) \
    .dtype_format(DataType.I32_NCHW, DataType.I32_NCHW, DataType.I32_NCHW) \
    .dtype_format(DataType.I64_NCHW, DataType.I32_NCHW, DataType.I64_NCHW) \
    .dtype_format(DataType.U8_NCHW, DataType.I32_NCHW, DataType.U8_NCHW) \
    .dtype_format(DataType.U16_NCHW, DataType.I32_NCHW, DataType.U16_NCHW) \
    .dtype_format(DataType.U32_NCHW, DataType.I32_NCHW, DataType.U32_NCHW) \
    .dtype_format(DataType.U64_NCHW, DataType.I32_NCHW, DataType.U64_NCHW) \
    .dtype_format(DataType.F16_NCHW, DataType.I32_NCHW, DataType.F16_NCHW) \
    .dtype_format(DataType.F32_NCHW, DataType.I32_NCHW, DataType.F32_NCHW) \
    .dtype_format(DataType.F64_NCHW, DataType.I32_NCHW, DataType.F64_NCHW) \
    .dtype_format(DataType.BOOL_Default, DataType.I64_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.I8_Default, DataType.I64_Default, DataType.I8_Default) \
    .dtype_format(DataType.I16_Default, DataType.I64_Default, DataType.I16_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default) \
    .dtype_format(DataType.U8_Default, DataType.I64_Default, DataType.U8_Default) \
    .dtype_format(DataType.U16_Default, DataType.I64_Default, DataType.U16_Default) \
    .dtype_format(DataType.U32_Default, DataType.I64_Default, DataType.U32_Default) \
    .dtype_format(DataType.U64_Default, DataType.I64_Default, DataType.U64_Default) \
    .dtype_format(DataType.F16_Default, DataType.I64_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.I64_Default, DataType.F32_Default) \
    .dtype_format(DataType.F64_Default, DataType.I64_Default, DataType.F64_Default) \
    .dtype_format(DataType.BOOL_NCHW, DataType.I64_NCHW, DataType.BOOL_NCHW) \
    .dtype_format(DataType.I8_NCHW, DataType.I64_NCHW, DataType.I8_NCHW) \
    .dtype_format(DataType.I16_NCHW, DataType.I64_NCHW, DataType.I16_NCHW) \
    .dtype_format(DataType.I32_NCHW, DataType.I64_NCHW, DataType.I32_NCHW) \
    .dtype_format(DataType.I64_NCHW, DataType.I64_NCHW, DataType.I64_NCHW) \
    .dtype_format(DataType.U8_NCHW, DataType.I64_NCHW, DataType.U8_NCHW) \
    .dtype_format(DataType.U16_NCHW, DataType.I64_NCHW, DataType.U16_NCHW) \
    .dtype_format(DataType.U32_NCHW, DataType.I64_NCHW, DataType.U32_NCHW) \
    .dtype_format(DataType.U64_NCHW, DataType.I64_NCHW, DataType.U64_NCHW) \
    .dtype_format(DataType.F16_NCHW, DataType.I64_NCHW, DataType.F16_NCHW) \
    .dtype_format(DataType.F32_NCHW, DataType.I64_NCHW, DataType.F32_NCHW) \
    .dtype_format(DataType.F64_NCHW, DataType.I64_NCHW, DataType.F64_NCHW) \
    .get_op_info()

@op_info_register(reverse_sequence_op_info)
def _reverse_sequence_aicpu():
    """ReverseSequence AiCPU register"""
    return

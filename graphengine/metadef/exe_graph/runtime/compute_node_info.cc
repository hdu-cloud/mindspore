/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "exe_graph/runtime/compute_node_info.h"
#include "common/checker.h"
namespace gert {
ge::graphStatus ComputeNodeInfo::CalcSize(size_t ir_inputs_num, size_t inputs_num, size_t outputs_num,
                                          size_t &total_size) {
  size_t ir_inputs_size;
  size_t inputs_size;
  size_t outputs_size;

  GE_ASSERT_TRUE(!ge::MulOverflow(sizeof(AnchorInstanceInfo), ir_inputs_num, ir_inputs_size));
  GE_ASSERT_TRUE(!ge::MulOverflow(sizeof(CompileTimeTensorDesc), inputs_num, inputs_size));
  GE_ASSERT_TRUE(!ge::MulOverflow(sizeof(CompileTimeTensorDesc), outputs_num, outputs_size));

  total_size = sizeof(ComputeNodeInfo);
  GE_ASSERT_TRUE(!ge::AddOverflow(total_size, ir_inputs_size, total_size));
  GE_ASSERT_TRUE(!ge::AddOverflow(total_size, inputs_size, total_size));
  GE_ASSERT_TRUE(!ge::AddOverflow(total_size, outputs_size, total_size));

  return ge::GRAPH_SUCCESS;
}
void ComputeNodeInfo::Init(size_t ir_inputs_num, size_t inputs_num, size_t outputs_num,
                           const char *node_name, const char *node_type) {
  ir_inputs_num_ = ir_inputs_num;
  inputs_num_ = inputs_num;
  outputs_num_ = outputs_num;
  node_name_ = node_name;
  node_type_ = node_type;
}
AnchorInstanceInfo *ComputeNodeInfo::MutableInputInstanceInfo(size_t ir_index) {
  return const_cast<AnchorInstanceInfo *>(GetInputInstanceInfo(ir_index));
}
CompileTimeTensorDesc *ComputeNodeInfo::MutableInputTdInfo(size_t index) {
  return const_cast<CompileTimeTensorDesc *>(GetInputTdInfo(index));
}
CompileTimeTensorDesc *ComputeNodeInfo::MutableOutputTdInfo(size_t index) {
  return const_cast<CompileTimeTensorDesc *>(GetOutputTdInfo(index));
}
RuntimeAttrs *ComputeNodeInfo::MutableAttrs() {
  return const_cast<RuntimeAttrs *>(GetAttrs());
}
}  // namespace gert
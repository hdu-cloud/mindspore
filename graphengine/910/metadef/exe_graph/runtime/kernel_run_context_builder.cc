/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "exe_graph/runtime/kernel_run_context_builder.h"
#include "exe_graph/lowering/bg_kernel_context_extend.h"
#include "graph/compute_graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_util.h"
#include "graph/def_types.h"

namespace gert {
KernelContextHolder KernelRunContextBuilder::Build(const ge::OpDescPtr &op_desc) {
  KernelContextHolder holder;
  size_t size = sizeof(KernelRunContext) + sizeof(Chain *) * (inputs_.size() + outputs_.size());
  holder.context_holder_ = ge::ComGraphMakeUnique<uint8_t[]>(size);
  if (holder.context_holder_ == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "Create context holder failed.");
    return holder;
  }
  size_t extend_info_size;
  holder.compute_node_extend_holder_ =
      bg::CreateComputeNodeInfo(MakeNode(op_desc), holder.buffer_pool_, extend_info_size);

  if (holder.compute_node_extend_holder_ == nullptr) {
    GELOGE(ge::GRAPH_FAILED,
           "Failed to create compute node info for node %s", op_desc->GetName().c_str());
    return holder;
  }
  auto compute_node_info = ge::PtrToPtr<uint8_t, ComputeNodeInfo>(holder.compute_node_extend_holder_.get());
  compute_node_info->SetNodeName(
      holder.buffer_pool_.GetBufById(reinterpret_cast<size_t>(compute_node_info->GetNodeName())));
  compute_node_info->SetNodeType(
      holder.buffer_pool_.GetBufById(reinterpret_cast<size_t>(compute_node_info->GetNodeType())));
  holder.context_ = ge::PtrToPtr<uint8_t, KernelContext>(holder.context_holder_.get());
  auto kernel_run_context = holder.context_->GetContext();
  kernel_run_context->input_size = inputs_.size();
  kernel_run_context->output_size = outputs_.size();
  kernel_run_context->compute_node_info = compute_node_info;
  kernel_run_context->output_start = &(kernel_run_context->values[kernel_run_context->input_size]);
  holder.value_holder_.resize(inputs_.size() + outputs_.size());
  for (size_t i = 0UL; i < holder.value_holder_.size(); ++i) {
    kernel_run_context->values[i] = ge::PtrToPtr<Chain, AsyncAnyValue>(&holder.value_holder_[i]);
  }
  for (size_t i = 0UL; i < inputs_.size(); ++i) {
    holder.value_holder_[i].Set(inputs_[i].first, inputs_[i].second);
  }
  for (size_t i = 0UL; i < outputs_.size(); ++i) {
    holder.value_holder_[inputs_.size() + i].Set(outputs_[i].first, outputs_[i].second);
  }
  return holder;
}

ge::NodePtr KernelRunContextBuilder::MakeNode(const ge::OpDescPtr &op_desc) {
  const auto node_id = op_desc->GetId();
  graph_ = std::make_shared<ge::ComputeGraph>("tmp");
  auto fake_node = graph_->AddNode(op_desc);
  for (size_t i = 0UL; i < op_desc->GetAllInputsSize(); ++i) {
    const auto input_desc = op_desc->GetInputDesc(i);
    if (input_desc.IsValid() != ge::GRAPH_SUCCESS) {
      GELOGD("Node: %s, input: %zu, is invalid, skip add edge.", op_desc->GetName().c_str(), i);
      continue;
    }
    auto op_data = ge::OpDescBuilder(std::to_string(i), "Data").AddInput("x").AddOutput("y").Build();
    auto data_node = graph_->AddNode(op_data);
    ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), fake_node->GetInDataAnchor(i));
  }
  // AddNode operation may change node id to 0, which need to be recovered
  op_desc->SetId(node_id);
  return fake_node;
}
}  // namespace gert

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
#include "exe_graph/lowering/frame_selector.h"
#include "exe_graph/lowering/value_holder.h"
#include "value_holder_inner.h"
#include "scoped_current_frame.h"

#include "graph/debug/ge_util.h"

#include "common/checker.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "exe_graph/runtime/execute_graph_types.h"
#include "exe_graph/lowering/graph_frame.h"

namespace gert {
namespace bg {
namespace {
ge::OutDataAnchorPtr CreateInnerData(const ge::ComputeGraphPtr &graph, const GraphFrame &graph_frame,
                                     const size_t index) {
  auto op_desc = ge::ComGraphMakeShared<ge::OpDesc>(ValueHolder::GenerateNodeName(kInnerData, graph_frame), kInnerData);
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_SUCCESS(op_desc->AddOutputDesc(ge::GeTensorDesc()));
  GE_ASSERT_TRUE(ge::AttrUtils::SetInt(op_desc, "index", index));
  auto node = graph->AddNode(op_desc);
  GE_ASSERT_NOTNULL(node);
  return node->GetOutDataAnchor(0);
}
ge::NodePtr GetOrCreateInnerNetOutput(const GraphFrame &frame) {
  auto netoutput = frame.GetExeGraph()->FindFirstNodeMatchType(kInnerNetOutput);
  if (netoutput != nullptr) {
    return netoutput;
  }
  return ValueHolder::AddNode(kInnerNetOutput, 0U, 0U, frame);
}
ge::graphStatus MoveGuardersToDeInit(const ge::NodePtr &init_node, const GraphFrame &root_frame,
                                     const std::vector<std::pair<ValueHolderPtr, size_t>> &guarders_and_out_index) {
  const auto de_init_node =
      root_frame.GetExeGraph()->FindFirstNodeMatchType(GetExecuteGraphTypeStr(ExecuteGraphType::kDeInit));
  GE_ASSERT_NOTNULL(de_init_node);
  auto de_init_graph = ge::NodeUtils::GetSubgraph(*de_init_node, 0U);
  GE_ASSERT_NOTNULL(de_init_graph);

  auto index = de_init_node->GetAllInDataAnchorsSize();
  GE_ASSERT_SUCCESS(ge::NodeUtils::AppendInputAnchor(de_init_node, index + guarders_and_out_index.size()));

  for (size_t i = 0U; i < guarders_and_out_index.size(); ++i) {
    auto guarder_node = guarders_and_out_index[i].first->GetNode();
    GE_ASSERT_NOTNULL(guarder_node);
    GE_ASSERT_SUCCESS(
        ge::GraphUtils::MoveNodeToGraph(const_cast<ge::Node *>(guarder_node)->shared_from_this(), *de_init_graph));
    GE_ASSERT_SUCCESS(
        ge::GraphUtils::AddEdge(init_node->GetOutDataAnchor(static_cast<int32_t>(guarders_and_out_index[i].second)),
                                de_init_node->GetInDataAnchor(static_cast<int32_t>(index + i))));
    GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(CreateInnerData(de_init_graph, root_frame, index + i),
                                              guarders_and_out_index[i].first->GetNode()->GetInDataAnchor(0)));
  }

  return ge::GRAPH_SUCCESS;
}
ge::graphStatus GetNewOutIndexes(const std::vector<ValueHolderPtr> &init_graph_outputs, const size_t start_index,
                                 size_t &new_out_num, std::map<ValueHolderPtr, size_t> &graph_outputs_to_out_index) {
  new_out_num = 0U;
  for (const auto &output : init_graph_outputs) {
    GE_ASSERT_NOTNULL(output, "Failed to construct on init graph, the graph builder return nullptr");
    const auto node = output->GetNode();
    GE_ASSERT_NOTNULL(node);
    const auto index = output->GetOutIndex();
    int32_t out_index = -1;
    for (const auto &anchor_and_node : ge::NodeUtils::GetOutDataNodesWithAnchorByIndex(*node, index)) {
      if (IsTypeInnerNetOutput(anchor_and_node.second->GetType().c_str())) {
        out_index = anchor_and_node.first->GetIdx();
        break;
      }
    }
    if (out_index < 0) {
      graph_outputs_to_out_index[output] = start_index + (new_out_num++);
    } else {
      graph_outputs_to_out_index[output] = static_cast<size_t>(out_index);
    }
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus ConnectOut(const ge::NodePtr &init_node,
                           GraphFrame &root_frame,
                           GraphFrame &init_frame,
                           const std::vector<ValueHolderPtr> &init_graph_outputs,
                           std::vector<ValueHolderPtr> &init_node_outputs) {
  const auto netoutput = GetOrCreateInnerNetOutput(init_frame);
  GE_ASSERT_NOTNULL(netoutput);
  const auto index = netoutput->GetAllInDataAnchorsSize();

  size_t new_out_num;
  std::map<ValueHolderPtr, size_t> graph_outputs_to_out_index;
  GE_ASSERT_SUCCESS(GetNewOutIndexes(init_graph_outputs, index, new_out_num, graph_outputs_to_out_index));

  GE_ASSERT_SUCCESS(ge::NodeUtils::AppendInputAnchor(netoutput, index + new_out_num));
  std::vector<std::pair<ValueHolderPtr, size_t>> guarders_and_out_index;
  guarders_and_out_index.reserve(init_graph_outputs.size());
  init_node_outputs.reserve(init_graph_outputs.size());
  for (const auto &holder : init_graph_outputs) {
    const auto &out_index = graph_outputs_to_out_index.at(holder);
    if (out_index >= index) {
      GE_ASSERT_NOTNULL(holder);
      GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(holder->GetNode()->GetOutDataAnchor(holder->GetOutIndex()),
                                                netoutput->GetInDataAnchor(out_index)));
    }

    auto guarder = holder->GetGuarder();
    if (guarder != nullptr) {
      guarders_and_out_index.emplace_back(guarder, out_index);
    }

    init_node_outputs.emplace_back(ValueHolder::CreateFromNode(init_node, static_cast<int32_t>(out_index),
                                                               ValueHolder::ValueHolderType::kOutput));
  }

  GE_ASSERT_SUCCESS(ge::NodeUtils::AppendOutputAnchor(init_node, index + new_out_num));
  GE_ASSERT_SUCCESS(MoveGuardersToDeInit(init_node, root_frame, guarders_and_out_index));

  for (size_t i = 0U; i < init_graph_outputs.size(); ++i) {
    init_node_outputs[i]->SetPlacement(init_graph_outputs[i]->GetPlacement());
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace
std::vector<ValueHolderPtr> FrameSelector::OnMainRoot(const std::function<std::vector<ValueHolderPtr>()> &builder) {
  if (builder == nullptr || GetGraphFrames().empty()) {
    return {};
  }
  std::vector<ValueHolderPtr> outputs;
  if (OnMainRoot(builder, outputs) != ge::GRAPH_SUCCESS) {
    GELOGW("Compatible mode, the air code is not the newest.");
    const ScopedCurrentFrame frame_guarder(GetGraphFrames().front().get());
    outputs = builder();
  }
  return outputs;
}
ge::graphStatus FrameSelector::OnMainRoot(const std::function<std::vector<ValueHolderPtr>()> &builder,
                                          std::vector<ValueHolderPtr> &outputs) {
  GE_ASSERT_NOTNULL(builder, "Failed to do frame selection, the builder is nullptr");
  // 栈底是root-frame，向上是main-frame，因此栈中至少有两个元素
  GE_ASSERT_TRUE(GetGraphFrames().size() > 1U, "Failed to do frame selection, there is no main-frame exists");

  const auto root_frame = GetGraphFrames().begin()->get();
  GE_ASSERT_NOTNULL(root_frame, "Failed to find the root frame");
  auto main_frame = (GetGraphFrames().begin() + 1)->get();
  GE_ASSERT_NOTNULL(main_frame, "Failed to find the main frame");

  // check if the main_frame is correct
  const auto root_graph = root_frame->GetExeGraph();
  GE_ASSERT_NOTNULL(root_graph, "Failed to find the root graph");
  const auto main_node = root_graph->FindFirstNodeMatchType(GetExecuteGraphTypeStr(ExecuteGraphType::kMain));
  GE_ASSERT_NOTNULL(main_node, "Failed to find the main node");
  const auto main_graph = ge::NodeUtils::GetSubgraph(*main_node, 0U);
  GE_ASSERT_TRUE(main_graph == main_frame->GetExeGraph(), "Failed to find the main frame");

  const ScopedCurrentFrame frame_guarder(main_frame);
  outputs = builder();
  return ge::GRAPH_SUCCESS;
}

ValueHolderPtr FrameSelector::OnMainRootLast(const std::function<bg::ValueHolderPtr()> &builder) {
  if (builder == nullptr || GetGraphFrames().empty()) {
    return nullptr;
  }
  GraphFrame *current_frame;
  if (GetGraphFrames().size() > 1U) {
    current_frame = (GetGraphFrames().begin() + 1)->get();
  } else {
    current_frame = GetGraphFrames().begin()->get();
  }
  const ScopedCurrentFrame frame_guarder(current_frame);
  auto output = builder();
  GetCurrentFrame()->SetLastExecNode(output);
  return output;
}

std::vector<ValueHolderPtr> FrameSelector::OnInitRoot(const std::function<std::vector<ValueHolderPtr>()> &builder) {
  std::vector<ValueHolderPtr> init_graph_outputs;
  std::vector<ValueHolderPtr> init_node_outputs;
  const auto ret = OnInitRoot(builder, init_graph_outputs, init_node_outputs);
  if (ret != ge::GRAPH_SUCCESS) {
    return {};
  }
  return init_node_outputs;
}
ge::graphStatus FrameSelector::OnInitRoot(const std::function<std::vector<ValueHolderPtr>()> &builder,
                                          std::vector<ValueHolderPtr> &init_graph_outputs,
                                          std::vector<ValueHolderPtr> &init_node_outputs) {
  GE_ASSERT_NOTNULL(builder, "Failed to do frame selection, the builder is nullptr");
  GE_ASSERT_TRUE(!GetGraphFrames().empty(), "Failed to do frame selection, there is no root-frame exists");

  const auto root_frame = GetGraphFrames().begin()->get();
  GE_ASSERT_NOTNULL(root_frame, "Failed to find the root frame");

  const auto init_node = root_frame->GetExeGraph()->FindFirstNodeMatchType(
      GetExecuteGraphTypeStr(ExecuteGraphType::kInit));
  GE_ASSERT_NOTNULL(init_node, "Failed to find the Init node from root graph");
  auto init_graph = ge::NodeUtils::GetSubgraph(*init_node, 0U);
  GE_ASSERT_NOTNULL(init_graph, "Failed to find the Init graph from init node %s", init_node->GetName().c_str());

  auto tmp_init_frame = ge::ComGraphMakeUnique<GraphFrame>(init_graph, *root_frame);
  GE_ASSERT_NOTNULL(tmp_init_frame);

  tmp_init_frame->SetCurrentComputeNode(ValueHolder::GetCurrentFrame()->GetCurrentComputeNode());
  const ScopedCurrentFrame frame_guarder(tmp_init_frame.get());
  init_graph_outputs = builder();
  if (!init_graph_outputs.empty()) {
    return ConnectOut(init_node, *root_frame, *tmp_init_frame, init_graph_outputs, init_node_outputs);
  }
  return ge::GRAPH_SUCCESS;
}
ValueHolderPtr HolderOnInit(const ValueHolderPtr &holder) {
  GE_ASSERT_NOTNULL(holder);
  const auto index = holder->GetOutIndex();
  GE_ASSERT_TRUE(index >= 0, "The holder is a ctrl holder");

  const auto holder_node = holder->GetNode();
  GE_ASSERT_NOTNULL(holder_node);
  if (strcmp(holder_node->GetType().c_str(), GetExecuteGraphTypeStr(ExecuteGraphType::kInit)) == 0) {
    const auto init_graph = ge::NodeUtils::GetSubgraph(*holder_node, 0U);
    GE_ASSERT_NOTNULL(init_graph);
    const auto netoutput = init_graph->FindFirstNodeMatchType(kInnerNetOutput);
    GE_ASSERT_NOTNULL(netoutput, "Can not find the InnerNetOutput node on the Init graph");
    const auto dst_anchor = netoutput->GetInDataAnchor(index);
    GE_ASSERT_NOTNULL(dst_anchor, "The InnerNetOutput does not have the in anchor %d", index);
    const auto src_anchor = dst_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(src_anchor);
    auto src_node = src_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(src_node);
    return ValueHolder::CreateFromNode(src_node, src_anchor->GetIdx(), ValueHolder::ValueHolderType::kOutput);
  }

  const auto holder_graph = holder_node->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(holder_graph);
  const auto parent_node = holder_graph->GetParentNode();
  GE_ASSERT_NOTNULL(parent_node, "The node %s is not on the Root graph", holder_node->GetName().c_str());
  if (strcmp(parent_node->GetType().c_str(), GetExecuteGraphTypeStr(ExecuteGraphType::kInit)) == 0) {
    return holder;
  }
  return nullptr;
}
}  // namespace bg
}  // namespace gert

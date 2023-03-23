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

#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/lowering/frame_selector.h"

#include <deque>
#include <stack>

#include <securec.h>
#include <cstdint>
#include "graph/debug/ge_log.h"
#include "graph/debug/ge_op_types.h"
#include "graph/utils/graph_utils.h"
#include "common/util/mem_utils.h"
#include "graph/utils/node_utils.h"
#include "common/checker.h"

#include "exe_graph/lowering/exe_graph_attrs.h"
#include "exe_graph/runtime/execute_graph_types.h"
#include "graph/debug/ge_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/def_types.h"
namespace gert {
namespace bg {
namespace {
constexpr char const *kNetOutput = "NetOutput";
constexpr char const *kInnerNetOutput = "InnerNetOutput";
thread_local std::deque<std::unique_ptr<GraphFrame>> graph_frames;
thread_local GraphFrame *current_frame;
bool IsGraphOutType(const char *node_type) {
  return strcmp(kNetOutput, node_type) == 0 || strcmp(kInnerNetOutput, node_type) == 0;
}
ge::OpDescPtr CreateOpDesc(const std::string &node_name, const char *node_type, size_t in_count, size_t out_count) {
  auto op_desc = ge::MakeShared<ge::OpDesc>(node_name, node_type);
  GE_ASSERT_NOTNULL(op_desc);
  for (size_t i = 0; i < in_count; ++i) {
    if (op_desc->AddInputDesc(ge::GeTensorDesc()) != ge::GRAPH_SUCCESS) {
      GE_LOGE("Failed to create OpDesc for node %s, io-count %zu/%zu, add input desc %zu failed ", node_name.c_str(),
              in_count, out_count, i);
      return nullptr;
    }
  }
  for (size_t i = 0; i < out_count; ++i) {
    if (op_desc->AddOutputDesc(ge::GeTensorDesc()) != ge::GRAPH_SUCCESS) {
      GE_LOGE("Failed to create OpDesc for node %s, io-count %zu/%zu, add output desc %zu failed ", node_name.c_str(),
              in_count, out_count, i);
      return nullptr;
    }
  }
  return op_desc;
}
struct ConnectionPathPoint {
  GraphFrame *frame;
  ge::NodePtr node;
};
ge::InDataAnchorPtr EnsureHasDataEdge(const ge::NodePtr &src, int32_t src_index, const ConnectionPathPoint &point) {
  auto src_anchor = src->GetOutDataAnchor(src_index);
  GE_ASSERT_NOTNULL(src_anchor);

  for (const auto &dst_anchor : src_anchor->GetPeerInDataAnchors()) {
    GE_ASSERT_NOTNULL(dst_anchor);
    auto dst_node = dst_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(dst_node);
    if (dst_node == point.node) {
      return dst_anchor;
    }
  }

  auto index = point.node->GetAllInDataAnchorsSize();
  GE_ASSERT_SUCCESS(ge::NodeUtils::AppendInputAnchor(point.node, index + 1));

  auto dst_anchor = point.node->GetInDataAnchor(static_cast<int32_t>(index));
  GE_ASSERT_NOTNULL(dst_anchor);
  src_anchor->LinkTo(dst_anchor);

  return dst_anchor;
}
ge::NodePtr EnsureHasData(const ConnectionPathPoint &point, int32_t index) {
  for (const auto &node : point.frame->GetExeGraph()->GetDirectNode()) {
    if (node->GetType() != "InnerData") {
      continue;
    }
    int32_t data_index = -1;
    GE_ASSERT_TRUE(ge::AttrUtils::GetInt(node->GetOpDesc(), ge::ATTR_NAME_INDEX, data_index));
    if (data_index == index) {
      return node;
    }
  }

  auto data = ValueHolder::AddNode("InnerData", 0, 1, *point.frame);
  GE_ASSERT_NOTNULL(data);
  GE_ASSERT_TRUE(ge::AttrUtils::SetInt(data->GetOpDesc(), ge::ATTR_NAME_INDEX, index));
  return data;
}
ge::OutDataAnchorPtr ConnectFromParents(ge::NodePtr src, int32_t src_index, const ge::NodePtr &dst) {
  std::stack<ConnectionPathPoint> connect_path;
  auto frame_iter = graph_frames.rbegin();
  for (; frame_iter != graph_frames.rend(); ++frame_iter) {
    if ((*frame_iter)->GetExeGraph() == dst->GetOwnerComputeGraph()) {
      break;
    }
  }

  auto next_graph = dst->GetOwnerComputeGraph();
  bool full_path = false;
  for (; frame_iter != graph_frames.rend(); ++frame_iter) {
    auto graph = (*frame_iter)->GetExeGraph();
    GE_ASSERT_NOTNULL(graph);
    if (graph != next_graph) {
      continue;
    }
    if (graph == src->GetOwnerComputeGraph()) {
      full_path = true;
      break;
    }
    auto parent_node = graph->GetParentNode();
    if (parent_node == nullptr) {
      // log out of loop scope
      break;
    }
    next_graph = parent_node->GetOwnerComputeGraph();
    connect_path.push({frame_iter->get(), std::move(parent_node)});
  }

  if (!full_path) {
    GE_LOGE("Failed to connect from %s index %d to node %s, the src node does not on the graph or on its parent graphs",
            src->GetName().c_str(), src_index, dst->GetName().c_str());
    return nullptr;
  }

  while (!connect_path.empty()) {
    auto point = std::move(connect_path.top());
    connect_path.pop();

    auto dst_anchor = EnsureHasDataEdge(src, src_index, point);
    GE_ASSERT_NOTNULL(dst_anchor);

    auto data_node = EnsureHasData(point, dst_anchor->GetIdx());
    GE_ASSERT_NOTNULL(data_node);

    src = data_node;
    src_index = 0;
  }

  return src->GetOutDataAnchor(src_index);
}
ge::graphStatus AddDataEdge(const ge::NodePtr &src, int32_t src_index, const ge::NodePtr &dst, int32_t dst_index) {
  auto src_anchor = ConnectFromParents(src, src_index, dst);
  if (src_anchor == nullptr) {
    GE_LOGE("Failed to connect from %s(%d) to %s(%d), connect from parents failed", src->GetName().c_str(), src_index,
            dst->GetName().c_str(), dst_index);
    return ge::GRAPH_FAILED;
  }
  auto ret = ge::GraphUtils::AddEdge(src_anchor, dst->GetInDataAnchor(dst_index));
  if (ret != ge::GRAPH_SUCCESS) {
    GELOGE(ret, "Failed to connect edge from %s:%d to %s:%d, error code %u", src->GetName().c_str(), src_index,
           dst->GetName().c_str(), dst_index, ret);
  }
  return ret;
}

HyperStatus AddDependencyBetweenNodes(const ge::Node &src, const ge::Node &dst) {
  if (src.GetOwnerComputeGraph() != dst.GetOwnerComputeGraph()) {
    return HyperStatus::ErrorStatus("The source node %s(%s) and dst node %s(%s) does not on the same graph",
                                    src.GetName().c_str(), src.GetType().c_str(), dst.GetName().c_str(),
                                    dst.GetType().c_str());
  }
  if (ge::GraphUtils::AddEdge(src.GetOutControlAnchor(), dst.GetInControlAnchor()) != ge::GRAPH_SUCCESS) {
    return HyperStatus::ErrorStatus("Failed to add control edge from %s to %s", src.GetName().c_str(),
                                    dst.GetName().c_str());
  }
  return HyperStatus::Success();
}
ge::graphStatus AddDependencyToGuarder(const ge::Node &src, const ge::Node &guarder) {
  auto guarder_graph = guarder.GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(guarder_graph);
  const ge::Node *current_node = &src;
  while (current_node->GetOwnerComputeGraph() != guarder_graph) {
    auto owner_graph = current_node->GetOwnerComputeGraph();
    GE_ASSERT_NOTNULL(owner_graph);
    auto parent_node = owner_graph->GetParentNode();
    GE_ASSERT_NOTNULL(parent_node,
                      "Failed to add dependency from node %s(%s) to guarder %s(%s), the guarder node does not on the "
                      "same graph or the parent graphs of the source node",
                      src.GetName().c_str(), src.GetType().c_str(), guarder.GetName().c_str(),
                      guarder.GetType().c_str());
    current_node = parent_node.get();
  }
  GE_ASSERT_HYPER_SUCCESS(AddDependencyBetweenNodes(*current_node, guarder));
  return ge::GRAPH_SUCCESS;
}
ge::NodePtr GetComputeNodeByIndex(const GraphFrame *frame, size_t index) {
  auto &indexes_to_node = frame->GetIndexesToNode();
  GE_ASSERT_TRUE(indexes_to_node.size() > index, "The current compute node index %zu out of range", index);
  return indexes_to_node[index];
}
}  // namespace
std::atomic<int64_t> ValueHolder::id_generator_{0};
ValueHolder::~ValueHolder() = default;

ValueHolder::ValueHolder()
    : id_(id_generator_++), type_(ValueHolderType::kValueHolderTypeEnd), index_(0), placement_(0) {}

bool ValueHolder::IsOk() const noexcept {
  return error_msg_ == nullptr;
}
ValueHolder::ValueHolderType ValueHolder::GetType() const noexcept {
  return type_;
}
const ge::Node *ValueHolder::GetNode() const noexcept {
  return node_.get();
}
int32_t ValueHolder::GetOutIndex() const noexcept {
  return index_;
}
int64_t ValueHolder::GetId() const noexcept {
  return id_;
}
const ValueHolder::GraphHolder *ValueHolder::GetGraph() const noexcept {
  return graph_.get();
}
ValueHolderPtr ValueHolder::CreateError(const char *fmt, va_list arg) {
  auto value_holder = std::shared_ptr<ValueHolder>(new (std::nothrow) ValueHolder());
  GE_ASSERT_NOTNULL(value_holder);
  value_holder->error_msg_ = std::unique_ptr<char[]>(CreateMessage(fmt, arg));
  return value_holder;
}
ValueHolderPtr ValueHolder::CreateError(const char *fmt, ...) {
  va_list arg;
  va_start(arg, fmt);
  auto holder = CreateError(fmt, arg);
  va_end(arg);
  return holder;
}
std::string ValueHolder::GenerateNodeName(const char *node_type, const GraphFrame &frame) {
  std::stringstream node_name;
  node_name << node_type;
  const auto &current_compute_node = frame.GetCurrentComputeNode();
  if (current_compute_node != nullptr) {
    node_name << '_' << current_compute_node->GetName();
  }
  node_name << '_' << id_generator_++;
  return node_name.str();
}
ValueHolder::NodeHolderPtr ValueHolder::AddNode(const char *node_type, size_t input_count, size_t output_count,
                                                GraphFrame &frame) {
  auto &graph = frame.GetExeGraph();
  GE_ASSERT_NOTNULL(graph);

  auto node = graph->AddNode(CreateOpDesc(GenerateNodeName(node_type, frame), node_type, input_count, output_count));
  GE_ASSERT_NOTNULL(node);

  // add compute node info index
  size_t index;
  if (frame.GetCurrentNodeIndex(index)) {
    if (!ge::AttrUtils::SetInt(node->GetOpDesc(), kComputeNodeIndex, static_cast<int64_t>(index))) {
      GE_LOGE("Failed to add node %s, add ComputeNodeIndex failed", node_type);
      return nullptr;
    }
  }

  return node;
}
ValueHolder::NodeHolderPtr ValueHolder::CreateNode(const char *node_type, const std::vector<ValueHolderPtr> &inputs,
                                                   size_t out_count) {
  auto frame = GetCurrentFrame();
  if (frame == nullptr) {
    GE_LOGE("The current frame does not exists, "
            "the function ValueHolder::PushGraphFrame should be called before construct the graph");
    return nullptr;
  }
  auto node = ValueHolder::AddNode(node_type, inputs.size(), out_count, *frame);

  /*
   * todo 检查是否有子图向父图连接的场景，这种场景需要报错
   *      父图向子图连接的场景，为父图节点创建一个InnerData
   */
  for (size_t i = 0; i < inputs.size(); ++i) {
    GE_ASSERT_NOTNULL(inputs[i]);
    GE_ASSERT_NOTNULL(inputs[i]->node_);
    GE_ASSERT_SUCCESS(AddDataEdge(inputs[i]->node_, inputs[i]->index_, node, static_cast<int32_t>(i)));
    if (inputs[i]->guarder_ != nullptr && !IsGraphOutType(node_type)) {
      GE_ASSERT_SUCCESS(AddDependencyToGuarder(*node, *(inputs[i]->guarder_->GetNode())));
    }
  }
  return node;
}
ValueHolderPtr ValueHolder::CreateFromNode(ge::NodePtr node, int32_t index, ValueHolderType type) {
  auto holder = std::shared_ptr<ValueHolder>(new (std::nothrow) ValueHolder());
  GE_ASSERT_NOTNULL(holder);

  holder->type_ = type;
  holder->graph_ = node->GetOwnerComputeGraph();
  holder->node_ = std::move(node);
  holder->index_ = index;
  return holder;
}
std::vector<ValueHolderPtr> ValueHolder::CreateFromNode(const NodeHolderPtr &node, size_t out_count) {
  return CreateFromNode(node, 0, out_count);
}
std::vector<ValueHolderPtr> ValueHolder::CreateFromNode(const NodeHolderPtr &node, size_t start_index,
                                                        size_t out_count) {
  std::vector<ValueHolderPtr> holders;
  for (size_t i = 0; i < out_count; ++i) {
    holders.emplace_back(CreateFromNode(node, static_cast<int32_t>(i + start_index), ValueHolderType::kOutput));
  }

  return holders;
}
std::vector<ValueHolderPtr> ValueHolder::CreateDataOutput(const char *node_type,
                                                          const std::vector<ValueHolderPtr> &inputs, size_t out_count) {
  auto node = CreateNode(node_type, inputs, out_count);
  if (node == nullptr) {
    return {out_count, nullptr};
  }
  return CreateFromNode(node, out_count);
}
ValueHolderPtr ValueHolder::CreateVoid(const char *node_type, const std::vector<ValueHolderPtr> &inputs) {
  auto node = CreateNode(node_type, inputs, 0);
  GE_ASSERT_NOTNULL(node);
  return CreateFromNode(node, -1, ValueHolderType::kOutput);
}
/**
 * @param data const数据
 * @param size const数据的长度
 * @param is_string 此const是否是个字符串, todo: 当前对string支持的不好
 * @return
 */
ValueHolderPtr ValueHolder::CreateConst(const void *data, size_t size, bool is_string) {
  auto node = ValueHolder::CreateNode("Const", {}, 1);
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_SUCCESS(node->GetOpDesc()->SetAttr("is_string", ge::AnyValue::CreateFrom(is_string)));
  GE_ASSERT_TRUE(ge::AttrUtils::SetZeroCopyBytes(node->GetOpDesc(), kConstValue,
                                                 ge::Buffer::CopyFrom(ge::PtrToPtr<void, uint8_t>(data), size)));
  return CreateFromNode(node, 0, ValueHolderType::kConst);
}
ValueHolderPtr ValueHolder::CreateFeed(int64_t index) {
  auto node = ValueHolder::CreateNode("Data", {}, 1U);
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_TRUE(ge::AttrUtils::SetInt(node->GetOpDesc(), kFeedIndex, index));
  return CreateFromNode(node, 0, ValueHolderType::kFeed);
}

ValueHolderPtr ValueHolder::CreateSingleDataOutput(const char *node_type, const std::vector<ValueHolderPtr> &inputs) {
  auto holders = CreateDataOutput(node_type, inputs, 1U);
  if (holders.empty()) {
    return nullptr;
  }
  return holders[0];
}
HyperStatus ValueHolder::AddDependency(const ValueHolderPtr &src, const ValueHolderPtr &dst) {
  if (src == nullptr || src->GetNode() == nullptr) {
    return HyperStatus::ErrorStatus("Failed to add control ege, because the src does not have a node.");
  }
  if (dst == nullptr || dst->GetNode() == nullptr) {
    return HyperStatus::ErrorStatus("Failed to add control ege, because the dst does not have a node.");
  }
  return AddDependencyBetweenNodes(*(src->GetNode()), *(dst->GetNode()));
}

GraphFrame *ValueHolder::PushGraphFrame() {
  if (!graph_frames.empty()) {
    GELOGE(ge::INTERNAL_ERROR,
           "Failed to push root graph frame, if you want to push a non-root graph frame, specify which ValueHolder the "
           "graph frame belongs and the ir name.");
    return nullptr;
  }
  auto graph = ge::MakeShared<ge::ComputeGraph>("ROOT");
  GE_ASSERT_NOTNULL(graph);
  auto frame = new (std::nothrow) GraphFrame(graph);
  GE_ASSERT_NOTNULL(frame);
  graph_frames.emplace_back(frame);
  return graph_frames.back().get();
}
GraphFrame *ValueHolder::PushGraphFrame(const ValueHolderPtr &belongs, const char *graph_name) {
  GE_ASSERT_NOTNULL(belongs);
  GE_ASSERT_NOTNULL(belongs->GetNode());
  GE_ASSERT_NOTNULL(graph_name);
  if (graph_frames.empty()) {
    GELOGE(ge::INTERNAL_ERROR, "Failed to push a non-root graph frame, there is no root graph frames exists");
    return nullptr;
  }
  auto &parent_frame = *graph_frames.back();
  auto instance_name = GenerateNodeName(graph_name, parent_frame);
  auto graph = ge::MakeShared<ge::ComputeGraph>(instance_name);
  GE_ASSERT_NOTNULL(graph);

  auto frame_holder = ge::ComGraphMakeUnique<GraphFrame>(graph, parent_frame);
  GE_ASSERT_NOTNULL(frame_holder);

  int64_t compute_node_index;
  if (ge::AttrUtils::GetInt(belongs->GetNode()->GetOpDesc(), kComputeNodeIndex, compute_node_index)) {
    auto compute_node = GetComputeNodeByIndex(frame_holder.get(), static_cast<size_t>(compute_node_index));
    if (compute_node != nullptr) {
      frame_holder->SetCurrentComputeNode(compute_node);
    }
  }

  GE_ASSERT_SUCCESS(ge::NodeUtils::AddSubgraph(*const_cast<ge::Node *>(belongs->GetNode()), graph_name, graph));

  auto frame = frame_holder.release();
  graph_frames.emplace_back(frame);
  return graph_frames.back().get();
}
std::unique_ptr<GraphFrame> ValueHolder::PopGraphFrame() {
  if (graph_frames.empty()) {
    return nullptr;
  }
  auto ret = std::move(graph_frames.back());
  graph_frames.pop_back();
  return ret;
}
GraphFrame *ValueHolder::GetCurrentFrame() {
  if (current_frame != nullptr) {
    return current_frame;
  }
  if (graph_frames.empty()) {
    return nullptr;
  }
  return graph_frames.back().get();
}
void ValueHolder::SetCurrentComputeNode(const ge::NodePtr &node) {
  auto frame = GetCurrentFrame();
  if (frame == nullptr) {
    GELOGW("Ignore to add current compute node, the current frame is nullptr");
    return;
  }
  frame->SetCurrentComputeNode(node);
}
void ValueHolder::AddRelevantInputNode(const ge::NodePtr &node) {
  auto frame = GetCurrentFrame();
  if (frame == nullptr) {
    GELOGW("Ignore to add relevant input node, the current frame is nullptr");
  } else {
    frame->AddRelevantInputNode(node);
  }
}
std::unique_ptr<ValueHolder::CurrentComputeNodeGuarder> ValueHolder::SetScopedCurrentComputeNode(  //
    const ge::NodePtr &node) {
  auto frame = GetCurrentFrame();
  GE_ASSERT_NOTNULL(frame);

  auto guarder = ge::ComGraphMakeUnique<CurrentComputeNodeGuarder>(frame->GetCurrentComputeNode());
  GE_ASSERT_NOTNULL(guarder);
  frame->SetCurrentComputeNode(node);
  return guarder;
}
ValueHolder::GraphHolder *ValueHolder::GetCurrentGraph() {
  auto frame = GetCurrentFrame();
  GE_ASSERT_NOTNULL(frame);
  return frame->GetExeGraph().get();
}
ge::graphStatus ValueHolder::RefFrom(const ValueHolderPtr &other) {
  GE_ASSERT_NOTNULL(node_);
  GE_ASSERT_NOTNULL(other);
  GE_ASSERT_NOTNULL(other->node_);

  if (index_ < 0 || other->index_ < 0) {
    GELOGE(ge::PARAM_INVALID, "Invalid index to ref %d -> %d", index_, other->index_);
    return ge::PARAM_INVALID;
  }

  auto td = node_->GetOpDesc()->MutableOutputDesc(index_);
  GE_ASSERT_NOTNULL(td);

  GE_ASSERT_TRUE(ge::AttrUtils::SetStr(td, kRefFromNode, other->GetNode()->GetName()));
  GE_ASSERT_TRUE(ge::AttrUtils::SetInt(td, kRefFromIndex, other->index_));
  return ge::GRAPH_SUCCESS;
}
ValueHolderPtr ValueHolder::CreateVoidGuarder(const char *node_type, const ValueHolderPtr &resource,
                                              const std::vector<ValueHolderPtr> &args) {
  std::vector<ValueHolderPtr> inputs;
  inputs.reserve(args.size() + 1);
  inputs.emplace_back(resource);
  inputs.insert(inputs.cend(), args.cbegin(), args.cend());
  auto ret = CreateVoid(node_type, inputs);
  GE_ASSERT_NOTNULL(ret);
  GE_ASSERT_TRUE(ge::AttrUtils::SetInt(ret->GetNode()->GetOpDesc(), kReleaseResourceIndex, 0));
  resource->guarder_ = ret;
  return ret;
}
const int32_t &ValueHolder::GetPlacement() const {
  return placement_;
}
void ValueHolder::SetPlacement(const int32_t &placement) {
  placement_ = placement;
}
void ValueHolder::ReleaseAfter(const ValueHolderPtr &other) {
  if (guarder_ == nullptr) {
    GELOGW("Current holder from node %s  index %d does not has a guarder", node_->GetName().c_str(), index_);
    return;
  }
  AddDependency(other, guarder_);
}
std::vector<ValueHolderPtr> ValueHolder::AppendOutputs(size_t append_count) {
  auto node = node_->shared_from_this();
  auto start_index = node->GetAllOutDataAnchorsSize();
  auto ret = ge::NodeUtils::AppendOutputAnchor(node, start_index + append_count);
  if (ret != ge::GRAPH_SUCCESS) {
    return {};
  }
  return CreateFromNode(node, start_index, append_count);
}
std::unique_ptr<GraphFrame> ValueHolder::PopGraphFrame(const std::vector<ValueHolderPtr> &outputs,
                                                       const std::vector<ValueHolderPtr> &targets) {
  const char *node_type = kNetOutput;
  if (graph_frames.size() > 1U) {
    // The NetOutput type means "Network outputs", subgraph use InnerNetOutput as output type
    node_type = kInnerNetOutput;
  }
  return PopGraphFrame(outputs, targets, node_type);
}
std::unique_ptr<GraphFrame> ValueHolder::PopGraphFrame(const std::vector<ValueHolderPtr> &outputs,
                                                       const std::vector<ValueHolderPtr> &targets,
                                                       const char *node_type) {
  GE_ASSERT_NOTNULL(node_type);
  auto out_holder = CreateVoid(node_type, outputs);
  GE_ASSERT_NOTNULL(out_holder);
  if (strcmp(ge::NETOUTPUT, node_type) == 0) {
    // the name of NetOutput node must be `NetOutput`
    out_holder->GetNode()->GetOpDesc()->SetName(node_type);
  }

  for (const auto &target : targets) {
    AddDependency(target, out_holder);
  }
  return PopGraphFrame();
}

std::vector<ValueHolderPtr> FrameSelector::OnMainRoot(const std::function<std::vector<ValueHolderPtr>()> &builder) {
  if (builder == nullptr || graph_frames.empty()) {
    return {};
  }
  std::vector<ValueHolderPtr> outputs;
  if (OnMainRoot(builder, outputs) != ge::GRAPH_SUCCESS) {
    GELOGW("Compatible mode, the air code is not the newest.");
    current_frame = graph_frames.front().get();
    outputs = builder();
    current_frame = nullptr;
  }
  return outputs;
}
ge::graphStatus FrameSelector::OnMainRoot(const std::function<std::vector<ValueHolderPtr>()> &builder,
                                          std::vector<ValueHolderPtr> &outputs) {
  GE_ASSERT_NOTNULL(builder, "Failed to do frame selection, the builder is nullptr");
  // 栈底是root-frame，向上是main-frame，因此栈中至少有两个元素
  GE_ASSERT_TRUE(graph_frames.size() > 1U, "Failed to do frame selection, there is no main-frame exists");

  auto root_frame = graph_frames.begin()->get();
  GE_ASSERT_NOTNULL(root_frame, "Failed to find the root frame");
  auto main_frame = (graph_frames.begin() + 1)->get();
  GE_ASSERT_NOTNULL(main_frame, "Failed to find the main frame");

  // check if the main_frame is correct
  auto root_graph = root_frame->GetExeGraph();
  GE_ASSERT_NOTNULL(root_graph, "Failed to find the root graph");
  auto main_node = root_graph->FindFirstNodeMatchType(GetExecuteGraphTypeStr(ExecuteGraphType::kMain));
  GE_ASSERT_NOTNULL(main_node, "Failed to find the main node");
  auto main_graph = ge::NodeUtils::GetSubgraph(*main_node, 0U);
  GE_ASSERT_TRUE(main_graph == main_frame->GetExeGraph(), "Failed to find the main frame");

  current_frame = main_frame;
  outputs = builder();
  current_frame = nullptr;
  return ge::GRAPH_SUCCESS;
}

std::vector<ValueHolderPtr> ValueHolder::GetLastExecNodes() {
  if (graph_frames.empty()) {
    return {};
  }
  auto frame = graph_frames.begin()->get();
  if (graph_frames.size() > 1U) {
    frame = (graph_frames.begin() + 1)->get();
  }
  return frame->GetLastExecNodes();
}

ValueHolderPtr FrameSelector::OnMainRootLast(const std::function<bg::ValueHolderPtr()> &builder) {
  if (builder == nullptr || graph_frames.empty()) {
    return nullptr;
  }
  current_frame = graph_frames.begin()->get();
  if (graph_frames.size() > 1U) {
    current_frame = (graph_frames.begin() + 1)->get();
  }
  auto output = builder();
  current_frame->SetLastExecNode(output);
  current_frame = nullptr;
  return output;
}
}  // namespace bg
}  // namespace gert

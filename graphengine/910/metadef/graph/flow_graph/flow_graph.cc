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

#include "flow_graph/flow_graph.h"
#include "common/checker.h"
#include "common/util/mem_utils.h"
#include "debug/ge_util.h"
#include "graph/flow_graph/data_flow_attr_define.h"
#include "graph/flow_graph/data_flow_utils.h"
#include "graph/flow_graph/flow_attr_util.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_desc_utils.h"
#include "proto/dflow.pb.h"

namespace ge {
namespace dflow {
using ComputeGraphPtr = std::shared_ptr<ComputeGraph>;

FlowOperator::FlowOperator(const char *name, const char *type) : ge::Operator(name, type) {}
FlowOperator::~FlowOperator() = default;

FlowData::FlowData(const char *name, int64_t index) : FlowOperator(name, "Data") {
  ge::Operator::InputRegister("x", "TensorType::ALL()");
  ge::Operator::OutputRegister("y", "TensorType::ALL()");
  ge::Operator::AttrRegister("index", index);
}
FlowData::~FlowData() = default;

class FlowNodeImpl {
public:
  explicit FlowNodeImpl(OpDescPtr op_desc, uint32_t input_num, uint32_t output_num)
      : op_desc_(op_desc), input_num_(input_num), output_num_(output_num) {}
  ~FlowNodeImpl() = default;
  graphStatus MapInput(uint32_t node_input_index, const ProcessPoint &pp, uint32_t pp_input_index,
                       const std::vector<DataFlowInputAttr> &attrs = {});
  graphStatus MapOutput(uint32_t node_output_index, const ProcessPoint &pp, uint32_t pp_output_index);
  graphStatus AddPp(const ProcessPoint &pp);

 private:
  graphStatus AddInEdges(uint32_t node_input_index, const ProcessPoint &pp, uint32_t pp_input_index);
  graphStatus AddOutEdges(uint32_t node_output_index, const ProcessPoint &pp, uint32_t pp_output_index);
  OpDescPtr op_desc_;
  uint32_t input_num_;
  uint32_t output_num_;
  // key1 : pp_name, key2: pp_input_index; value : node_input_index;
  std::map<std::string, std::map<uint32_t, uint32_t>> in_edges_;
  // key1 : pp_name, key2: pp_output_index; value : node_output_index;
  std::map<std::string, std::map<uint32_t, uint32_t>> out_edges_;
  std::map<std::string, bool> added_pps_;
};

graphStatus FlowNodeImpl::AddInEdges(uint32_t node_input_index, const ProcessPoint &pp, uint32_t pp_input_index) {
  std::vector<std::string> pps;
  auto flow_node_name = op_desc_->GetName();
  GE_ASSERT_TRUE(ge::AttrUtils::GetListStr(op_desc_, ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pps));

  dataflow::ProcessPoint process_point;
  for (std::string &pp_str : pps) {
    GE_ASSERT_TRUE(process_point.ParseFromString(pp_str));
    if (process_point.name() != pp.GetProcessPointName()) {
      GELOGD("current pp(%s) is skipped for it's not equal to MapInput pp name(%s).",
             process_point.name().c_str(), pp.GetProcessPointName());
      continue;
    }
    // duplicate check
    if ((pp_input_index < static_cast<uint32_t>(process_point.in_edges_size())) &&
       (process_point.in_edges(pp_input_index).node_name() != "")) {
      GELOGE(GRAPH_FAILED, "pp name(%s) has duplicate map input index(%u).", pp.GetProcessPointName(), pp_input_index);
      return ge::GRAPH_FAILED;
    }

    process_point.add_in_edges();
    if (pp_input_index < static_cast<uint32_t>(process_point.in_edges_size())) {
      auto in_edge = process_point.mutable_in_edges(pp_input_index);
      in_edge->set_node_name(flow_node_name.c_str());
      in_edge->set_index(node_input_index);
      GELOGI("add pp(%s) input index(%u) map node(%s) index(%u).", pp.GetProcessPointName(), pp_input_index,
             flow_node_name.c_str(), node_input_index);
    } else {
      in_edges_[pp.GetProcessPointName()][pp_input_index] = node_input_index;
    }

    for (auto it = in_edges_[pp.GetProcessPointName()].begin(); it != in_edges_[pp.GetProcessPointName()].end();) {
      if (static_cast<int32_t>(it->first) < process_point.in_edges_size()) {
        auto in_edge = process_point.mutable_in_edges(it->first);
        in_edge->set_node_name(flow_node_name.c_str());
        in_edge->set_index(it->second);
        GELOGI("add pp(%s) input index(%u) map node(%s) index(%u).", pp.GetProcessPointName(), it->first,
               flow_node_name.c_str(), it->second);
        in_edges_[pp.GetProcessPointName()].erase(it++);
      } else {
        it++;
      }
    }
    process_point.SerializeToString(&pp_str);
  }

  GE_ASSERT_TRUE(ge::AttrUtils::SetListStr(op_desc_, ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pps));
  return ge::GRAPH_SUCCESS;
}

graphStatus FlowNodeImpl::AddOutEdges(uint32_t node_output_index, const ProcessPoint &pp, uint32_t pp_output_index) {
  std::vector<std::string> pps;
  auto name = op_desc_->GetName();
  GE_ASSERT_TRUE(ge::AttrUtils::GetListStr(op_desc_, ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pps));

  dataflow::ProcessPoint process_point;
  for (std::string &pp_str : pps) {
    GE_ASSERT_TRUE(process_point.ParseFromString(pp_str));
    if (process_point.name() != pp.GetProcessPointName()) {
      GELOGD("current pp(%s) is skipped for it's not equal to MapInput pp name(%s)",
             process_point.name().c_str(), pp.GetProcessPointName());
      continue;
    }
    // duplicate check
    if ((pp_output_index < static_cast<uint32_t>(process_point.out_edges_size())) &&
        (process_point.out_edges(pp_output_index).node_name() != "")) {
      GELOGE(GRAPH_FAILED, "pp name(%s) has duplicate map input index(%u).", pp.GetProcessPointName(), pp_output_index);
      return ge::GRAPH_FAILED;
    }

    process_point.add_out_edges();
    if (pp_output_index < static_cast<uint32_t>(process_point.out_edges_size())) {
      auto out_edge = process_point.mutable_out_edges(pp_output_index);
      out_edge->set_node_name(name.c_str());
      out_edge->set_index(node_output_index);
      GELOGI("add pp(%s) output index(%u) map node(%s) index(%u)", pp.GetProcessPointName(), pp_output_index,
             name.c_str(), node_output_index);
    } else {
      out_edges_[pp.GetProcessPointName()][pp_output_index] = node_output_index;
    }
    // proc back in_edges
    for (auto it = out_edges_[pp.GetProcessPointName()].begin(); it != out_edges_[pp.GetProcessPointName()].end();) {
      if (static_cast<int32_t>(it->first) < process_point.out_edges_size()) {
        auto out_edge = process_point.mutable_out_edges(it->first);
        out_edge->set_node_name(name.c_str());
        out_edge->set_index(it->second);
        GELOGI("add pp(%s) output index(%u) map node(%s) index(%u)", pp.GetProcessPointName(), it->first,
               name.c_str(), it->second);
        out_edges_[pp.GetProcessPointName()].erase(it++);
      } else {
        it++;
      }
    }
    process_point.SerializeToString(&pp_str);
  }

  GE_ASSERT_TRUE(ge::AttrUtils::SetListStr(op_desc_, ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pps));
  return ge::GRAPH_SUCCESS;
}

graphStatus FlowNodeImpl::MapInput(uint32_t node_input_index, const ProcessPoint &pp, uint32_t pp_input_index,
                                   const std::vector<DataFlowInputAttr> &attrs) {
  if (pp.GetProcessPointName() == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "The process point name is nullptr.");
    return GRAPH_PARAM_INVALID;
  }
  auto flow_node_name = op_desc_->GetName();
  if (node_input_index >= input_num_) {
    GELOGE(GRAPH_PARAM_INVALID, "invalid node(%s) input index[%u]. valid range is [0, %u)", flow_node_name.c_str(),
           node_input_index, input_num_);
    return GRAPH_PARAM_INVALID;
  }
  if (!added_pps_[pp.GetProcessPointName()]) {
    GELOGE(GRAPH_PARAM_INVALID, "Please add pp[%s] to node(%s) first.", pp.GetProcessPointName(),
           flow_node_name.c_str());
    return GRAPH_PARAM_INVALID;
  }

  auto input_tensor_desc = op_desc_->MutableInputDesc(node_input_index);
  if (input_tensor_desc == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] Node(%s)'s input(%u) tensor desc is nullptr.", flow_node_name.c_str(),
           node_input_index);
    return GRAPH_PARAM_INVALID;
  }
  const auto ret = FlowAttrUtil::SetAttrsToTensorDesc(attrs, input_tensor_desc);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(ret, "Failed to set attrs to node(%s)'s input(%u) tensor desc.", flow_node_name.c_str(), node_input_index);
    return ret;
  }
  return AddInEdges(node_input_index, pp, pp_input_index);
}

graphStatus FlowNodeImpl::MapOutput(uint32_t node_output_index, const ProcessPoint &pp, uint32_t pp_output_index) {
  if (pp.GetProcessPointName() == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "The process point name is nullptr.");
    return GRAPH_PARAM_INVALID;
  }
  auto flow_node_name = op_desc_->GetName();
  if (node_output_index >= output_num_) {
    GELOGE(GRAPH_PARAM_INVALID, "invalid node(%s) output index[%u]. valid range is [0, %u)", flow_node_name.c_str(),
           node_output_index, output_num_);
    return GRAPH_PARAM_INVALID;
  }

  if (!added_pps_[pp.GetProcessPointName()]) {
    GELOGE(GRAPH_PARAM_INVALID, "Please add pp[%s] to node(%s) first.", pp.GetProcessPointName(),
           flow_node_name.c_str());
    return GRAPH_PARAM_INVALID;
  }

  return AddOutEdges(node_output_index, pp, pp_output_index);
}

graphStatus FlowNodeImpl::AddPp(const ProcessPoint &pp) {
  auto flow_node_name = op_desc_->GetName();
  if (added_pps_[pp.GetProcessPointName()]) {
    GELOGI("Process point(%s) has been added to node[%s].", pp.GetProcessPointName(), flow_node_name.c_str());
    return GRAPH_SUCCESS;
  }

  std::vector<std::string> pp_attrs;
  (void)ge::AttrUtils::GetListStr(op_desc_, ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp_attrs);
  ge::AscendString target_str;
  pp.Serialize(target_str);
  pp_attrs.emplace_back(target_str.GetString(), target_str.GetLength());
  GE_ASSERT_TRUE(ge::AttrUtils::SetListStr(op_desc_, ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp_attrs),
                 "Failed to set attr[%s] to node[%s].", ATTR_NAME_DATA_FLOW_PROCESS_POINTS, flow_node_name.c_str());
  added_pps_[pp.GetProcessPointName()] = true;
  return GRAPH_SUCCESS;
}

FlowNode::FlowNode(const char *name, uint32_t input_num, uint32_t output_num) : FlowOperator(name, "FlowNode") {
  ge::Operator::DynamicInputRegister(ATTR_NAME_DATA_FLOW_INPUT, input_num);
  ge::Operator::DynamicOutputRegister(ATTR_NAME_DATA_FLOW_OUTPUT, output_num);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(*this);
  if (op_desc == nullptr) {
    GELOGE(GRAPH_FAILED, "get flow node op desc failed, name=%s.", (name == nullptr) ? "nullptr" : name);
  } else {
    impl_ = MakeShared<FlowNodeImpl>(op_desc, input_num, output_num);
    if (impl_ == nullptr) {
      GELOGE(GRAPH_FAILED, "FlowNode make shared failed.");
    }
  }
}

FlowNode::~FlowNode() = default;

FlowNode &FlowNode::SetInput(uint32_t dst_index, const FlowOperator &src_op, uint32_t src_index) {
  ge::Operator::SetInput(dst_index, src_op, src_index);
  return *this;
}

FlowNode &FlowNode::MapInput(uint32_t node_input_index, const ProcessPoint &pp, uint32_t pp_input_index,
                             const std::vector<DataFlowInputAttr> &attrs) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] MapInput:FlowNodeImpl is nullptr, check failed.");
    REPORT_INNER_ERROR("E18888", "MapInput failed: FlowNode can not be used, impl is nullptr.");
    return *this;
  }
  if (impl_->MapInput(node_input_index, pp, pp_input_index, attrs) != GRAPH_SUCCESS) {
    REPORT_INNER_ERROR("E18888", "MapInput failed.");
  }
  return *this;
}

FlowNode &FlowNode::MapOutput(uint32_t node_output_index, const ProcessPoint &pp, uint32_t pp_output_index) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] MapOutput:FlowNodeImpl is nullptr, check failed.");
    REPORT_INNER_ERROR("E18888", "MapOutput failed: FlowNode can not be used, impl is nullptr.");
    return *this;
  }
  if (impl_->MapOutput(node_output_index, pp, pp_output_index) != GRAPH_SUCCESS) {
    REPORT_INNER_ERROR("E18888", "MapOutput failed.");
  }
  return *this;
}

FlowNode &FlowNode::AddPp(const ProcessPoint &pp) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] FlowNodeImpl is nullptr, check failed.");
    REPORT_INNER_ERROR("E18888", "AddPp failed: FlowNode can not be used, impl is nullptr.");
    return *this;
  }

  if (pp.GetProcessPointType() == ProcessPointType::FUNCTION) {
    const FunctionPp *function_pp = dynamic_cast<const FunctionPp *>(&pp);
    if (function_pp == nullptr) {
      GELOGE(GRAPH_FAILED, "ProcessPoint(%s) cast failed.", pp.GetProcessPointName());
      REPORT_INNER_ERROR("E18888", "AddPp failed: ProcessPoint(%s) cast failed.", pp.GetProcessPointName());
      return *this;
    }

    auto invoked_closures = function_pp->GetInvokedClosures();
    if (invoked_closures.empty()) {
      (void) impl_->AddPp(pp);
      return *this;
    }
    this->SubgraphRegister(pp.GetProcessPointName(), true);
    this->SubgraphCountRegister(pp.GetProcessPointName(), invoked_closures.size());
    uint32_t i = 0;
    for (auto iter = invoked_closures.cbegin(); iter != invoked_closures.cend(); ++iter) {
      const auto &graph_pp = iter->second;
      auto flow_graph_builder = [graph_pp]() {
        Graph graph;
        (void) DataFlowUtils::BuildInvokedGraphFromGraphPp(graph_pp, graph);
        return graph;
      };
      this->SetSubgraphBuilder(pp.GetProcessPointName(), i++, flow_graph_builder);
    }
  } else if (pp.GetProcessPointType() == ProcessPointType::GRAPH) {
    const GraphPp *graph_pp = dynamic_cast<const GraphPp *>(&pp);
    if (graph_pp == nullptr) {
      GELOGE(GRAPH_FAILED, "ProcessPoint(%s) cast failed.", pp.GetProcessPointName());
      REPORT_INNER_ERROR("E18888", "AddPp failed: ProcessPoint(%s) cast failed.", pp.GetProcessPointName());
      return *this;
    }

    this->SubgraphRegister(pp.GetProcessPointName(), false);
    this->SubgraphCountRegister(pp.GetProcessPointName(), 1);
    GraphBuilder builder = graph_pp->GetGraphBuilder();
    if (builder == nullptr) {
      GELOGE(GRAPH_FAILED, "GraphPp(%s)'s graph builder is nullptr.", graph_pp->GetProcessPointName());
      REPORT_INNER_ERROR("E18888", "AddPp failed: GraphPp(%s)'s graph builder is nullptr.",
                         graph_pp->GetProcessPointName());
      return *this;
    }
    this->SetSubgraphBuilder(pp.GetProcessPointName(), 0, builder);
  } else {
    GELOGE(GRAPH_FAILED, "process point type[%u] is invalid.", static_cast<uint32_t>(pp.GetProcessPointType()));
    REPORT_INNER_ERROR("E18888", "AddPp failed: Process point type[%u] is invalid.",
                       static_cast<uint32_t>(pp.GetProcessPointType()));
    return *this;
  }

  (void) impl_->AddPp(pp);
  return *this;
}

class FlowGraphImpl {
public:
  explicit FlowGraphImpl(const char *name) : name_(name), graph_(Graph(name)) {}
  ~FlowGraphImpl() = default;

  const ge::Graph &ToGeGraph() const {
    return graph_;
  }

  void SetInputs(const std::vector<FlowOperator> &inputs) {
    std::vector<ge::Operator> op_inputs;
    for (auto iter = inputs.cbegin(); iter != inputs.cend(); ++iter) {
      op_inputs.emplace_back(*iter);
    }

    (void)graph_.SetInputs(op_inputs);
    const auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph_);
    AttrUtils::SetBool(compute_graph, ATTR_NAME_IS_DATA_FLOW_GRAPH, true);
    return;
  }

  void SetOutputs(const std::vector<FlowOperator> &outputs) {
    std::vector<ge::Operator> op_outputs;
    for (auto iter = outputs.cbegin(); iter != outputs.cend(); ++iter) {
      op_outputs.emplace_back(*iter);
    }

    (void)graph_.SetOutputs(op_outputs);
    return;
  }

  const char *GetName() const {
    return name_.c_str();
  }
private:
  const std::string name_;
  ge::Graph graph_;
};

FlowGraph::FlowGraph(const char *name) {
  if (name != nullptr) {
    impl_ = ComGraphMakeShared<FlowGraphImpl>(name);
    if (impl_ == nullptr) {
      GELOGE(GRAPH_FAILED, "FlowGraphImpl make shared failed.");
    }
  } else {
    impl_ = nullptr;
    GELOGE(GRAPH_FAILED, "Input graph name is nullptr.");
  }
}
FlowGraph::~FlowGraph() = default;

const ge::Graph &FlowGraph::ToGeGraph() const {
  if (impl_ == nullptr) {
    static ge::Graph graph;
    GELOGE(GRAPH_FAILED, "ToGeGraph failed: graph can not be used, impl is nullptr.");
    REPORT_INNER_ERROR("E18888", "ToGeGraph failed: graph can not be used, impl is nullptr.");
    return graph;
  }

  return impl_->ToGeGraph();
}

FlowGraph &FlowGraph::SetInputs(const std::vector<FlowOperator> &inputs) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "SetInputs failed: graph can not be used, impl is nullptr.");
    REPORT_INNER_ERROR("E18888", "SetInputs failed: graph can not be used, impl is nullptr.");
    return *this;
  }

  if (inputs.empty()) {
    GELOGE(GRAPH_FAILED, "SetInputs failed: input operator size can not be 0.");
    REPORT_INNER_ERROR("E18888", "SetInputs failed: input operator size can not be 0.");
    return *this;
  }

  impl_->SetInputs(inputs);
  return *this;
}

FlowGraph &FlowGraph::SetOutputs(const std::vector<FlowOperator> &outputs) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "SetOutputs failed: graph can not be used, impl is nullptr.");
    REPORT_INNER_ERROR("E18888", "SetOutputs failed: graph can not be used, impl is nullptr.");
    return *this;
  }

  if (outputs.empty()) {
    GELOGE(GRAPH_FAILED, "SetOutputs failed: outputs operator size can not be 0.");
    REPORT_INNER_ERROR("E18888", "SetOutputs failed: outputs operator size can not be 0.");
    return *this;
  }

  impl_->SetOutputs(outputs);
  const std::string err_msg = ErrorManager::GetInstance().GetErrorMessage();
  if (!err_msg.empty()) {
    std::cout << err_msg << std::endl;
  }
  return *this;
}

const char *FlowGraph::GetName() const {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "GetName failed: graph can not be used, impl is nullptr.");
    return nullptr;
  }

  return impl_->GetName();
}
}  // namespace dflow
}  // namespace ge
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

#include "flow_graph/data_flow_utils.h"
#include "common/ge_common/debug/ge_log.h"

namespace ge {
namespace dflow {
graphStatus DataFlowUtils::BuildInvokedGraphFromGraphPp(const GraphPp &graph_pp, Graph &graph) {
  const auto pp_name = graph_pp.GetProcessPointName();
  GraphBuilder builder = graph_pp.GetGraphBuilder();
  if (builder == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "GraphPp(%s)'s graph builder is nullptr.",
           ((pp_name == nullptr) ? "nullptr" : pp_name));
    return GRAPH_PARAM_INVALID;
  }
  std::string graph_name = pp_name;
  graph_name += "_invoked";
  // add one FlowData and one FlowNode
  FlowGraph flow_graph(graph_name.c_str());
  auto flow_data = FlowData("flow_data", 0);
  // 1 input and 1 output
  auto flow_node = FlowNode("flow_node", 1, 1).SetInput(0, flow_data).AddPp(graph_pp);
  graph = flow_graph.SetInputs({flow_data}).SetOutputs({flow_node}).ToGeGraph();
  return GRAPH_SUCCESS;
}
}  // namespace dflow
}  // namespace ge
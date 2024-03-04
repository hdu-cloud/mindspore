/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "graph/operator_factory_impl.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_builder_utils.h"
#include "graph/op_desc.h"
#include "proto/dflow.pb.h"
#include "gtest/gtest.h"

using namespace ge::dflow;

namespace ge {
class DataFlowUtilsUTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    backup_operator_creators_ = OperatorFactoryImpl::operator_creators_;
    RegisterOpCreator("Data", {}, {"y"});
  }
  static void TearDownTestCase() {
    OperatorFactoryImpl::operator_creators_ = std::move(backup_operator_creators_);
  }
  void SetUp() {}
  void TearDown() {}

  static void RegisterOpCreator(const std::string &op_type, const std::vector<std::string> &input_names,
                                const std::vector<std::string> &output_names) {
    auto op_creator = [op_type, input_names, output_names](const std::string &name) -> Operator {
      auto op_desc = std::make_shared<OpDesc>(name, op_type);
      for (const auto &tensor_name : input_names) {
        op_desc->AddInputDesc(tensor_name, {});
      }
      for (const auto &tensor_name : output_names) {
        op_desc->AddOutputDesc(tensor_name, {});
      }
      return OpDescUtils::CreateOperatorFromOpDesc(op_desc);
    };
    OperatorFactoryImpl::RegisterOperatorCreator(op_type, op_creator);
  }
  static std::shared_ptr<std::map<std::string, OpCreator>> backup_operator_creators_;
};
std::shared_ptr<std::map<std::string, OpCreator>> DataFlowUtilsUTest::backup_operator_creators_;

TEST_F(DataFlowUtilsUTest, BuildInvokedGraphFromGraphPp_Success) {
  GraphBuilder graph_build = []() {
    ut::GraphBuilder builder = ut::GraphBuilder("subgraph");
    auto data = builder.AddNode("Data", "Data", 0, 1);
    auto transdata = builder.AddNode("Transdata", "Transdata", 1, 1);
    auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
    builder.AddDataEdge(data, 0, transdata, 0);
    builder.AddDataEdge(transdata, 0, netoutput, 0);
    ComputeGraphPtr cgp = builder.GetGraph();
    Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);
    return graph;
  };
  auto graph_pp = GraphPp("graph_pp", graph_build).SetCompileConfig("./graph_pp.json");
  Graph graph;
  DataFlowUtils::BuildInvokedGraphFromGraphPp(graph_pp, graph);
  ASSERT_EQ(graph.GetName(), "graph_pp_invoked");
  ASSERT_EQ(graph.GetDirectNode().size(), 2);
  bool has_flow_node = false;
  for (const auto &node : graph.GetDirectNode()) {
    AscendString type;
    node.GetType(type);
    std::string type_str(type.GetString());
    if (type_str == "FlowNode") {
      has_flow_node = true;
      std::vector<AscendString> pp_attrs;
      ASSERT_EQ(node.GetAttr("_dflow_process_points", pp_attrs), ge::GRAPH_SUCCESS);
      ASSERT_EQ(pp_attrs.size(), 1);
      auto process_point = dataflow::ProcessPoint();
      auto flag = process_point.ParseFromString(pp_attrs[0].GetString());
      ASSERT_TRUE(flag);
      ASSERT_EQ(process_point.name(), "graph_pp");
      ASSERT_EQ(process_point.type(), dataflow::ProcessPoint_ProcessPointType_GRAPH);
      ASSERT_EQ(process_point.compile_cfg_file(), "./graph_pp.json");
      ASSERT_EQ(process_point.funcs_size(), 0);
      ASSERT_EQ(process_point.graphs_size(), 1);
      ASSERT_EQ(process_point.invoke_pps_size(), 0);
      ASSERT_EQ(process_point.in_edges_size(), 0);
      ASSERT_EQ(process_point.out_edges_size(), 0);
    }
  }
  ASSERT_EQ(has_flow_node, true);
}

TEST_F(DataFlowUtilsUTest, BuildInvokedGraphFromGraphPp_Failed) {
  auto graph_pp = GraphPp(nullptr, nullptr);
  Graph graph;
  auto ret = DataFlowUtils::BuildInvokedGraphFromGraphPp(graph_pp, graph);
  ASSERT_EQ(ret, GRAPH_PARAM_INVALID);
  graph_pp = GraphPp("graph_pp", nullptr);
  ret = DataFlowUtils::BuildInvokedGraphFromGraphPp(graph_pp, graph);
  ASSERT_EQ(ret, GRAPH_PARAM_INVALID);
}
}  // namespace ge
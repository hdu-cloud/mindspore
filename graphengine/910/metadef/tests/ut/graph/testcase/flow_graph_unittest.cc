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
#include <gtest/gtest.h>
#include "flow_graph/data_flow.h"
#include "proto/dflow.pb.h"
#include "graph/utils/op_desc_utils.h"
#define protected public
#define private public
#include "inc/common/util/error_manager/error_manager.h"
#define protected public
#define private public

using namespace ge::dflow;

namespace ge {
class FlowGraphUTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(FlowGraphUTest, DflowFuncBasicTest_AddPp) {
  auto data0 = FlowData("Data0", 0);
  auto data1 = FlowData("Data1", 1);
  auto data2 = FlowData("Data2", 2);
  ge::Graph graph("user_graph");
  GraphBuilder graph_build = [graph]() { return graph; };
  auto pp1 = GraphPp("pp1", graph_build).SetCompileConfig("./pp1.json");
  auto node0 = FlowNode("node0", 3, 2).SetInput(0, data0).SetInput(1, data1).SetInput(2, data2).AddPp(pp1).AddPp(pp1);

  std::vector<std::string> pp_attrs;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(node0);
  ge::AttrUtils::GetListStr(op_desc, "_dflow_process_points", pp_attrs);
  auto process_point = dataflow::ProcessPoint();
  auto flag = process_point.ParseFromString(pp_attrs[0]);
  ASSERT_TRUE(flag);
  ASSERT_EQ(process_point.name(), "pp1");
  ASSERT_EQ(process_point.type(), dataflow::ProcessPoint_ProcessPointType_GRAPH);
  ASSERT_EQ(process_point.compile_cfg_file(), "./pp1.json");
  ASSERT_EQ(process_point.funcs_size(), 0);
  ASSERT_EQ(process_point.graphs_size(), 1);
  ASSERT_EQ(process_point.invoke_pps_size(), 0);
  ASSERT_EQ(process_point.in_edges_size(), 0);
  ASSERT_EQ(process_point.out_edges_size(), 0);
}

TEST_F(FlowGraphUTest, DflowFuncBasicTest_Map) {
  auto data0 = FlowData("Data0", 0);
  auto data1 = FlowData("Data1", 1);
  auto data2 = FlowData("Data2", 2);
  auto pp1 = FunctionPp("pp1").SetCompileConfig("./pp1.json");
  auto pp2 = FunctionPp("pp2");
  auto node0 = FlowNode("node0", 4, 3)
                   .SetInput(0, data0)
                   .SetInput(1, data1)
                   .SetInput(2, data2)
                   .AddPp(pp1)
                   .MapInput(0, pp1, 2)
                   .MapInput(1, pp1, 1)
                   .MapInput(2, pp1, 0)
                   .MapInput(3, pp2, 0)
                   .MapInput(10, pp2, 0)
                   .MapOutput(0, pp1, 1)
                   .MapOutput(1, pp1, 0)
                   .MapOutput(2, pp2, 0)
                   .MapOutput(10, pp2, 0);

  std::vector<std::string> pp_attrs;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(node0);
  ge::AttrUtils::GetListStr(op_desc, "_dflow_process_points", pp_attrs);
  auto process_point = dataflow::ProcessPoint();
  auto flag = process_point.ParseFromString(pp_attrs[0]);
  ASSERT_TRUE(flag);
  ASSERT_EQ(process_point.name(), "pp1");
  ASSERT_EQ(process_point.type(), dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  ASSERT_EQ(process_point.compile_cfg_file(), "./pp1.json");
  ASSERT_EQ(process_point.funcs_size(), 0);
  ASSERT_EQ(process_point.graphs_size(), 0);
  ASSERT_EQ(process_point.invoke_pps_size(), 0);

  // in_edges check
  ASSERT_EQ(process_point.in_edges_size(), 3);
  auto in_edges0 = process_point.in_edges(0);
  ASSERT_EQ(in_edges0.node_name(), "node0");
  ASSERT_EQ(in_edges0.index(), 2);
  auto in_edges1 = process_point.in_edges(1);
  ASSERT_EQ(in_edges1.node_name(), "node0");
  ASSERT_EQ(in_edges1.index(), 1);
  auto in_edges2 = process_point.in_edges(2);
  ASSERT_EQ(in_edges2.node_name(), "node0");
  ASSERT_EQ(in_edges2.index(), 0);

  // out_edges check
  ASSERT_EQ(process_point.out_edges_size(), 2);
  auto out_edges0 = process_point.out_edges(0);
  ASSERT_EQ(out_edges0.node_name(), "node0");
  ASSERT_EQ(out_edges0.index(), 1);
  auto out_edges1 = process_point.out_edges(1);
  ASSERT_EQ(out_edges1.node_name(), "node0");
  ASSERT_EQ(out_edges1.index(), 0);

  FlowGraph flow_graph("flow_graph");
  std::vector<FlowOperator> inputs_operator{data0, data1, data2};
  std::vector<FlowOperator> outputs_operator{node0};
  std::vector<FlowOperator> empty_flow_ops;
  flow_graph.SetInputs(inputs_operator).SetOutputs(outputs_operator);
  ASSERT_EQ(strcmp(flow_graph.GetName(), "flow_graph"), 0);
  ASSERT_EQ(flow_graph.ToGeGraph().GetName(), "flow_graph");

  FlowGraph flow_graph2(nullptr);
  flow_graph2.SetInputs(empty_flow_ops).SetOutputs(empty_flow_ops);
  ASSERT_EQ(flow_graph2.GetName(), nullptr);

  FlowGraph flow_graph3("flow_graph");
  flow_graph3.SetInputs(empty_flow_ops).SetOutputs(empty_flow_ops);
  ASSERT_EQ(flow_graph3.ToGeGraph().GetName(), "flow_graph");
}

TEST_F(FlowGraphUTest, DflowInvokePp) {
  auto data0 = FlowData("Data0", 0);
  auto data1 = FlowData("Data1", 1);
  auto data2 = FlowData("Data2", 2);

  ge::Graph ge_graph("ge_graph");
  GraphBuilder graph_build = [ge_graph]() { return ge_graph; };
  GraphBuilder graph_build2 = []() { return ge::Graph("ge_graph2"); };
  auto graphPp1 = GraphPp("graphPp_1", graph_build).SetCompileConfig("./graph.json");
  auto graphPp2 = GraphPp("graphPp_2", graph_build2).SetCompileConfig("./graph2.json");
  auto pp1 = FunctionPp("pp1")
                 .SetCompileConfig("./pp1.json")
                 .AddInvokedClosure("graph1", graphPp1)
                 .AddInvokedClosure("graph2", graphPp2);
  auto node0 = FlowNode("node0", 3, 2).SetInput(0, data0).SetInput(1, data1).SetInput(2, data2).AddPp(pp1);
  std::vector<std::string> pp_attrs;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(node0);
  ge::AttrUtils::GetListStr(op_desc, "_dflow_process_points", pp_attrs);
  auto process_point = dataflow::ProcessPoint();
  auto flag = process_point.ParseFromString(pp_attrs[0]);
  ASSERT_TRUE(flag);
  ASSERT_EQ(process_point.name(), "pp1");
  ASSERT_EQ(process_point.type(), dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  ASSERT_EQ(process_point.compile_cfg_file(), "./pp1.json");
  ASSERT_EQ(process_point.invoke_pps_size(), 2);

  auto invoke_pps = process_point.invoke_pps();
  auto invoke_pp0 = invoke_pps["graph1"];
  ASSERT_EQ(invoke_pp0.name(), "graphPp_1");
  ASSERT_EQ(invoke_pp0.type(), dataflow::ProcessPoint_ProcessPointType_GRAPH);
  ASSERT_EQ(invoke_pp0.compile_cfg_file(), "./graph.json");
  ASSERT_EQ(invoke_pp0.graphs(0), "graphPp_1");

  auto invoke_pp1 = invoke_pps["graph2"];
  ASSERT_EQ(invoke_pp1.name(), "graphPp_2");
  ASSERT_EQ(invoke_pp1.type(), dataflow::ProcessPoint_ProcessPointType_GRAPH);
  ASSERT_EQ(invoke_pp1.compile_cfg_file(), "./graph2.json");
  ASSERT_EQ(invoke_pp1.graphs(0), "graphPp_2");
}

TEST_F(FlowGraphUTest, TestPrintErrMsg) {
  auto &instance = ErrorManager::GetInstance();
  instance.is_init_ = true;
  auto data0 = FlowData("Data0", 0);
  auto flow_node = FlowNode("FlowNode", 2, 1);
  flow_node.SetInput(2, data0);
  auto flow_graph = FlowGraph("FlowGraph");
  flow_graph.SetInputs({data0}).SetOutputs({flow_node});
  ASSERT_EQ(flow_graph.ToGeGraph().IsValid(), false);
}

TEST_F(FlowGraphUTest, MapInputAndMapOutputFailed) {
  auto data0 = FlowData("Data0", 0);
  auto data1 = FlowData("Data1", 1);
  auto data2 = FlowData("Data2", 2);
  auto pp1 = FunctionPp("pp1").SetCompileConfig("./pp1.json");
  auto node0 = FlowNode("node0", 3, 2)
                   .SetInput(0, data0)
                   .SetInput(1, data1)
                   .SetInput(2, data2)
                   .AddPp(pp1)
                   .MapInput(0, pp1, 0)
                   .MapInput(1, pp1, 0)
                   .MapInput(2, pp1, 0)
                   .MapOutput(0, pp1, 0)
                   .MapOutput(1, pp1, 0)
                   .MapOutput(2, pp1, 0);
  std::vector<std::string> pp_attrs;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(node0);
  ge::AttrUtils::GetListStr(op_desc, "_dflow_process_points", pp_attrs);
  dataflow::ProcessPoint process_point;
  auto flag = process_point.ParseFromString(pp_attrs[0]);
  ASSERT_TRUE(flag);
}

TEST_F(FlowGraphUTest, FlowNode_FlowNodeImpl_nullptr) {
  auto pp = FunctionPp("func_pp");
  auto node = FlowNode(nullptr, 0, 0).AddPp(pp).MapInput(0, pp, 0).MapOutput(0, pp, 0);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(node);
  std::vector<std::string> pp_attrs;
  ASSERT_FALSE(ge::AttrUtils::GetListStr(op_desc, "_dflow_process_points", pp_attrs));
}

namespace {
class StubProcessPoint : public ProcessPoint {
 public:
  StubProcessPoint(const char_t *name, ProcessPointType type) : ProcessPoint(name, type) {}
  void Serialize(ge::AscendString &str) const override {
    return;
  }
};
}  // namespace

TEST_F(FlowGraphUTest, FlowNode_Invalid_Pp) {
  auto pp = FunctionPp(nullptr);
  auto node = FlowNode("node", 1, 1).AddPp(pp).MapInput(0, pp, 0).MapOutput(0, pp, 0);
  auto stub_pp = StubProcessPoint("stub_pp", ProcessPointType::FUNCTION);
  node.AddPp(stub_pp);
  stub_pp = StubProcessPoint("stub_pp", ProcessPointType::GRAPH);
  node.AddPp(stub_pp);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(node);
  std::vector<std::string> pp_attrs;
  ASSERT_FALSE(ge::AttrUtils::GetListStr(op_desc, "_dflow_process_points", pp_attrs));
}

TEST_F(FlowGraphUTest, FlowNode_MapInput_Failed) {
  auto pp = FunctionPp("pp");
  auto node = FlowNode("node", 1, 1).AddPp(pp);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(node);
  DataFlowInputAttr attr{DataFlowAttrType::INVALID, nullptr};
  node.MapInput(0, pp, 0, {attr});
  auto input_desc = op_desc->MutableInputDesc(0);
  input_desc->SetDataType(DT_UNDEFINED);
  input_desc->SetFormat(FORMAT_RESERVED);
  node.MapInput(0, pp, 0);
  std::vector<std::string> pp_attrs;
  ASSERT_TRUE(ge::AttrUtils::GetListStr(op_desc, "_dflow_process_points", pp_attrs));
  dataflow::ProcessPoint process_point;
  auto flag = process_point.ParseFromString(pp_attrs[0]);
  ASSERT_TRUE(flag);
  ASSERT_EQ(process_point.in_edges_size(), 0);
}

TEST_F(FlowGraphUTest, FlowNode_AddPp_Failed) {
  auto pp = GraphPp("graphpp", nullptr);
  auto node = FlowNode("node", 1, 1).AddPp(pp);
  std::vector<std::string> pp_attrs;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(node);
  ASSERT_FALSE(ge::AttrUtils::GetListStr(op_desc, "_dflow_process_points", pp_attrs));
}

TEST_F(FlowGraphUTest, FlowGraph_FlowGraphImpl_nullptr) {
  auto data0 = FlowData("Data0", 0);
  auto flow_node = FlowNode("FlowNode", 2, 1);
  flow_node.SetInput(2, data0);
  auto flow_graph = FlowGraph(nullptr);
  flow_graph.SetInputs({data0}).SetOutputs({flow_node});
  ASSERT_EQ(flow_graph.ToGeGraph().IsValid(), false);
}
}  // namespace ge
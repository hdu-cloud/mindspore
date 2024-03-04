/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "graph/ge_local_context.h"
#include "graph/ge_context.h"

#define protected public
#define private public
#include "graph/compute_graph.h"
#include "graph/compute_graph_impl.h"
#include "graph/op_desc.h"
#include "graph/op_desc_impl.h"
#include "graph/ge_tensor.h"
#include "graph/utils/ge_ir_utils.h"
#include "graph_builder_utils.h"
#include "common/ge_common/ge_types.h"
#undef private
#undef protected
#include "debug/ge_op_types.h"
#include "inc/graph/debug/ge_attr_define.h"
#include "graph/utils/transformer_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils_ex.h"

namespace {
/*
 *   netoutput1
 *       |
 *      add
 *     /   \
 * data1   data2
 */
ge::ComputeGraphPtr BuildSubGraph(const std::string &name) {
  ge::ut::GraphBuilder builder(name);
  auto data1 = builder.AddNode(name + "data1", "Data", 1, 1);
  auto data2 = builder.AddNode(name + "data2", "Data", 1, 1);
  auto add = builder.AddNode(name + "sub", "Sub", 2, 1);
  auto netoutput = builder.AddNode(name + "_netoutput", "NetOutput", 1, 1);

  ge::AttrUtils::SetInt(data1->GetOpDesc(), "_parent_node_index", static_cast<int>(0));
  ge::AttrUtils::SetInt(data2->GetOpDesc(), "_parent_node_index", static_cast<int>(1));
  ge::AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", static_cast<int>(0));

  builder.AddDataEdge(data1, 0, add, 0);
  builder.AddDataEdge(data2, 0, add, 1);
  builder.AddDataEdge(add, 0, netoutput, 0);

  return builder.GetGraph();
}
/*
 *   netoutput
 *       |
 *      if
 *     /   \
 * data1   data2
 */
ge::ComputeGraphPtr BuildMainGraphWithIf(const std::string &name) {
  ge::ut::GraphBuilder builder(name);
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto if1 = builder.AddNode("if", "If", 2, 1);
  auto netoutput1 = builder.AddNode("netoutput", "NetOutput", 1, 1);

  builder.AddDataEdge(data1, 0, if1, 0);
  builder.AddDataEdge(data2, 0, if1, 1);
  builder.AddDataEdge(if1, 0, netoutput1, 0);

  auto main_graph = builder.GetGraph();

  auto sub1 = BuildSubGraph("sub1");
  sub1->SetParentGraph(main_graph);
  sub1->SetParentNode(main_graph->FindNode("if"));
  main_graph->FindNode("if")->GetOpDesc()->AddSubgraphName("sub1");
  main_graph->FindNode("if")->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");
  main_graph->AddSubgraph("sub1", sub1);

  auto sub2 = BuildSubGraph("sub2");
  sub2->SetParentGraph(main_graph);
  sub2->SetParentNode(main_graph->FindNode("if"));
  main_graph->FindNode("if")->GetOpDesc()->AddSubgraphName("sub2");
  main_graph->FindNode("if")->GetOpDesc()->SetSubgraphInstanceName(1, "sub2");
  main_graph->AddSubgraph("sub2", sub2);

  return main_graph;
}
/*
 *          netoutput
 *         |    \    \
 *       node4 node5 node6
 *       |      \
 *     node2  node3
 *      \    /
 *      node1
 */
ge::ComputeGraphPtr BuildNormalGraph(const std::string &name) {
  auto builder = ge::ut::GraphBuilder(name);
  const auto &node1 = builder.AddNode("node1", "node1", 0, 2);
  const auto &node2 = builder.AddNode("node2", "node2", 1, 1);
  const auto &node3 = builder.AddNode("node3", "node3", 1, 1);
  const auto &node4 = builder.AddNode("node4", "node4", 1, 1);
  const auto &node5 = builder.AddNode("node5", "node5", 1, 1);
  const auto &node6 = builder.AddNode("node6", "node6", 0, 1);
  const auto &netoutput = builder.AddNode("netoutput", "netoutput", 3, 1);

  builder.AddDataEdge(node1, 0, node2, 0);
  builder.AddDataEdge(node1, 1, node3, 0);
  builder.AddDataEdge(node2, 0, node4, 0);
  builder.AddDataEdge(node3, 0, node5, 0);
  builder.AddDataEdge(node4, 0, netoutput, 0);
  builder.AddDataEdge(node5, 0, netoutput, 1);
  builder.AddDataEdge(node6, 0, netoutput, 2);

  builder.AddControlEdge(node1, node2);
  builder.AddControlEdge(node1, node3);
  builder.AddControlEdge(node2, node4);
  builder.AddControlEdge(node3, node5);
  builder.AddControlEdge(node4, netoutput);
  builder.AddControlEdge(node5, netoutput);
  builder.AddControlEdge(node6, netoutput);
  return builder.GetGraph();
}

/*
 *         variable      data
 *          /     \        |
 *       node1  node2     node3
 *         |       |       |
 *         |       |     node4
 *          \      |      /
 *                node5
 */
ge::ComputeGraphPtr BuildDelayTopoGraph(const std::string &name) {
  auto builder = ge::ut::GraphBuilder(name);
  const auto &variable = builder.AddNode("variable", ge::VARIABLE, 0, 2);
  const auto &node1 = builder.AddNode("node1", "node1", 1, 1);
  const auto &node2 = builder.AddNode("node2", "node2", 1, 1);
  const auto &node3 = builder.AddNode("node3", "node3", 1, 1);
  const auto &node4 = builder.AddNode("node4", "node4", 1, 1);
  const auto &node5 = builder.AddNode("node5", "node5", 3, 0);
  const auto &data = builder.AddNode("data", "DATA", 0, 1);

  builder.AddDataEdge(variable, 0, node1, 0);
  builder.AddDataEdge(variable, 1, node2, 0);
  builder.AddDataEdge(node1, 0, node5, 0);
  builder.AddDataEdge(node2, 0, node5, 1);
  builder.AddDataEdge(data, 0, node3, 0);
  builder.AddDataEdge(node3, 0, node4, 0);
  builder.AddDataEdge(node4, 0, node5, 2);

  builder.AddControlEdge(node2, node3);
  return builder.GetGraph();
}

/*
 *  constant const variable      data
 *          \  |  /     \        |
 *           node1    node2     node3
 *              |        |       |
 *              |        |     node4
 *               \       |      /
 *                     node5
 */
ge::ComputeGraphPtr BuildDelayTopoGraphMultiInput(const std::string &name, bool all_is_log_life = true) {
  auto builder = ge::ut::GraphBuilder(name);
  const auto &constant = builder.AddNode("const", ge::CONSTANT, 0, 1);
  auto type = ge::CONSTANTOP;
  if (!all_is_log_life) {
    type = "test";
  }
  const auto &constantop = builder.AddNode("constant", type, 0, 1);
  const auto &variable = builder.AddNode("variable", ge::VARIABLE, 0, 2);
  const auto &node1 = builder.AddNode("node1", "node1", 3, 1);
  const auto &node2 = builder.AddNode("node2", "node2", 1, 1);
  const auto &node3 = builder.AddNode("node3", "node3", 1, 1);
  const auto &node4 = builder.AddNode("node4", "node4", 1, 1);
  const auto &node5 = builder.AddNode("node5", "node5", 3, 0);
  const auto &data = builder.AddNode("data", "DATA", 0, 1);

  builder.AddDataEdge(constant, 0, node1, 0);
  builder.AddDataEdge(constantop, 0, node1, 1);
  builder.AddDataEdge(variable, 0, node1, 2);
  builder.AddDataEdge(variable, 1, node2, 0);
  builder.AddDataEdge(node1, 0, node5, 0);
  builder.AddDataEdge(node2, 0, node5, 1);
  builder.AddDataEdge(data, 0, node3, 0);
  builder.AddDataEdge(node3, 0, node4, 0);
  builder.AddDataEdge(node4, 0, node5, 2);

  builder.AddControlEdge(node2, node3);
  return builder.GetGraph();
}
}

namespace ge
{
  class UtestComputeGraph : public testing::Test {
    protected:
    void SetUp() override {}
    void TearDown() override {}
  };

TEST_F(UtestComputeGraph, GetAllNodes_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetDataType(DT_FLOAT);
  tensor_desc->SetFormat(FORMAT_CHWN);

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddInputDesc(tensor_desc->Clone());
  graph->AddNode(op_desc);
  graph->AddNode(op_desc);

  auto node_filter = [](const Node &node){ return true;};
  auto graph_filter = [](const Node &node, const char *str, const ComputeGraphPtr &graph){ return true;};
  auto out_nodes = graph->GetAllNodes(node_filter, graph_filter);
  EXPECT_EQ(out_nodes.size(), 2);
}

TEST_F(UtestComputeGraph, GetNodes_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetDataType(DT_FLOAT);
  tensor_desc->SetFormat(FORMAT_CHWN);

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddInputDesc(tensor_desc->Clone());
  graph->AddNode(op_desc);
  graph->AddNode(op_desc);
  auto node_filter = [](const Node &node){ return true;};
  auto graph_filter = [](const Node &node, const char *str, const ComputeGraphPtr &graph){ return true;};
  auto out_nodes = graph->GetNodes(true, node_filter, graph_filter);
  EXPECT_EQ(out_nodes.size(), 2);
}

TEST_F(UtestComputeGraph, AddNodeFront_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetDataType(DT_FLOAT);
  tensor_desc->SetFormat(FORMAT_CHWN);

  auto op_desc = std::make_shared<OpDesc>("node1", "node1");
  op_desc->AddInputDesc(tensor_desc->Clone());
  auto node = graph->AddNode(op_desc);

  auto op_desc1 = std::make_shared<OpDesc>("add_front", "add_front");
  op_desc1->AddInputDesc(tensor_desc->Clone());
  auto nodeptr = graph->AddNodeFront(node);
  EXPECT_EQ(node, nodeptr);
}

TEST_F(UtestComputeGraph, RemoveNode_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetDataType(DT_FLOAT);
  tensor_desc->SetFormat(FORMAT_CHWN);
  auto op_desc = std::make_shared<OpDesc>("node1", "node1");
  op_desc->AddInputDesc(tensor_desc->Clone());
  auto node = graph->AddNode(op_desc);

  EXPECT_EQ(graph->RemoveNode(node), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, GraphMembersAreEqual_success) {
  auto graph1 = std::make_shared<ComputeGraph>("graph");
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetDataType(DT_FLOAT);
  tensor_desc->SetFormat(FORMAT_CHWN);

  auto op_desc = std::make_shared<OpDesc>("node1", "node1");
  op_desc->AddInputDesc(tensor_desc->Clone());
  graph1->AddNode(op_desc);
  graph1->AddNode(op_desc);

  auto graph2 = std::make_shared<ComputeGraph>("graph");
  graph2->AddNode(op_desc);
  EXPECT_EQ(graph1->GraphMembersAreEqual(*(graph2)), false);
  graph2->AddNode(op_desc);
  EXPECT_EQ(graph1->GraphMembersAreEqual(*(graph2)), true);
}

TEST_F(UtestComputeGraph, GraphAttrsAreEqual_success) {
  auto graph1 = std::make_shared<ComputeGraph>("graph1");

  int64_t val = 0;
  AnyValue anyvalue;
  anyvalue.SetValue(val);
  graph1->SetAttr("test", anyvalue);

  auto graph2 = std::make_shared<ComputeGraph>("graph2");
  EXPECT_EQ(graph1->GraphAttrsAreEqual(*(graph2)), false);

  graph2->SetAttr("test", anyvalue);
  EXPECT_EQ(graph1->GraphAttrsAreEqual(*(graph2)), true);
}

TEST_F(UtestComputeGraph, VectorInputNodePtrIsEqual_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetDataType(DT_FLOAT);
  tensor_desc->SetFormat(FORMAT_CHWN);

  auto op_desc = std::make_shared<OpDesc>("node1", "node1");
  op_desc->AddInputDesc(tensor_desc->Clone());
  auto node = graph->AddNode(op_desc);

  std::vector<NodePtr> leftnodes{node};
  std::vector<NodePtr> rightnodes{node};
  EXPECT_EQ(graph->VectorInputNodePtrIsEqual(leftnodes, rightnodes), true);
  rightnodes.push_back(node);
  EXPECT_EQ(graph->VectorInputNodePtrIsEqual(leftnodes, rightnodes), false);
}

TEST_F(UtestComputeGraph, RemoveConstInput_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetDataType(DT_FLOAT);
  tensor_desc->SetFormat(FORMAT_CHWN);

  auto op_desc = std::make_shared<OpDesc>("node1", CONSTANT);
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddOutputDesc(tensor_desc->Clone());

  auto node1 = graph->AddNode(op_desc);
  auto node2 = graph->AddNode(op_desc);
  GraphUtils::AddEdge(node1->GetOutControlAnchor(), node2->GetInControlAnchor());
  EXPECT_EQ(graph->RemoveConstInput(node1), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, RemoveSubGraph_success) {
  auto rootgraph = std::make_shared<ComputeGraph>("rootgraph");
  auto subgraph = std::make_shared<ComputeGraph>("subgraph");
  rootgraph->AddSubGraph(subgraph);
  EXPECT_EQ(rootgraph->RemoveSubGraph(subgraph), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, Set_GetShareParamLayer_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  std::map<std::vector<std::string>, std::vector<std::string>> params_share_map{{{"test"},{"test"}}};
  graph->SetShareParamLayer(params_share_map);
  EXPECT_EQ(graph->GetShareParamLayer().size(), 1);
}

TEST_F(UtestComputeGraph, Set_GetGraphOutNodes_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  std::map<std::string, std::vector<int32_t>> out_nodes_map{{"test",{1}}};
  auto opdesc = std::make_shared<OpDesc>();
  graph->SetGraphOutNodes(out_nodes_map);
  EXPECT_EQ(graph->GetGraphOutNodes().size(), 1);
  std::map<std::string, std::vector<int32_t>> append_out_nodes_map{{"test2",{2}}};
  graph->AppendGraphOutNodes(append_out_nodes_map);
  EXPECT_EQ(graph->GetGraphOutNodes().size(), 2);
}

TEST_F(UtestComputeGraph, Set_GetOrigGraph_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto origin_graph = std::make_shared<ComputeGraph>("origin_graph");
  graph->SetOrigGraph(origin_graph);
  EXPECT_NE(graph->GetOrigGraph(), nullptr);
}

TEST_F(UtestComputeGraph, GetOutputSize_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto nodes = std::make_shared<Node>();
  graph->AddOutputNode(nodes);
  EXPECT_EQ(graph->GetOutputSize(), 1);
}

TEST_F(UtestComputeGraph, GetInputSize_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto nodes = std::make_shared<Node>();
  graph->AddInputNode(nodes);
  EXPECT_EQ(graph->GetInputSize(), 1);
}

TEST_F(UtestComputeGraph, Set_GetNeedIteration_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  graph->SetNeedIteration(true);
  EXPECT_EQ(graph->GetNeedIteration(), true);
}

TEST_F(UtestComputeGraph, UpdateInputMapping_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  auto opdesc = std::make_shared<OpDesc>(ATTR_NAME_PARENT_NODE_INDEX, DATA);
  opdesc->AddInputDesc("name1", tensor_desc->Clone());
  opdesc->AddOutputDesc("name2", tensor_desc->Clone());
  auto node = graph->AddNode(opdesc);
  ge::AttrUtils::SetInt(opdesc, ATTR_NAME_PARENT_NODE_INDEX, 1);

  graph->AddInputNode(node);
  std::map<uint32_t, uint32_t> input_mapping{{0,1}};
  EXPECT_EQ(graph->UpdateInputMapping(input_mapping), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, UpdateOutputMapping_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  auto opdesc = std::make_shared<OpDesc>(ATTR_NAME_PARENT_NODE_INDEX, NETOUTPUT);
  opdesc->AddInputDesc("name1", tensor_desc->Clone());
  opdesc->AddOutputDesc("name2", tensor_desc->Clone());
  auto node = graph->AddNode(opdesc);
  ge::AttrUtils::SetInt(opdesc, ATTR_NAME_PARENT_NODE_INDEX, 1);
  graph->AddOutputNode(node);
  std::map<uint32_t, uint32_t> output_mapping{{0,1}};
  EXPECT_EQ(graph->UpdateOutputMapping(output_mapping), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, ReorderEventNodes_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode(ATTR_NAME_PARENT_NODE_INDEX, SEND, 1, 1);
  const auto &node2 = builder.AddNode(ATTR_NAME_PARENT_NODE_INDEX, RECV, 1, 1);
  const auto &node3 = builder.AddNode(ATTR_NAME_PARENT_NODE_INDEX, RECV, 1, 1);
  builder.AddControlEdge(node1, node2);
  builder.AddControlEdge(node3, node1);
  builder.AddControlEdge(node2, node3);
  auto graph = builder.GetGraph();

  EXPECT_EQ(graph->ReorderEventNodes(), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, DFSTopologicalSorting_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 1, 1);
  const auto &node2 = builder.AddNode("node2", "node2", 1, 1);
  const auto &node3 = builder.AddNode("node3", "node3", 1, 1);
  std::vector<NodePtr> vec_nodes{node1, node2, node3};

  builder.AddControlEdge(node1, node2);
  builder.AddControlEdge(node3, node1);

  std::vector<NodePtr> stack{};
  auto graph = builder.GetGraph();
  std::map<NodePtr, uint32_t> map_in_edge_num{};
  EXPECT_EQ(graph->DFSTopologicalSorting(vec_nodes, map_in_edge_num, stack, false),
    GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, BFSTopologicalSorting_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 1, 1);
  const auto &node2 = builder.AddNode("node2", "node2", 1, 1);
  const auto &node3 = builder.AddNode("node3", "node3", 1, 1);
  std::vector<NodePtr> vec_nodes{node1, node2, node3};

  builder.AddControlEdge(node1, node2);
  builder.AddControlEdge(node3, node1);

  std::deque<NodePtr> stack{};
  auto graph = builder.GetGraph();
  std::map<NodePtr, uint32_t> map_in_edge_num{};
  EXPECT_EQ(graph->BFSTopologicalSorting(vec_nodes, map_in_edge_num, stack), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, CollectBreadthOutNode_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 2, 2);
  const auto &node2 = builder.AddNode("node2", "node2", 1, 1);
  const auto &node3 = builder.AddNode("node3", "node3", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  builder.AddDataEdge(node2, 0, node1, 0);
  builder.AddControlEdge(node2, node1);
  builder.AddControlEdge(node1, node3);
  std::map<NodePtr, uint32_t> map_in_edge_num{};
  std::map<std::string, NodePtr> breadth_node_map{};
  auto graph = builder.GetGraph();
  EXPECT_EQ(graph->CollectBreadthOutNode(node1, map_in_edge_num, breadth_node_map), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, TopologicalSorting_success) {
  const auto func = [](const NodePtr &node1, const NodePtr &node2) -> bool { return true; };
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 0, 0);
  const auto &node2 = builder.AddNode("node2", "node2", 0, 0);
  auto graph = builder.GetGraph();
  graph->TopologicalSorting(func);
  EXPECT_EQ(node1->GetOpDesc()->GetId(), 1);
  EXPECT_EQ(node2->GetOpDesc()->GetId(), 0);
}

/*
 *          netoutput
 *         |    \    \
 *       node4 node5 node6
 *       |      \
 *     node2  node3
 *      \    /
 *      node1
 */
TEST_F(UtestComputeGraph, TopologicalSortingMode_success) {
  std::map<std::string, std::string> options_map;

  auto graph = BuildNormalGraph("test_topo_graph");
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  std::vector<std::string> expected_bfs_names = {"node1", "node2", "node3", "node4", "node5", "node6", "netoutput"};
  std::vector<std::string> expected_dfs_names = {"node1", "node3", "node5", "node2", "node4", "node6", "netoutput"};
  std::vector<std::string> bfs_names;
  std::vector<std::string> dfs_names;
  options_map.emplace("ge.topoSortingMode", "0");
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  const auto &graph_bfs_topo = graph->GetAllNodes();
  for (auto &node : graph_bfs_topo) {
    bfs_names.push_back(node->GetName());
  }

  options_map["ge.topoSortingMode"] = "1";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);

  const auto &graph_dfs_topo = graph->GetAllNodes();
  for (auto &node : graph_dfs_topo) {
    dfs_names.push_back(node->GetName());
  }
  options_map["ge.topoSortingMode"] = "2";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);

  EXPECT_EQ(bfs_names, expected_bfs_names);
  EXPECT_EQ(dfs_names, expected_dfs_names);
}

TEST_F(UtestComputeGraph, BFSTopologicalSortingInPriorityMode_success) {
  std::map<std::string, std::string> options_map;

  auto graph = BuildNormalGraph("test_bfs_topo_graph");
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  std::vector<std::string> expected_bfs_names = {"node1", "node2", "node3", "node4", "node5", "node6", "netoutput"};
  std::vector<std::string> bfs_names;
  options_map["ge.topoSortingMode"] = "0";
  options_map["ge.exec.memoryOptimizationPolicy"] = "MemoryPriority";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  const auto &graph_bfs_topo = graph->GetAllNodes();
  for (auto &node : graph_bfs_topo) {
    bfs_names.push_back(node->GetName());
  }

  EXPECT_EQ(bfs_names, expected_bfs_names);
}

TEST_F(UtestComputeGraph, TrainTopologicalSortingInPriorityMode_BFS_success) {
  std::map<std::string, std::string> options_map;

  auto graph = BuildNormalGraph("test_train_topo_graph_bfs");
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  std::vector<std::string> expected_bfs_names = {"node1", "node2", "node3", "node4", "node5", "node6", "netoutput"};
  std::vector<std::string> bfs_names;
  options_map["ge.graphRunMode"] = "1";  // tarin
  options_map["ge.topoSortingMode"] = ""; // no topo sort mode
  options_map["ge.exec.memoryOptimizationPolicy"] = "MemoryPriority";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  const auto &graph_bfs_topo = graph->GetAllNodes();
  for (auto &node : graph_bfs_topo) {
    bfs_names.push_back(node->GetName());
  }

  EXPECT_EQ(bfs_names, expected_bfs_names);
}

TEST_F(UtestComputeGraph, TrainAndInvalidTopologicalSortingInPriorityMode_BFS_success) {
  std::map<std::string, std::string> options_map;

  auto graph = BuildNormalGraph("test_train_topo_graph_with_invalid_sort_mode_bfs");
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  std::vector<std::string> expected_bfs_names = {"node1", "node2", "node3", "node4", "node5", "node6", "netoutput"};
  std::vector<std::string> bfs_names;
  options_map["ge.graphRunMode"] = "1";  // tarin
  options_map["ge.topoSortingMode"] = "10"; // invalid topo sort mode
  options_map["ge.exec.memoryOptimizationPolicy"] = "MemoryPriority";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  const auto &graph_bfs_topo = graph->GetAllNodes();
  for (auto &node : graph_bfs_topo) {
    bfs_names.push_back(node->GetName());
  }

  EXPECT_EQ(bfs_names, expected_bfs_names);
}

TEST_F(UtestComputeGraph, DFSTopologicalSortingInPriorityMode_success) {
  std::map<std::string, std::string> options_map;

  auto graph = BuildNormalGraph("test_dfs_topo_graph");
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  std::vector<std::string> expected_dfs_names = {"node6", "node1", "node2", "node3", "node4", "node5", "netoutput"};
  std::vector<std::string> dfs_names;
  options_map["ge.topoSortingMode"] = "1";
  options_map["ge.exec.memoryOptimizationPolicy"] = "MemoryPriority";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  const auto &graph_dfs_topo = graph->GetAllNodes();
  for (auto &node : graph_dfs_topo) {
    dfs_names.push_back(node->GetName());
  }

  EXPECT_EQ(dfs_names, expected_dfs_names);
}

TEST_F(UtestComputeGraph, NotTrainTopologicalSortingInPriorityMode_DFS_success) {
  std::map<std::string, std::string> options_map;

  auto graph = BuildNormalGraph("test_dfs_not_train_topo_graph_dfx");
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  std::vector<std::string> expected_dfs_names = {"node6", "node1", "node2", "node3", "node4", "node5", "netoutput"};
  std::vector<std::string> dfs_names;
  options_map["ge.graphRunMode"] = "0";  // not tarin
  options_map["ge.topoSortingMode"] = "";
  options_map["ge.exec.memoryOptimizationPolicy"] = "MemoryPriority";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  const auto &graph_dfs_topo = graph->GetAllNodes();
  for (auto &node : graph_dfs_topo) {
    dfs_names.push_back(node->GetName());
  }

  EXPECT_EQ(dfs_names, expected_dfs_names);
}

TEST_F(UtestComputeGraph, ReverseDfsTopologicalSortingInPriorityMode_success) {
  std::map<std::string, std::string> options_map;

  auto graph = BuildNormalGraph("test_reverse_dfs_topo_graph");
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  std::vector<std::string> expected_dfs_names = {"node6", "node1", "node2", "node4", "node3", "node5", "netoutput"};
  std::vector<std::string> dfs_names;
  options_map["ge.topoSortingMode"] = "1";
  options_map["ge.exec.memoryOptimizationPolicy"] = "MemoryPriority";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_EQ(graph->TopologicalSortingGraph(true), GRAPH_SUCCESS);
  const auto &graph_dfs_topo = graph->GetAllNodes();
  for (auto &node : graph_dfs_topo) {
    dfs_names.push_back(node->GetName());
  }

  EXPECT_EQ(dfs_names, expected_dfs_names);
}

TEST_F(UtestComputeGraph, SortNodes_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 1, 1);
  const auto &node2 = builder.AddNode("node2", "node2", 1, 1);
  const auto &node3 = builder.AddNode("node3", "node3", 1, 1);
  const auto &node4 = builder.AddNode("node4", "node4", 1, 0);

  builder.AddControlEdge(node1, node2);
  builder.AddControlEdge(node3, node1);
  builder.AddControlEdge(node2, node4);
  auto graph = builder.GetGraph();
  std::map<NodePtr, uint32_t> map_in_edge_num{{node1, 2},{node2, 2},{node3, 2}};
  std::vector<NodePtr> stack{};
  EXPECT_EQ(graph->SortNodes(stack, map_in_edge_num), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, GetInEdgeSize_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 2, 0);
  const auto &node2 = builder.AddNode("node2", "node2", 0, 1);
  const auto &node3 = builder.AddNode("node3", "node3", 0, 1);
  builder.AddDataEdge(node2, 0, node1, 0);
  builder.AddDataEdge(node3, 0, node1, 1);
  auto graph = builder.GetGraph();
  EXPECT_EQ(graph->GetInEdgeSize(node1), 2);
}

TEST_F(UtestComputeGraph, GetOutEdgeSize_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 0, 2);
  const auto &node2 = builder.AddNode("node2", "node2", 1, 0);
  const auto &node3 = builder.AddNode("node3", "node3", 1, 0);
  builder.AddDataEdge(node1, 0, node2, 0);
  builder.AddDataEdge(node1, 1, node3, 0);
  auto graph = builder.GetGraph();
  graph->Dump();
  EXPECT_EQ(graph->GetOutEdgeSize(node1), 2);
}

TEST_F(UtestComputeGraph, IsValid_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  EXPECT_EQ(graph->IsValid(), false);
}

TEST_F(UtestComputeGraph, InValid_success) {
  const auto func = [](const NodePtr &node1, const NodePtr &node2) -> bool { return true; };
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 0, 0);
  const auto &node2 = builder.AddNode("node2", "node2", 0, 0);
  auto graph = builder.GetGraph();
  graph->TopologicalSorting(func);
  EXPECT_EQ(graph->IsValid(), true);
  graph->InValid();
  EXPECT_EQ(graph->IsValid(), false);
}

TEST_F(UtestComputeGraph, Swap_success) {
  auto builder1 = ut::GraphBuilder("graph1");
  const auto &node1 = builder1.AddNode("node1", "node1", 0, 0);
  auto graph1 = builder1.GetGraph();
  auto builder2 = ut::GraphBuilder("graph2");
  const auto &node2 = builder2.AddNode("node2", "node2", 0, 0);
  const auto &node3 = builder2.AddNode("node3", "node3", 0, 0);
  auto graph2 = builder2.GetGraph();

  graph1->Swap(*(graph2));
  EXPECT_EQ(graph1->GetNodes(false).size(), 2);
  EXPECT_EQ(graph2->GetNodes(false).size(), 1);
  EXPECT_EQ(graph1->GetName(), "graph2");
  EXPECT_EQ(graph2->GetName(), "graph1");
}

TEST_F(UtestComputeGraph, Swap_with_subgraph_success) {
  auto graph1 = BuildMainGraphWithIf("root_graph_1");
  auto graph2 = BuildMainGraphWithIf("root_graph_2");

  graph1->Swap(*(graph2));
  auto if_node_1 = graph1->FindFirstNodeMatchType("If");
  ASSERT_NE(if_node_1, nullptr);
  auto if_node_2 = graph2->FindFirstNodeMatchType("If");
  ASSERT_NE(if_node_2, nullptr);
  EXPECT_EQ(graph1->GetName(), "root_graph_2");
  EXPECT_EQ(graph2->GetName(), "root_graph_1");
  EXPECT_EQ(if_node_1->GetOwnerComputeGraph()->GetName(), "root_graph_2");
  EXPECT_EQ(if_node_2->GetOwnerComputeGraph()->GetName(), "root_graph_1");

  const auto if_1_subgraph_name = if_node_1->GetOpDesc()->GetSubgraphInstanceName(0);
  const auto if_1_subgraph = graph1->GetSubgraph(if_1_subgraph_name);
  ASSERT_NE(if_1_subgraph, nullptr);
  EXPECT_EQ(if_1_subgraph->GetParentGraph()->GetName(), graph1->GetName());

  const auto if_2_subgraph_name = if_node_2->GetOpDesc()->GetSubgraphInstanceName(0);
  const auto if_2_subgraph = graph2->GetSubgraph(if_2_subgraph_name);
  ASSERT_NE(if_2_subgraph, nullptr);
  EXPECT_EQ(if_2_subgraph->GetParentGraph()->GetName(), graph2->GetName());
}

TEST_F(UtestComputeGraph, InsertToNodeList_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 0, 0);
  const auto &node2 = builder.AddNode("node2", "node2", 0, 0);
  const auto &node3 = builder.AddNode("node3", "node1", 0, 0);
  auto graph = builder.GetGraph();
  graph->impl_->InsertToNodeList(graph->impl_->nodes_.begin(), node3);
  EXPECT_EQ(*(graph->impl_->nodes_.begin()), node3);
}

TEST_F(UtestComputeGraph, PushBackToNodeList_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 0, 0);
  const auto &node2 = builder.AddNode("node2", "node2", 0, 0);
  const auto &node3 = builder.AddNode("node3", "node3", 0, 0);
  auto graph = builder.GetGraph();
  graph->impl_->PushBackToNodeList(node1);
  auto node_list = graph->GetDirectNode();
  EXPECT_EQ(*(node_list.end() - 1), node1);
}

TEST_F(UtestComputeGraph, EmplaceBackToNodeList_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 0, 0);
  const auto &node2 = builder.AddNode("node2", "node2", 0, 0);
  const auto &node3 = builder.AddNode("node3", "node1", 0, 0);
  auto graph = builder.GetGraph();
  graph->impl_->EmplaceBackToNodeList(node1);
  auto node_list = graph->GetDirectNode();
  EXPECT_EQ(*(node_list.end() - 1), node1);
}

TEST_F(UtestComputeGraph, ClearNodeList_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 0, 0);
  const auto &node2 = builder.AddNode("node2", "node2", 0, 0);
  const auto &node3 = builder.AddNode("node3", "node1", 0, 0);
  auto graph = builder.GetGraph();
  graph->ClearNodeList();
  EXPECT_EQ(graph->GetDirectNode().size(), 0);
}

TEST_F(UtestComputeGraph, IsolateNode_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 2, 2);
  const auto &node2 = builder.AddNode("node2", "node2", 0, 1);
  const auto &node3 = builder.AddNode("node3", "node3", 1, 0);
  const auto &node4 = builder.AddNode("node4", "node4", 0, 1);
  const auto &node5 = builder.AddNode("node5", "node5", 1, 0);
  builder.AddDataEdge(node2, 0, node1, 0);
  builder.AddDataEdge(node1, 0, node3, 0);
  builder.AddControlEdge(node1, node4);
  builder.AddControlEdge(node5, node1);
  auto graph = builder.GetGraph();
  EXPECT_EQ(graph->IsolateNode(node1), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, RemoveExtraOutEdge_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 1, 1);
  const auto &node2 = builder.AddNode("node2", "node2", 0, 1);
  const auto &node3 = builder.AddNode("node3", "node3", 1, 0);
  builder.AddControlEdge(node1, node2);
  builder.AddControlEdge(node3, node1);
  auto graph = builder.GetGraph();
  EXPECT_EQ(graph->RemoveExtraOutEdge(node1), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, Verify_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 0, 0);
  auto graph = builder.GetGraph();
  EXPECT_EQ(GraphUtilsEx::Verify(graph), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, InferOriginFormat_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 1, 0);
  const auto &node2 = builder.AddNode("node2", "node2", 0, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  auto graph = builder.GetGraph();
  EXPECT_EQ(GraphUtilsEx::InferOriginFormat(graph), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, InferShapeInNeed_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 1, 0);
  const auto &node2 = builder.AddNode("node2", "node2", 0, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  auto graph = builder.GetGraph();
  EXPECT_EQ(GraphUtilsEx::InferShapeInNeed(graph), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, SetSessionID_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto session_id = graph->GetSessionID() + 1;
  graph->SetSessionID(session_id);
  EXPECT_EQ(graph->GetSessionID(), session_id);
}

TEST_F(UtestComputeGraph, SetGraphID_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto graph_id = graph->GetGraphID() + 1;
  graph->SetGraphID(graph_id);
  EXPECT_EQ(graph->GetGraphID(), graph_id);
}

TEST_F(UtestComputeGraph, SetSummaryGraph_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto summary_flag = !graph->IsSummaryGraph();
  graph->SetSummaryFlag(summary_flag);
  EXPECT_EQ(graph->IsSummaryGraph(), summary_flag);
}
TEST_F(UtestComputeGraph, DFSPOSTORDERTopologicalSorting_success) {
  auto builder = ut::GraphBuilder("graph_reverse_dfs");
  const auto &node1 = builder.AddNode("node1", "node1", 0, 3, FORMAT_NCHW, DT_FLOAT, {1,1});
  const auto &node2 = builder.AddNode("node2", "node2", 0, 3,FORMAT_NCHW, DT_FLOAT, {1,1});
  const auto &node3 = builder.AddNode("node3", "node3", 0, 3,FORMAT_NCHW, DT_FLOAT, {1,1});
  const auto &node4 = builder.AddNode("node4", "node4", 0, 3,FORMAT_NCHW, DT_FLOAT, {1,1});
  const auto &node5 = builder.AddNode("node5", "node5", 1, 1,FORMAT_NCHW, DT_FLOAT, {1,2});
  const auto &node6 = builder.AddNode("node6", "node6", 1, 1,FORMAT_NCHW, DT_FLOAT, {1,3});
  const auto &node7 = builder.AddNode("node7", "node7", 1, 1,FORMAT_NCHW, DT_FLOAT, {1,4});
  const auto &node8 = builder.AddNode("node8", "node8", 1, 1,FORMAT_NCHW, DT_FLOAT, {1,5});
  const auto &node9 = builder.AddNode("node9", "node9", 1, 1,FORMAT_NCHW, DT_FLOAT, {2,2});
  const auto &node10 = builder.AddNode("node10", "node10", 1, 1,FORMAT_NCHW, DT_FLOAT, {2,3});
  const auto &node11 = builder.AddNode("node11", "node11", 1, 1,FORMAT_NCHW, DT_FLOAT, {2,3});
  const auto &node12 = builder.AddNode("node12", "node12", 1, 1,FORMAT_NCHW, DT_FLOAT, {2,4});
  const auto &node13 = builder.AddNode("node13", "node13", 1, 1,FORMAT_NCHW, DT_FLOAT, {3,2});
  const auto &node14 = builder.AddNode("node14", "node14", 1, 1,FORMAT_NCHW, DT_FLOAT, {3,3});
  const auto &node15 = builder.AddNode("node15", "node15", 1, 1,FORMAT_NCHW, DT_FLOAT, {3,4});
  const auto &node16 = builder.AddNode("node16", "node16", 1, 1,FORMAT_NCHW, DT_FLOAT, {3,5});
  const auto &node17 = builder.AddNode("node17", "node17", 4, 1,FORMAT_NCHW, DT_FLOAT, {4,2});
  const auto &node18 = builder.AddNode("node18", "node18", 4, 1,FORMAT_NCHW, DT_FLOAT, {4,2});
  const auto &node19 = builder.AddNode("node19", "node19", 4, 1,FORMAT_NCHW, DT_FLOAT, {4,2});
  const auto &node20 = builder.AddNode("node20", "node20", 3, 0);

  builder.AddDataEdge(node1, 0, node5, 0);
  builder.AddDataEdge(node1, 1, node9, 0);
  builder.AddDataEdge(node1, 2, node13, 0);
  builder.AddDataEdge(node2, 0, node6, 0);
  builder.AddDataEdge(node2, 1, node10, 0);
  builder.AddDataEdge(node2, 2, node14, 0);
  builder.AddDataEdge(node3, 0, node7, 0);
  builder.AddDataEdge(node3, 1, node11, 0);
  builder.AddDataEdge(node3, 2, node15, 0);
  builder.AddDataEdge(node4, 0, node8, 0);
  builder.AddDataEdge(node4, 1, node12, 0);
  builder.AddDataEdge(node4, 2, node16, 0);
  builder.AddDataEdge(node5, 0, node17, 0);
  builder.AddDataEdge(node6, 0, node17, 1);
  builder.AddDataEdge(node7, 0, node17, 2);
  builder.AddDataEdge(node8, 0, node17, 3);
  builder.AddDataEdge(node9, 0, node18, 0);
  builder.AddDataEdge(node10, 0, node18, 1);
  builder.AddDataEdge(node11, 0, node18, 2);
  builder.AddDataEdge(node12, 0, node18, 3);
  builder.AddDataEdge(node13, 0, node19, 0);
  builder.AddDataEdge(node14, 0, node19, 1);
  builder.AddDataEdge(node15, 0, node19, 2);
  builder.AddDataEdge(node16, 0, node19, 3);

  builder.AddControlEdge(node17, node20);
  builder.AddControlEdge(node18, node20);
  builder.AddControlEdge(node19, node20);
  GetThreadLocalContext().SetGraphOption({});
  auto graph = builder.GetGraph();
  std::map<std::string, std::string> options_map;
  options_map["ge.topoSortingMode"] = "2";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);

  EXPECT_EQ(node1->GetOpDesc()->GetId(), 0);
  EXPECT_EQ(node2->GetOpDesc()->GetId(), 2);
  EXPECT_EQ(node3->GetOpDesc()->GetId(), 4);
  EXPECT_EQ(node4->GetOpDesc()->GetId(), 6);
  EXPECT_EQ(node5->GetOpDesc()->GetId(), 1);
  EXPECT_EQ(node6->GetOpDesc()->GetId(), 3);
  EXPECT_EQ(node7->GetOpDesc()->GetId(), 5);
  EXPECT_EQ(node8->GetOpDesc()->GetId(), 7);
  EXPECT_EQ(node9->GetOpDesc()->GetId(), 9);
  EXPECT_EQ(node10->GetOpDesc()->GetId(), 10);
  EXPECT_EQ(node11->GetOpDesc()->GetId(), 11);
  EXPECT_EQ(node12->GetOpDesc()->GetId(), 12);
  EXPECT_EQ(node13->GetOpDesc()->GetId(), 14);
  EXPECT_EQ(node14->GetOpDesc()->GetId(), 15);
  EXPECT_EQ(node15->GetOpDesc()->GetId(), 16);
  EXPECT_EQ(node16->GetOpDesc()->GetId(), 17);
  EXPECT_EQ(node17->GetOpDesc()->GetId(), 8);
  EXPECT_EQ(node18->GetOpDesc()->GetId(), 13);
  EXPECT_EQ(node19->GetOpDesc()->GetId(), 18);
  EXPECT_EQ(node20->GetOpDesc()->GetId(), 19);
}

TEST_F(UtestComputeGraph, DFSPOSTORDERTopologicalSorting_ringing_fail) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 1, 1);
  const auto &node2 = builder.AddNode("node2", "node2", 1, 1);
  const auto &node3 = builder.AddNode("node3", "node3", 1, 1);

  builder.AddDataEdge(node1, 0, node2, 0);
  builder.AddDataEdge(node2, 0, node3, 0);
  builder.AddDataEdge(node3, 0, node1, 0);

  auto graph = builder.GetGraph();
  std::map<std::string, std::string> options_map;
  options_map["ge.topoSortingMode"] = "2";
  options_map["ge.exec.memoryOptimizationPolicy"] = "MemoryPriority";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_NE(graph->TopologicalSorting(), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, DelayTopologicalSorting) {
  auto graph = BuildDelayTopoGraph("test_delay_topo_graph");
  std::map<std::string, std::string> options_map;
  options_map["ge.topoSortingMode"] = "2";
  options_map["ge.exec.memoryOptimizationPolicy"] = "MemoryPriority";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  std::vector<std::string> expected_dfs_names = {"variable", "data", "node2", "node3", "node4", "node1", "node5"};
  std::vector<std::string> dfs_names;
  const auto &graph_dfs_topo = graph->GetAllNodes();
  for (auto &node : graph_dfs_topo) {
    dfs_names.push_back(node->GetName());
  }

  EXPECT_EQ(dfs_names, expected_dfs_names);
}

TEST_F(UtestComputeGraph, NoDelayTopologicalSorting) {
  auto graph = BuildDelayTopoGraph("test_delay_topo_graph");
  std::map<std::string, std::string> options_map;
  options_map["ge.topoSortingMode"] = "2";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  std::vector<std::string> expected_dfs_names = {"variable", "node1", "node2", "data", "node3", "node4", "node5"};
  std::vector<std::string> dfs_names;
  const auto &graph_dfs_topo = graph->GetAllNodes();
  for (auto &node : graph_dfs_topo) {
    dfs_names.push_back(node->GetName());
  }

  EXPECT_EQ(dfs_names, expected_dfs_names);
}

TEST_F(UtestComputeGraph, DelayTopologicalSortingMultiInput) {
  auto graph = BuildDelayTopoGraphMultiInput("test_delay_topo_graph");
  std::map<std::string, std::string> options_map;
  options_map["ge.topoSortingMode"] = "2";
  options_map["ge.exec.memoryOptimizationPolicy"] = "MemoryPriority";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  std::vector<std::string> expected_dfs_names =
      {"const", "constant", "variable", "data", "node2", "node3", "node4", "node1", "node5"};
  std::vector<std::string> dfs_names;
  const auto &graph_dfs_topo = graph->GetAllNodes();
  for (auto &node : graph_dfs_topo) {
    dfs_names.push_back(node->GetName());
  }

  EXPECT_EQ(dfs_names, expected_dfs_names);
}

TEST_F(UtestComputeGraph, NoDelayTopologicalSortingMultiInput) {
  auto graph = BuildDelayTopoGraphMultiInput("test_delay_topo_graph", false);
  std::map<std::string, std::string> options_map;
  options_map["ge.topoSortingMode"] = "2";
  options_map["ge.exec.memoryOptimizationPolicy"] = "MemoryPriority";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  std::vector<std::string> expected_dfs_names =
    {"const", "constant", "variable", "node1", "data", "node2", "node3", "node4", "node5"};
  std::vector<std::string> dfs_names;
  const auto &graph_dfs_topo = graph->GetAllNodes();
  for (auto &node : graph_dfs_topo) {
    dfs_names.push_back(node->GetName());
  }

  EXPECT_EQ(dfs_names, expected_dfs_names);
}

TEST_F(UtestComputeGraph, ReorderByNodeId) {
  auto graph = BuildDelayTopoGraphMultiInput("test_delay_topo_graph");
  const auto &constant = graph->FindNode("const");
  const auto &constantop = graph->FindNode("constant");
  const auto &variable = graph->FindNode("variable");
  const auto &node1 = graph->FindNode("node1");
  const auto &node2 = graph->FindNode("node2");
  const auto &node3 = graph->FindNode("node3");
  const auto &node4 = graph->FindNode("node4");
  const auto &node5 = graph->FindNode("node5");
  const auto &data = graph->FindNode("data");
  int64_t seq_id = 0L;
  std::vector<NodePtr> nodes{node5, node4, node3, node2, node1, variable, data, constantop, constant};
  for (auto &node : nodes) {
    node->GetOpDesc()->SetId(seq_id++);
  }
  graph->ReorderByNodeId();
  auto sorted_nodes = graph->GetDirectNode();
  ASSERT_TRUE(sorted_nodes.size() == nodes.size());
  int32_t id = 0;
  for (auto &node : nodes) {
    EXPECT_EQ(node, sorted_nodes.at(id++));
  }
}

}
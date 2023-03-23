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

#define protected public
#define private public

#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/op_desc_impl.h"
#include "graph/ge_local_context.h"
#include "graph_builder_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_util.h"
#include "graph/utils/op_desc_utils.h"

#undef private
#undef protected

namespace ge {
namespace {
bool IfNodeExist(const ComputeGraphPtr &graph, std::function<bool(const NodePtr &)> filter,
                 bool direct_node_flag = true) {
  for (const auto &node : graph->GetNodes(direct_node_flag)) {
    if (filter(node)) {
      return true;
    }
  }
  return false;
}
/*
 *             data1  const1         data2  const2
 *                \    /                \    /
 *                 add1                  add2
 *                   |                    |
 *                 cast1                cast2
 *                   |                    |
 *                square1  var1  var2  square2
 *                     \   /  |  |  \   /
 *                     less1  |  |  less2
 *                          \ |  | /
 *                            mul
 *                             |
 *                             |
 *                             |
 *                          netoutput
 */
void BuildGraphForUnfold(ComputeGraphPtr &graph, ComputeGraphPtr &subgraph) {
  auto builder = ut::GraphBuilder("root");
  const auto &input1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &var1 = builder.AddNode("var1", VARIABLEV2, 1, 1);
  const auto &input2 = builder.AddNode("data2", DATA, 1, 1);
  const auto &var2 = builder.AddNode("var2", VARIABLEV2, 1, 1);
  const auto &func = builder.AddNode("func", PARTITIONEDCALL, 4, 1);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(input1, 0, func, 0);
  builder.AddDataEdge(var1, 0, func, 1);
  builder.AddDataEdge(input2, 0, func, 2);
  builder.AddDataEdge(var2, 0, func, 3);
  builder.AddDataEdge(func, 0, netoutput, 0);

  graph = builder.GetGraph();

  auto sub_builder = ut::GraphBuilder("sub");
  const auto &data1 = sub_builder.AddNode("data1", DATA, 1, 1);
  const auto &const1 = sub_builder.AddNode("const1", CONSTANTOP, 0, 1);
  const auto &add1 = sub_builder.AddNode("add1", "Add", 2, 1);
  const auto &cast1 = sub_builder.AddNode("cast1", "Cast", 1, 1);
  const auto &func1 = sub_builder.AddNode("func1", PARTITIONEDCALL, 2, 1);
  const auto &data2 = sub_builder.AddNode("data2", DATA, 1, 1);
  const auto &data3 = sub_builder.AddNode("data3", DATA, 1, 1);
  const auto &const2 = sub_builder.AddNode("const2", CONSTANTOP, 0, 1);
  const auto &add2 = sub_builder.AddNode("add2", "Add", 2, 1);
  const auto &cast2 = sub_builder.AddNode("cast2", "Cast", 1, 1);
  const auto &func2 = sub_builder.AddNode("func2", PARTITIONEDCALL, 2, 1);
  const auto &data4 = sub_builder.AddNode("data4", DATA, 1, 1);
  const auto &mul = sub_builder.AddNode("mul", "Mul", 2, 1);
  const auto &netoutput0 = sub_builder.AddNode("netoutput0", NETOUTPUT, 1, 0);
  sub_builder.AddDataEdge(data1, 0, add1, 0);
  sub_builder.AddDataEdge(const1, 0, add1, 1);
  sub_builder.AddDataEdge(add1, 0, cast1, 0);
  sub_builder.AddDataEdge(cast1, 0, func1, 0);
  sub_builder.AddDataEdge(data2, 0, func1, 1);
  sub_builder.AddDataEdge(data3, 0, add2, 0);
  sub_builder.AddDataEdge(const2, 0, add2, 1);
  sub_builder.AddDataEdge(add2, 0, cast2, 0);
  sub_builder.AddDataEdge(cast2, 0, func2, 0);
  sub_builder.AddDataEdge(data4, 0, func2, 1);
  sub_builder.AddDataEdge(func1, 0, mul, 0);
  sub_builder.AddDataEdge(func2, 0, mul, 1);
  sub_builder.AddDataEdge(mul, 0, netoutput0, 0);

  subgraph = sub_builder.GetGraph();
  subgraph->SetGraphUnknownFlag(true);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(data3->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 2);
  AttrUtils::SetInt(data4->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 3);
  AttrUtils::SetInt(netoutput0->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  func->GetOpDesc()->AddSubgraphName("f");
  func->GetOpDesc()->SetSubgraphInstanceName(0, subgraph->GetName());
  graph->AddSubGraph(subgraph);
  subgraph->SetParentNode(func);
  subgraph->SetParentGraph(graph);

  auto sub_sub_builder1 = ut::GraphBuilder("sub_sub1");
  const auto &data5 = sub_sub_builder1.AddNode("data5", DATA, 1, 1);
  const auto &data6 = sub_sub_builder1.AddNode("data6", DATA, 1, 1);
  const auto &square1 = sub_sub_builder1.AddNode("square1", "Square", 1, 1);
  const auto &less1 = sub_sub_builder1.AddNode("less1", "Less", 2, 1);
  const auto &netoutput1 = sub_sub_builder1.AddNode("netoutput1", NETOUTPUT, 1, 0);
  sub_sub_builder1.AddDataEdge(data5, 0, square1, 0);
  sub_sub_builder1.AddDataEdge(square1, 0, less1, 0);
  sub_sub_builder1.AddDataEdge(data6, 0, less1, 1);
  sub_sub_builder1.AddDataEdge(less1, 0, netoutput1, 0);

  const auto &sub_subgraph1 = sub_sub_builder1.GetGraph();
  sub_subgraph1->SetGraphUnknownFlag(true);
  AttrUtils::SetInt(data5->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(data6->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(netoutput1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  func1->GetOpDesc()->AddSubgraphName("f");
  func1->GetOpDesc()->SetSubgraphInstanceName(0, sub_subgraph1->GetName());
  graph->AddSubGraph(sub_subgraph1);
  sub_subgraph1->SetParentNode(func1);
  sub_subgraph1->SetParentGraph(subgraph);

  auto sub_sub_builder2 = ut::GraphBuilder("sub_sub2");
  const auto &data7 = sub_sub_builder2.AddNode("data7", DATA, 1, 1);
  const auto &data8 = sub_sub_builder2.AddNode("data8", DATA, 1, 1);
  const auto &square2 = sub_sub_builder2.AddNode("square2", "Square", 1, 1);
  const auto &less2 = sub_sub_builder2.AddNode("less2", "Less", 2, 1);
  const auto &netoutput2 = sub_sub_builder2.AddNode("netoutput2", NETOUTPUT, 1, 0);
  sub_sub_builder2.AddDataEdge(data7, 0, square2, 0);
  sub_sub_builder2.AddDataEdge(square2, 0, less2, 0);
  sub_sub_builder2.AddDataEdge(data8, 0, less2, 1);
  sub_sub_builder2.AddDataEdge(less2, 0, netoutput2, 0);

  const auto &sub_subgraph2 = sub_sub_builder2.GetGraph();
  sub_subgraph2->SetGraphUnknownFlag(false);
  AttrUtils::SetInt(data7->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(data8->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(netoutput2->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  func2->GetOpDesc()->AddSubgraphName("f");
  func2->GetOpDesc()->SetSubgraphInstanceName(0, sub_subgraph2->GetName());
  graph->AddSubGraph(sub_subgraph2);
  sub_subgraph2->SetParentNode(func2);
  sub_subgraph2->SetParentGraph(subgraph);

  return;
}
/*                                   --------------             
 *                                  |              |
 *             data1  const1     data2  const2     |
 *              |  \    /           \    /         |
 *              |   add1             add2          |
 *              |    |                 |           |
 *              |  cast1              cast2        |
 *              |    |                 |           |
 *              |    |                 |           |
 *              |     \               /            |
 *              \      ------  mul ------------------
 *               \              |
 *                \             |
 *                 \            |
 *                  ------- netoutput
 */
void BuildGraphForUnfoldWithControlEdge(ComputeGraphPtr &graph, ComputeGraphPtr &subgraph) {
  auto builder = ut::GraphBuilder("root");
  const auto &input1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &input2 = builder.AddNode("data2", DATA, 1, 1);
  const auto &func = builder.AddNode("func", PARTITIONEDCALL, 4, 1);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(input1, 0, func, 0);
  builder.AddDataEdge(input2, 0, func, 1);
  builder.AddDataEdge(func, 0, netoutput, 0);

  graph = builder.GetGraph();

  auto sub_builder = ut::GraphBuilder("sub");
  const auto &data1 = sub_builder.AddNode("data1", DATA, 1, 1);
  const auto &const1 = sub_builder.AddNode("const1", CONSTANTOP, 0, 1);
  const auto &add1 = sub_builder.AddNode("add1", "Add", 2, 1);
  const auto &cast1 = sub_builder.AddNode("cast1", "Cast", 1, 1);
  const auto &data2 = sub_builder.AddNode("data2", DATA, 1, 1);
  const auto &const2 = sub_builder.AddNode("const2", CONSTANTOP, 0, 1);
  const auto &add2 = sub_builder.AddNode("add2", "Add", 2, 1);
  const auto &cast2 = sub_builder.AddNode("cast2", "Cast", 1, 1);
  const auto &mul = sub_builder.AddNode("mul", "Mul", 2, 1);
  const auto &netoutput0 = sub_builder.AddNode("netoutput0", NETOUTPUT, 1, 0);
  sub_builder.AddDataEdge(data1, 0, add1, 0);
  sub_builder.AddControlEdge(data1, netoutput0);
  sub_builder.AddDataEdge(const1, 0, add1, 1);
  sub_builder.AddDataEdge(add1, 0, cast1, 0);
  sub_builder.AddDataEdge(cast1, 0, mul, 0);
  sub_builder.AddControlEdge(data2, mul);
  sub_builder.AddDataEdge(data2, 0, add2, 0);
  sub_builder.AddDataEdge(const2, 0, add2, 1);
  sub_builder.AddDataEdge(add2, 0, cast2, 0);
  sub_builder.AddDataEdge(cast2, 0, mul, 1);
  sub_builder.AddDataEdge(mul, 0, netoutput0, 0);

  subgraph = sub_builder.GetGraph();
  subgraph->SetGraphUnknownFlag(true);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(netoutput0->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  func->GetOpDesc()->AddSubgraphName("f");
  func->GetOpDesc()->SetSubgraphInstanceName(0, subgraph->GetName());
  graph->AddSubGraph(subgraph);
  subgraph->SetParentNode(func);
  subgraph->SetParentGraph(graph);
  return;
}

void BuildGraphWithPlaceholderAndEnd(ComputeGraphPtr &graph) {
  auto builder = ut::GraphBuilder("root");
  const auto &input1 = builder.AddNode("pld1", PLACEHOLDER, 1, 1);
  const auto &input2 = builder.AddNode("pld2", PLACEHOLDER, 1, 1);
  const auto &data1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &data2 = builder.AddNode("data2", DATA, 1, 1);
  const auto &end = builder.AddNode("end", END, 1, 1);
  const auto &add1 = builder.AddNode("add1", "Add", 2, 1);
  const auto &add2 = builder.AddNode("add2", "Add", 2, 1);
  const auto &add3 = builder.AddNode("add3", "Add", 2, 1);
  builder.AddDataEdge(input1, 0, add1, 0);
  builder.AddDataEdge(input2, 0, add1, 1);
  builder.AddDataEdge(data1, 0, add2, 0);
  builder.AddDataEdge(data2, 0, add2, 1);
  builder.AddDataEdge(add1, 0, add3, 0);
  builder.AddDataEdge(add2, 0, add3, 1);
  builder.AddDataEdge(add3, 0, end, 0);
  graph = builder.GetGraph();
  graph->AddOutputNode(end);
}

ComputeGraphPtr BuildGraphWithSubGraph() {
  auto root_builder = ut::GraphBuilder("root");
  const auto &case0 = root_builder.AddNode("case0", "Case", 0, 0);
  const auto &root_graph = root_builder.GetGraph();

  auto sub_builder1 = ut::GraphBuilder("sub1");
  const auto &data1 = sub_builder1.AddNode("data1", "Data", 0, 0);
  const auto &sub_graph1 = sub_builder1.GetGraph();
  root_graph->AddSubGraph(sub_graph1);
  sub_graph1->SetParentNode(case0);
  sub_graph1->SetParentGraph(root_graph);
  case0->GetOpDesc()->AddSubgraphName("branch1");
  case0->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");

  auto sub_builder2 = ut::GraphBuilder("sub2");
  const auto &data2 = sub_builder2.AddNode("data2", "Data", 0, 0);
  const auto &sub_graph2 = sub_builder2.GetGraph();
  root_graph->AddSubGraph(sub_graph2);
  sub_graph2->SetParentNode(case0);
  sub_graph2->SetParentGraph(root_graph);
  case0->GetOpDesc()->AddSubgraphName("branch1");
  case0->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");
  case0->GetOpDesc()->AddSubgraphName("branch2");
  case0->GetOpDesc()->SetSubgraphInstanceName(1, "sub2");
  return root_graph;
}
} // namespace

namespace {
class UtestComputeGraphBuilder : public ComputeGraphBuilder {
 public:
  virtual ComputeGraphPtr Build(graphStatus &error_code, std::string &error_msg) {
    auto graph = std::make_shared<ComputeGraph>("test");
    auto op_desc = std::make_shared<OpDesc>("node", "node");
    NodePtr node = graph->AddNode(op_desc);
    std::map<std::string, NodePtr> node_names_;
    node_names_.insert(pair<std::string, NodePtr>("node", node));
    return graph;
  }

  NodePtr GetNode(const std::string &name);
  std::vector<NodePtr> GetAllNodes();
  void BuildNodes(graphStatus &error_code, std::string &error_msg);
};

NodePtr UtestComputeGraphBuilder::GetNode(const std::string &name) {
  return ComputeGraphBuilder::GetNode(name);
}

std::vector<NodePtr> UtestComputeGraphBuilder::GetAllNodes() {
  return ComputeGraphBuilder::GetAllNodes();
}

void UtestComputeGraphBuilder::BuildNodes(graphStatus &error_code, std::string &error_msg) {
  return ComputeGraphBuilder::BuildNodes(error_code, error_msg);
}

} // namespace

class UtestGraphUtils : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

/*
*               var                               var
*  atomicclean  |  \                             |   \
*         \\    |   assign                       |   assign
*          \\   |   //         =======>          |   //
*           allreduce                         identity  atomicclean
*             |                                 |       //
*            netoutput                        allreduce
*                                               |
*                                           netoutput
 */
TEST_F(UtestGraphUtils, InsertNodeBefore_DoNotMoveCtrlEdgeFromAtomicClean) {
  // build test graph
  auto builder = ut::GraphBuilder("test");
  const auto &var = builder.AddNode("var", VARIABLE, 0, 1);
  const auto &assign = builder.AddNode("assign", "Assign", 1, 1);
  const auto &allreduce = builder.AddNode("allreduce", "HcomAllReduce", 1, 1);
  const auto &atomic_clean = builder.AddNode("atomic_clean", ATOMICADDRCLEAN, 0, 0);
  const auto &netoutput1 = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  const auto &identity = builder.AddNode("identity", "Identity", 1, 1);

  builder.AddDataEdge(var, 0, assign, 0);
  builder.AddDataEdge(var,0,allreduce,0);
  builder.AddControlEdge(assign, allreduce);
  builder.AddControlEdge(atomic_clean, allreduce);
  auto graph = builder.GetGraph();

  // insert identity before allreduce
  GraphUtils::InsertNodeBefore(allreduce->GetInDataAnchor(0), identity, 0, 0);

  // check assign control-in on identity
  ASSERT_EQ(identity->GetInControlNodes().at(0)->GetName(), "assign");
  // check atomicclean control-in still on allreuce
  ASSERT_EQ(allreduce->GetInControlNodes().at(0)->GetName(), "atomic_clean");
}

TEST_F(UtestGraphUtils, GetSubgraphs) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &case0 = root_builder.AddNode("case0", "Case", 0, 0);
  const auto &root_graph = root_builder.GetGraph();

  auto sub_builder1 = ut::GraphBuilder("sub1");
  const auto &case1 = sub_builder1.AddNode("case1", "Case", 0, 0);
  const auto &sub_graph1 = sub_builder1.GetGraph();
  root_graph->AddSubGraph(sub_graph1);
  sub_graph1->SetParentNode(case0);
  sub_graph1->SetParentGraph(root_graph);
  case0->GetOpDesc()->AddSubgraphName("branch1");
  case0->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");

  auto sub_builder2 = ut::GraphBuilder("sub2");
  const auto &sub_graph2 = sub_builder2.GetGraph();
  root_graph->AddSubGraph(sub_graph2);
  sub_graph2->SetParentNode(case1);
  sub_graph2->SetParentGraph(sub_graph1);
  case1->GetOpDesc()->AddSubgraphName("branch1");
  case1->GetOpDesc()->SetSubgraphInstanceName(0, "sub2");
  case1->GetOpDesc()->AddSubgraphName("branch2");
  case1->GetOpDesc()->SetSubgraphInstanceName(1, "not_exist");

  std::vector<ComputeGraphPtr> subgraphs1;
  ASSERT_EQ(GraphUtils::GetSubgraphsRecursively(root_graph, subgraphs1), GRAPH_SUCCESS);
  ASSERT_EQ(subgraphs1.size(), 2);

  std::vector<ComputeGraphPtr> subgraphs2;
  ASSERT_EQ(GraphUtils::GetSubgraphsRecursively(sub_graph1, subgraphs2), GRAPH_SUCCESS);
  ASSERT_EQ(subgraphs2.size(), 1);

  std::vector<ComputeGraphPtr> subgraphs3;
  ASSERT_EQ(GraphUtils::GetSubgraphsRecursively(sub_graph2, subgraphs3), GRAPH_SUCCESS);
  ASSERT_TRUE(subgraphs3.empty());
}

TEST_F(UtestGraphUtils, GetSubgraphs_nullptr_graph) {
  std::vector<ComputeGraphPtr> subgraphs;
  ASSERT_NE(GraphUtils::GetSubgraphsRecursively(nullptr, subgraphs), GRAPH_SUCCESS);
  ASSERT_TRUE(subgraphs.empty());
}

TEST_F(UtestGraphUtils, ReplaceEdgeSrc) {
  auto builder = ut::GraphBuilder("root");
  const auto &node0 = builder.AddNode("node0", "node", 1, 1);
  const auto &node1 = builder.AddNode("node1", "node", 1, 1);
  const auto &node2 = builder.AddNode("node2", "node", 1, 1);
  const auto &node3 = builder.AddNode("node3", "node", 1, 1);
  builder.AddDataEdge(node0, 0, node2, 0);
  ASSERT_EQ(GraphUtils::ReplaceEdgeSrc(node0->GetOutDataAnchor(0), node2->GetInDataAnchor(0),
                                       node1->GetOutDataAnchor(0)), GRAPH_SUCCESS);
  ASSERT_NE(GraphUtils::ReplaceEdgeSrc(node0->GetOutDataAnchor(0), node2->GetInDataAnchor(0),
                                       node3->GetOutDataAnchor(0)), GRAPH_SUCCESS);

  builder.AddControlEdge(node0, node2);
  ASSERT_EQ(GraphUtils::ReplaceEdgeSrc(node0->GetOutControlAnchor(), node2->GetInControlAnchor(),
                                       node1->GetOutControlAnchor()), GRAPH_SUCCESS);
  ASSERT_NE(GraphUtils::ReplaceEdgeSrc(node0->GetOutControlAnchor(), node2->GetInControlAnchor(),
                                       node3->GetOutControlAnchor()), GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, ReplaceEdgeDst) {
  auto builder = ut::GraphBuilder("root");
  const auto &node0 = builder.AddNode("node0", "node", 1, 1);
  const auto &node1 = builder.AddNode("node1", "node", 1, 1);
  const auto &node2 = builder.AddNode("node2", "node", 1, 1);
  const auto &node3 = builder.AddNode("node3", "node", 1, 1);
  builder.AddDataEdge(node0, 0, node2, 0);
  ASSERT_EQ(GraphUtils::ReplaceEdgeDst(node0->GetOutDataAnchor(0), node2->GetInDataAnchor(0),
                                       node1->GetInDataAnchor(0)), GRAPH_SUCCESS);
  ASSERT_NE(GraphUtils::ReplaceEdgeDst(node0->GetOutDataAnchor(0), node2->GetInDataAnchor(0),
                                       node3->GetInDataAnchor(0)), GRAPH_SUCCESS);

  builder.AddControlEdge(node0, node2);
  ASSERT_EQ(GraphUtils::ReplaceEdgeDst(node0->GetOutControlAnchor(), node2->GetInControlAnchor(),
                                       node1->GetInControlAnchor()), GRAPH_SUCCESS);
  ASSERT_NE(GraphUtils::ReplaceEdgeDst(node0->GetOutControlAnchor(), node2->GetInControlAnchor(),
                                       node3->GetInControlAnchor()), GRAPH_SUCCESS);
}

/*
 *          data0  data1
 *             \    /|
 *              add1 | data2
 *                 \ |  /|
 *                  add2 | data3
 *                     \ |  /|
 *                      add3 |  data4
 *                         \ |  / | \
 *                          add4  | cast1
 *                              \ | / |
 *                              add5  |
 *                                | \ |
 *                                | cast2
 *                                | /
 *                             netoutput
 */
TEST_F(UtestGraphUtils, BuildSubgraphWithNodes) {
  auto builder = ut::GraphBuilder("root");
  const auto &data0 = builder.AddNode("data0", DATA, 1, 1);
  const auto &data1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &data2 = builder.AddNode("data2", DATA, 1, 1);
  const auto &data3 = builder.AddNode("data3", DATA, 1, 1);
  const auto &data4 = builder.AddNode("data4", DATA, 1, 1);

  const auto &add1 = builder.AddNode("add1", "Add", 2, 1);
  const auto &add2 = builder.AddNode("add2", "Add", 2, 1);
  const auto &add3 = builder.AddNode("add3", "Add", 2, 1);
  const auto &add4 = builder.AddNode("add4", "Add", 2, 1);
  const auto &add5 = builder.AddNode("add5", "Add", 2, 1);

  const auto &cast1 = builder.AddNode("cast1", "Cast", 1, 1);
  const auto &cast2 = builder.AddNode("cast2", "Cast", 1, 1);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, add1, 0);
  builder.AddDataEdge(data1, 0, add1, 1);
  builder.AddDataEdge(add1, 0, add2, 0);
  builder.AddDataEdge(data2, 0, add2, 1);
  builder.AddDataEdge(add2, 0, add3, 0);
  builder.AddDataEdge(data3, 0, add3, 1);
  builder.AddDataEdge(add3, 0, add4, 0);
  builder.AddDataEdge(data4, 0, add4, 1);
  builder.AddDataEdge(data4, 0, cast1, 0);
  builder.AddDataEdge(add4, 0, add5, 0);
  builder.AddDataEdge(cast1, 0, add5, 1);
  builder.AddDataEdge(add5, 0, cast2, 0);
  builder.AddDataEdge(cast2, 0, netoutput, 0);

  builder.AddControlEdge(data1, add2);
  builder.AddControlEdge(data2, add3);
  builder.AddControlEdge(data3, add4);
  builder.AddControlEdge(data4, add5);
  builder.AddControlEdge(add5, netoutput);
  builder.AddControlEdge(cast1, cast2);

  ASSERT_EQ(GraphUtils::BuildSubgraphWithNodes(nullptr, {}, "subgraph1"), nullptr);

  const auto &graph = builder.GetGraph();
  ASSERT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  ASSERT_EQ(GraphUtils::BuildSubgraphWithNodes(graph, {}, "subgraph1"), nullptr);

  std::set<NodePtr> nodes = { data1, add2, add3, add4, add5, cast2 };
  ASSERT_EQ(GraphUtils::BuildSubgraphWithNodes(graph, nodes, "subgraph1"), nullptr);

  ASSERT_TRUE(AttrUtils::SetStr(graph, "_session_graph_id", "_session_graph_id"));
  const auto &subgraph1 = GraphUtils::BuildSubgraphWithNodes(graph, nodes, "subgraph1");
  ASSERT_NE(subgraph1, nullptr);
  ASSERT_EQ(subgraph1->GetParentGraph(), graph);
  ASSERT_TRUE(subgraph1->HasAttr("_session_graph_id"));
  ASSERT_FALSE(IfNodeExist(graph, [](const NodePtr &node) { return node->GetName() == "data1"; }));
  ASSERT_FALSE(IfNodeExist(graph, [](const NodePtr &node) { return node->GetName() == "add2"; }));
  ASSERT_FALSE(IfNodeExist(graph, [](const NodePtr &node) { return node->GetName() == "add3"; }));
  ASSERT_FALSE(IfNodeExist(graph, [](const NodePtr &node) { return node->GetName() == "add4"; }));
  ASSERT_FALSE(IfNodeExist(graph, [](const NodePtr &node) { return node->GetName() == "add5"; }));
  ASSERT_FALSE(IfNodeExist(graph, [](const NodePtr &node) { return node->GetName() == "cast2"; }));
  ASSERT_TRUE(IfNodeExist(graph, [](const NodePtr &node) { return node->GetType() == PARTITIONEDCALL; }));
  ASSERT_EQ(graph->GetAllSubgraphs().size(), 1);

  ASSERT_EQ(GraphUtils::BuildSubgraphWithNodes(graph, {cast1}, "subgraph1"), nullptr);
}

TEST_F(UtestGraphUtils, UnfoldSubgraph) {
  ComputeGraphPtr graph;
  ComputeGraphPtr subgraph;
  BuildGraphForUnfold(graph, subgraph);
  ASSERT_NE(graph, nullptr);
  ASSERT_NE(subgraph, nullptr);

  const auto &filter = [](const ComputeGraphPtr &graph) {
    const auto &parent_node = graph->GetParentNode();
    if (parent_node == nullptr || parent_node->GetOpDesc() == nullptr) {
      return false;
    }
    if ((parent_node->GetType() != PARTITIONEDCALL) ||
        (parent_node->GetOpDesc()->GetSubgraphInstanceNames().size() != 1)) {
      return false;
    }
    return graph->GetGraphUnknownFlag();
  };
  ASSERT_EQ(GraphUtils::UnfoldSubgraph(subgraph, filter), GRAPH_SUCCESS);

  ASSERT_EQ(graph->GetAllSubgraphs().size(), 1);
  ASSERT_FALSE(graph->GetAllSubgraphs()[0]->GetGraphUnknownFlag());
}

TEST_F(UtestGraphUtils, UnfoldSubgraph_InnerDataHasOutControl) {
  ComputeGraphPtr graph;
  ComputeGraphPtr subgraph;
  BuildGraphForUnfoldWithControlEdge(graph, subgraph);
  ASSERT_NE(graph, nullptr);
  ASSERT_NE(subgraph, nullptr);

  const auto &filter = [](const ComputeGraphPtr &graph) {
    const auto &parent_node = graph->GetParentNode();
    if (parent_node == nullptr || parent_node->GetOpDesc() == nullptr) {
      return false;
    }
    if (parent_node->GetType() == PARTITIONEDCALL) {
      return true;
    }
    return false;
  };
  ASSERT_EQ(GraphUtils::UnfoldSubgraph(subgraph, filter), GRAPH_SUCCESS);
  ASSERT_EQ(graph->GetAllSubgraphs().size(), 0);
  ASSERT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);

}

TEST_F(UtestGraphUtils, UnfoldSubgraph_ForPartition) {
  ComputeGraphPtr graph;
  ComputeGraphPtr subgraph;
  BuildGraphForUnfold(graph, subgraph);
  ASSERT_NE(graph, nullptr);
  ASSERT_NE(subgraph, nullptr);
  std::vector<NodePtr> inputs;
  std::vector<NodePtr> outputs;
  const auto &new_graph = GraphUtils::CloneGraph(graph, "", inputs, outputs);
  const auto &node_size_before_unfold = new_graph->GetDirectNode().size();
  const auto &filter = [](const ComputeGraphPtr &graph) {
    const auto &parent_node = graph->GetParentNode();
    if (parent_node == nullptr || parent_node->GetOpDesc() == nullptr) {
      return false;
    }
    if ((parent_node->GetType() != PARTITIONEDCALL) ||
        (parent_node->GetOpDesc()->GetSubgraphInstanceNames().size() != 1)) {
      return false;
    }
    return graph->GetGraphUnknownFlag();
  };
  ASSERT_EQ(GraphUtils::UnfoldGraph(subgraph, new_graph, new_graph->FindNode(subgraph->GetParentNode()->GetName()),
                                       filter), GRAPH_SUCCESS);
  ASSERT_NE(node_size_before_unfold, new_graph->GetDirectNode().size());
}

TEST_F(UtestGraphUtils, GetIndependentCompileGraphs) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &partitioned_call0 = root_builder.AddNode("PartitionedCall", "PartitionedCall", 0, 0);
  const auto &root_graph = root_builder.GetGraph();
  (void)AttrUtils::SetBool(*root_graph, ATTR_NAME_PIPELINE_PARTITIONED, true);

  auto sub_builder1 = ut::GraphBuilder("sub1");
  const auto &data1 = sub_builder1.AddNode("Data", "Data", 0, 0);
  const auto &sub_graph1 = sub_builder1.GetGraph();
  root_graph->AddSubGraph(sub_graph1);
  sub_graph1->SetParentNode(partitioned_call0);
  sub_graph1->SetParentGraph(root_graph);
  partitioned_call0->GetOpDesc()->AddSubgraphName("sub1");
  partitioned_call0->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");

  std::vector<ComputeGraphPtr> independent_compile_subgraphs;
  ASSERT_EQ(GraphUtils::GetIndependentCompileGraphs(root_graph, independent_compile_subgraphs), GRAPH_SUCCESS);
  ASSERT_EQ(independent_compile_subgraphs.size(), 1);
  ASSERT_EQ(independent_compile_subgraphs[0]->GetName(), "sub1");

  (void)AttrUtils::SetBool(*root_graph, ATTR_NAME_PIPELINE_PARTITIONED, false);
  independent_compile_subgraphs.clear();
  ASSERT_EQ(GraphUtils::GetIndependentCompileGraphs(root_graph, independent_compile_subgraphs), GRAPH_SUCCESS);
  ASSERT_EQ(independent_compile_subgraphs.size(), 1);
  ASSERT_EQ(independent_compile_subgraphs[0]->GetName(), "root");
}

TEST_F(UtestGraphUtils, InsertNodeAfter) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &graph0 = graph_builder0.GetGraph();

  auto graph_builder1 = ut::GraphBuilder("test_graph1");
  const auto &node1 = graph_builder1.AddNode("data1", DATA, 1, 1);
  const auto &graph1 = graph_builder1.GetGraph();

  std::vector<ComputeGraphPtr> independent_compile_subgraphs;
  ASSERT_EQ(GraphUtils::InsertNodeAfter(node0->GetOutDataAnchor(0), {}, node1, 0, 0), GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, CheckDumpGraphNum) {
  std::map<std::string, std::string> session_option{{"ge.maxDumpFileNum", "3"}};
  GetThreadLocalContext().SetSessionOption(session_option);
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &graph0 = graph_builder0.GetGraph();
  GraphUtils::DumpGEGrph(graph0, "./", "1");
  GraphUtils::DumpGEGrph(graph0, "./", "1");
  GraphUtils::DumpGEGrph(graph0, "./", "1");
  GraphUtils::DumpGEGrph(graph0, "./", "1");
}

TEST_F(UtestGraphUtils, CopyRootComputeGraph) {
  auto graph = BuildGraphWithSubGraph();
  // check origin graph size
  ASSERT_EQ(graph->GetAllNodesSize(), 3);
  ComputeGraphPtr dst_compute_graph = std::make_shared<ComputeGraph>(ComputeGraph("dst"));
  // test copy root graph success
  auto ret = GraphUtils::CopyComputeGraph(graph, dst_compute_graph);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ASSERT_EQ(dst_compute_graph->GetAllNodesSize(), 3);
  // test copy subgraph failed
  auto sub1_graph = graph->GetSubgraph("sub1");
  ret = GraphUtils::CopyComputeGraph(sub1_graph, dst_compute_graph);
  ASSERT_EQ(ret, GRAPH_FAILED);

  // test copy dst compute_graph null
  ComputeGraphPtr empty_dst_compute_graph;
  ret = GraphUtils::CopyComputeGraph(graph, empty_dst_compute_graph);
  ASSERT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, DumpGraphByPath) {
  auto ge_tensor = std::make_shared<GeTensor>();
  uint8_t data_buf[4096] = {0};
  data_buf[0] = 7;
  data_buf[10] = 8;
  ge_tensor->SetData(data_buf, 4096);

  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 0, 1);
  auto const_node = builder.AddNode("Const", "Const", 0, 1);
  AttrUtils::SetTensor(const_node->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, ge_tensor);
  auto add_node = builder.AddNode("Add", "Add", 2, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data_node, 0, add_node, 0);
  builder.AddDataEdge(const_node, 0, add_node, 0);
  builder.AddDataEdge(add_node, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  // test dump_level 0
  auto ret = GraphUtils::DumpGEGraphByPath(graph, "./not-exists-path/test_graph_0.txt", ge::DumpLevel::NO_DUMP);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  ret = GraphUtils::DumpGEGraphByPath(graph, "/", ge::DumpLevel::NO_DUMP);
  ASSERT_EQ((ret != 0), true);
  ret = GraphUtils::DumpGEGraphByPath(graph, "test_graph_0.txt", ge::DumpLevel::NO_DUMP);
  ASSERT_EQ((ret != 0), true);
  ret = GraphUtils::DumpGEGraphByPath(graph, "./test_graph_0.txt", ge::DumpLevel::NO_DUMP);
  ASSERT_EQ(ret, 0);
  ComputeGraphPtr com_graph0 = std::make_shared<ComputeGraph>("TestGraph0");
  bool state = GraphUtils::LoadGEGraph("./test_graph_0.txt", *com_graph0);
  ASSERT_EQ(state, true);
  ASSERT_EQ(com_graph0->GetAllNodesSize(), 4);
  for (auto &node_ptr : com_graph0->GetAllNodes()) {
    ASSERT_EQ((node_ptr == nullptr), false);
    if (node_ptr->GetType() == CONSTANT) {
      auto op_desc = node_ptr->GetOpDesc();
      ASSERT_EQ((op_desc == nullptr), false);
      ConstGeTensorPtr ge_tensor_ptr;
      ASSERT_EQ(AttrUtils::GetTensor(op_desc, ATTR_NAME_WEIGHTS, ge_tensor_ptr), false);
    }
  }

  // test dump_level 1
  ret = GraphUtils::DumpGEGraphByPath(graph, "./test_graph_1.txt", ge::DumpLevel::DUMP_ALL);
  ASSERT_EQ(ret, 0);
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("TestGraph1");
  state = GraphUtils::LoadGEGraph("./test_graph_1.txt", *com_graph1);
  ASSERT_EQ(state, true);
  ASSERT_EQ(com_graph1->GetAllNodesSize(), 4);
  for (auto &node_ptr : com_graph1->GetAllNodes()) {
    ASSERT_EQ((node_ptr == nullptr), false);
    if (node_ptr->GetType() == CONSTANT) {
      auto op_desc = node_ptr->GetOpDesc();
      ASSERT_EQ((op_desc == nullptr), false);
      ConstGeTensorPtr ge_tensor_ptr;
      ASSERT_EQ(AttrUtils::GetTensor(op_desc, ATTR_NAME_WEIGHTS, ge_tensor_ptr), true);
      ASSERT_EQ((ge_tensor_ptr == nullptr), false);
      const TensorData tensor_data = ge_tensor_ptr->GetData();
      const uint8_t *buff = tensor_data.GetData();
      ASSERT_EQ((buff == nullptr), false);
      ASSERT_EQ(buff[0], 7);
      ASSERT_EQ(buff[10], 8);
    }
  }
}

TEST_F(UtestGraphUtils, AddEdgeAnchorPtrIsNull) {
  AnchorPtr src;
  AnchorPtr dst;
  int ret = GraphUtils::AddEdge(src, dst);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, AddEdgeAnchorPtrSuccess) {
  auto builder = ut::GraphBuilder("root");
  const auto &node0 = builder.AddNode("node0", "node", 1, 1);
  const auto &node1 = builder.AddNode("node1", "node", 1, 1);
  int ret = GraphUtils::AddEdge(node0->GetOutAnchor(0), node1->GetInAnchor(0));
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  int ret2 = GraphUtils::AddEdge(node0->GetOutAnchor(0), node1->GetInControlAnchor());
  EXPECT_EQ(ret2, GRAPH_SUCCESS);

  int ret3 = GraphUtils::AddEdge(node0->GetOutControlAnchor(), node1->GetInControlAnchor());
  EXPECT_EQ(ret3, GRAPH_SUCCESS);

  int ret4 = GraphUtils::AddEdge(node0->GetOutControlAnchor(), node1->GetInDataAnchor(0));
  EXPECT_EQ(ret4, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, AddEdgeControlAnchorPtrIsNull) {
  OutControlAnchorPtr src;
  InControlAnchorPtr dst;
  int ret = GraphUtils::AddEdge(src, dst);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, AddEdgeDataAnchorPtrIsNull) {
  OutDataAnchorPtr src;
  InControlAnchorPtr dst;
  int ret = GraphUtils::AddEdge(src, dst);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RemoveEdgeAnchorPtrIsNull) {
  AnchorPtr src;
  AnchorPtr dst;
  int ret = GraphUtils::RemoveEdge(src, dst);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RemoveEdgeOutDataAnchorPtrIsNull) {
  OutDataAnchorPtr src;
  InControlAnchorPtr  dst;
  int ret = GraphUtils::RemoveEdge(src, dst);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RemoveEdgeFail) {
  auto builder = ut::GraphBuilder("root");
  const auto &node0 = builder.AddNode("node0", "node", 1, 1);
  const auto &node1 = builder.AddNode("node1", "node", 1, 1);
  builder.AddDataEdge(node0, 0, node1, 0);
  builder.AddControlEdge(node0, node1);
  int ret = GraphUtils::RemoveEdge(node0->GetOutDataAnchor(0), node1->GetInControlAnchor());
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, InsertNodeBetweenDataAnchorsSuccess) {
  auto builder = ut::GraphBuilder("root");
  const auto &node0 = builder.AddNode("node0", "node", 1, 1);
  const auto &node1 = builder.AddNode("node1", "node", 1, 1);
  const auto &node2 = builder.AddNode("node2", "node", 1, 1);
  NodePtr new_node(node1);
  builder.AddDataEdge(node0, 0, node2, 0);
  builder.AddControlEdge(node0, node2);
  int ret = GraphUtils::InsertNodeBetweenDataAnchors(node0->GetOutDataAnchor(0),
                                                     node2->GetInDataAnchor(0), new_node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, RemoveSubgraphRecursivelyRemoveNodeIsNull) {
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("Test0");
  NodePtr remove_node;
  int ret = GraphUtils::RemoveSubgraphRecursively(compute_graph, remove_node);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RemoveSubgraphRecursivelyFail) {
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("Test0");
  auto builder = ut::GraphBuilder("root");
  const auto &node0 = builder.AddNode("node0", "node", 1, 1);
  NodePtr remove_node(node0);
  int ret = GraphUtils::RemoveSubgraphRecursively(compute_graph, remove_node);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RemoveNodeWithoutRelinkNodePtrIsNull) {
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("Test0");
  NodePtr remove_node;
  int ret = GraphUtils::RemoveNodeWithoutRelink(compute_graph, remove_node);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RemoveNodeWithoutRelinkFail) {
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("Test0");
  auto builder = ut::GraphBuilder("root");
  const auto &node0 = builder.AddNode("node0", "node", 1, 1);
  NodePtr remove_node(node0);
  int ret = GraphUtils::RemoveNodeWithoutRelink(compute_graph, remove_node);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, InsertNodeAfterAddEdgefail) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &graph0 = graph_builder0.GetGraph();
  std::vector<InDataAnchorPtr> dsts;
  dsts.push_back(node0->GetInDataAnchor(0));
  int ret = GraphUtils::InsertNodeAfter(node0->GetOutDataAnchor(0), dsts, node0, 1, 0);
  EXPECT_EQ(ret, GRAPH_FAILED);

  int ret2 = GraphUtils::InsertNodeAfter(node0->GetOutDataAnchor(0), dsts, node0, 0, 1);
  EXPECT_EQ(ret2, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, InsertNodeAfterTypeIsSwitch) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", SWITCH, 1, 1);
  const auto &graph0 = graph_builder0.GetGraph();
  std::vector<InDataAnchorPtr> dsts;
  dsts.push_back(node0->GetInDataAnchor(0));
  int ret = GraphUtils::InsertNodeAfter(node0->GetOutDataAnchor(0), dsts, node0, 0, 0);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, InsertNodeAfterSrcOwnerComputeGraphNotEqualDstOwnerComputeGraph) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &graph0 = graph_builder0.GetGraph();

  auto graph_builder1 = ut::GraphBuilder("test_graph1");
  const auto &node1 = graph_builder1.AddNode("data1", DATA, 1, 1);
  const auto &graph1 = graph_builder1.GetGraph();

  std::vector<InDataAnchorPtr> dsts;
  dsts.push_back(node1->GetInDataAnchor(0));
  int ret = GraphUtils::InsertNodeAfter(node0->GetOutDataAnchor(0), dsts, node0, 0, 0);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, InsertNodeBeforeGetOwnerComputeGraphFail) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &graph0 = graph_builder0.GetGraph();

  auto graph_builder1 = ut::GraphBuilder("test_graph1");
  const auto &node1 = graph_builder1.AddNode("data1", DATA, 1, 1);
  const auto &graph1 = graph_builder1.GetGraph();

  int ret = GraphUtils::InsertNodeBefore(node0->GetInDataAnchor(0), node1, 0, 0);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, InsertNodeBeforeInsertCodeGetInDataAnchorFail) {
  auto builder = ut::GraphBuilder("test");
  const auto &var = builder.AddNode("var", VARIABLE, 0, 1);
  const auto &assign = builder.AddNode("assign", "Assign", 1, 1);
  const auto &allreduce = builder.AddNode("allreduce", "HcomAllReduce", 1, 1);
  const auto &atomic_clean = builder.AddNode("atomic_clean", ATOMICADDRCLEAN, 0, 0);
  const auto &netoutput1 = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  const auto &identity = builder.AddNode("identity", "Identity", 1, 1);

  builder.AddDataEdge(var, 0, assign, 0);
  builder.AddDataEdge(var,0,allreduce,0);
  builder.AddControlEdge(assign, allreduce);
  builder.AddControlEdge(atomic_clean, allreduce);
  auto graph = builder.GetGraph();

  int ret = GraphUtils::InsertNodeBefore(allreduce->GetInDataAnchor(0), identity, 0, 5);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RemoveJustNodeNodeIsNull) {
  ComputeGraph compute_graph("test_graph0");
  int ret = GraphUtils::RemoveJustNode(compute_graph, nullptr);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RemoveJustNodeFail) {
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("Test0");
  auto graph_builder0 = ut::GraphBuilder("Test0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  int ret = GraphUtils::RemoveJustNode(compute_graph, node0);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, MatchDumpStrIsFalse) {
  std::string suffix;
  bool ret = GraphUtils::MatchDumpStr(suffix);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestGraphUtils, LoadGEGraphComputeGraphIsNull) {
  char_t *file = nullptr;
  ge::ComputeGraph compute_graph("");
  bool ret = GraphUtils::LoadGEGraph(file, compute_graph);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestGraphUtils, LoadGEGraphFileIsNull) {
  char_t *file = nullptr;
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("Test0");
  bool ret = GraphUtils::LoadGEGraph(file, compute_graph);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestGraphUtils, LoadGEGraphComputeGraphPtrSuccess) {
  char_t *file = "./test_graph_0.txt";
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("");
  bool ret = GraphUtils::LoadGEGraph(file, compute_graph);
  EXPECT_EQ(ret, true);
}

TEST_F(UtestGraphUtils, ReadProtoFromTextFileFileIsNull) {
  google::protobuf::Message *proto;
  bool ret = GraphUtils::ReadProtoFromTextFile(nullptr, proto);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestGraphUtils, IsolateNodeNodeIsNull) {
  NodePtr node;
  std::vector<int> io_map = {1, 2, 3};
  int ret = GraphUtils::IsolateNode(node, io_map);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphUtils, IsolateNodeIoMapIsNull) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  std::vector<int> io_map;
  int ret = GraphUtils::IsolateNode(node0, io_map);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, IsolateNodeIoMapSizeIsGreaterThanOutDataAnchorsSize) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  std::vector<int> io_map = {1, 2, 3, 4};
  int ret = GraphUtils::IsolateNode(node0, io_map);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphUtils, IsolateNodeOutDataAnchorsIsNull) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 0);
  std::vector<int> io_map = {1};
  int ret = GraphUtils::IsolateNode(node0, io_map);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphUtils, IsolateNodeInDataAnchorsIsNull) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 0, 1);
  std::vector<int> io_map = {1};
  int ret = GraphUtils::IsolateNode(node0, io_map);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphUtils, IsolateNodeInitializerListTest) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  std::initializer_list<int> io_map;
  int ret = GraphUtils::IsolateNode(node0, io_map);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, ReplaceNodeDataAnchorsNodeIsNull) {
  NodePtr new_node;
  NodePtr old_node;
  std::vector<int> inputs_map = {1, 2};
  std::vector<int> outputs_map = {1, 2};
  int ret = GraphUtils::ReplaceNodeDataAnchors(new_node, old_node, inputs_map, outputs_map);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphUtils, ReplaceNodeDataAnchorsReplaceOutDataAnchorsFail) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &new_node = graph_builder0.AddNode("data1", DATA, 1, 1);
  const auto &old_node = graph_builder0.AddNode("data0", DATA, 0, 0);
  std::vector<int> inputs_map;
  std::vector<int> outputs_map = {1, 2};
  int ret = GraphUtils::ReplaceNodeDataAnchors(new_node, old_node, inputs_map, outputs_map);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, ReplaceNodeDataAnchorsReplaceInDataAnchorsFail) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &new_node = graph_builder0.AddNode("data1", DATA, 1, 1);
  const auto &old_node = graph_builder0.AddNode("data0", DATA, 0, 0);
  std::vector<int> inputs_map = {1, 2};
  std::vector<int> outputs_map;
  int ret = GraphUtils::ReplaceNodeDataAnchors(new_node, old_node, inputs_map, outputs_map);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, ReplaceNodeDataAnchorsSuccess) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &new_node = graph_builder0.AddNode("data1", DATA, 1, 1);
  const auto &old_node = graph_builder0.AddNode("data0", DATA, 0, 0);
  std::vector<int> inputs_map;
  std::vector<int> outputs_map;
  int ret = GraphUtils::ReplaceNodeDataAnchors(new_node, old_node, inputs_map, outputs_map);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, IsolateNodeOneIONodeIsNull) {
  NodePtr node;
  int ret = GraphUtils::IsolateNodeOneIO(node);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphUtils, IsolateNodeOneIOInDataIs0) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node = graph_builder0.AddNode("data1", DATA, 0, 1);
  int ret = GraphUtils::IsolateNodeOneIO(node);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphUtils, IsolateNodeOneIOOutDataIs0) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node = graph_builder0.AddNode("data1", DATA, 1, 0);
  int ret = GraphUtils::IsolateNodeOneIO(node);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphUtils, IsolateNodeOneIOSuccess) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node = graph_builder0.AddNode("data1", DATA, 1, 1);
  int ret = GraphUtils::IsolateNodeOneIO(node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, ReplaceNodeAnchorsNodeIsNull) {
  NodePtr new_node;
  NodePtr old_node;
  std::vector<int> inputs_map = {1, 2};
  std::vector<int> outputs_map = {1, 2};
  int ret = GraphUtils::ReplaceNodeAnchors(new_node, old_node, inputs_map, outputs_map);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphUtils, ReplaceNodeAnchorsReplaceNodeDataAnchorsFail) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &new_node = graph_builder0.AddNode("data1", DATA, 1, 1);
  const auto &old_node = graph_builder0.AddNode("data0", DATA, 0, 0);
  std::vector<int> inputs_map = {1, 2};
  std::vector<int> outputs_map = {1, 2};
  int ret = GraphUtils::ReplaceNodeAnchors(new_node, old_node, inputs_map, outputs_map);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, ReplaceNodeAnchorsSuccess) {
  auto builder = ut::GraphBuilder("test_graph0");
  const auto &new_node = builder.AddNode("data1", "node", 1, 1);
  const auto &old_node = builder.AddNode("data0", "node", 1, 1);
  builder.AddDataEdge(new_node, 0, old_node, 0);
  builder.AddControlEdge(new_node, old_node);
  std::vector<int> inputs_map = {0};
  std::vector<int> outputs_map = {0};
  int ret = GraphUtils::ReplaceNodeAnchors(new_node, old_node, inputs_map, outputs_map);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, ReplaceNodeAnchorsInitializerListTest) {
  auto builder = ut::GraphBuilder("test_graph0");
  const auto &new_node = builder.AddNode("data1", "node", 1, 1);
  const auto &old_node = builder.AddNode("data0", "node", 1, 1);
  builder.AddDataEdge(new_node, 0, old_node, 0);
  builder.AddControlEdge(new_node, old_node);
  std::initializer_list<int> inputs_map;
  std::initializer_list<int> outputs_map;
  int ret = GraphUtils::ReplaceNodeAnchors(new_node, old_node, inputs_map, outputs_map);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, ReplaceNodeDataAnchorsInitializerListTest) {
  auto builder = ut::GraphBuilder("test_graph0");
  const auto &new_node = builder.AddNode("data1", DATA, 1, 1);
  const auto &old_node = builder.AddNode("data0", DATA, 1, 1);
  std::initializer_list<int> inputs_map;
  std::initializer_list<int> outputs_map;
  int ret = GraphUtils::ReplaceNodeDataAnchors(new_node, old_node, inputs_map, outputs_map);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, CopyInCtrlEdgesNodeIsNull) {
  NodePtr src_node;
  NodePtr dst_node;
  int ret = GraphUtils::CopyInCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphUtils, CopyInCtrlEdgesSrcCtrlInNodesIsEmpty) {
  auto builder = ut::GraphBuilder("test_graph0");
  const auto &src_node = builder.AddNode("data0", "data", 1, 1);
  NodePtr dst_node = builder.AddNode("data1", "data", 1, 1);
  int ret = GraphUtils::CopyInCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, CopyInCtrlEdgesSuccess) {
  auto builder = ut::GraphBuilder("test");
  const auto &src_node = builder.AddNode("src_node", "node", 1, 1);
  NodePtr dst_node = builder.AddNode("dst_node", "node", 1, 1);
  builder.AddDataEdge(src_node, 0, dst_node, 0);
  builder.AddControlEdge(src_node, dst_node);
  int ret = GraphUtils::CopyInCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, MoveInCtrlEdgesNodeIsNull) {
  NodePtr src_node;
  NodePtr dst_node;
  int ret = GraphUtils::MoveInCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, MoveInCtrlEdgesSuccess) {
  auto builder = ut::GraphBuilder("test_graph0");
  const auto &src_node = builder.AddNode("data0", "data", 1, 1);
  NodePtr dst_node = builder.AddNode("data1", "data", 1, 1);
  int ret = GraphUtils::MoveInCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, CopyOutCtrlEdgesNodeIsNull) {
  NodePtr src_node;
  NodePtr dst_node;
  int ret = GraphUtils::CopyOutCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, CopyOutCtrlEdgesOutCtrlNodesIsEmpty) {
  auto builder = ut::GraphBuilder("test_graph0");
  const auto &src_node = builder.AddNode("data0", "data", 1, 1);
  NodePtr dst_node = builder.AddNode("data1", "data", 1, 1);
  int ret = GraphUtils::CopyOutCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, CopyOutCtrlEdgesSuccess) {
  auto builder = ut::GraphBuilder("test_graph0");
  const auto &src_node = builder.AddNode("src_node", NETOUTPUT, 1, 1);
  NodePtr dst_node = builder.AddNode("dst_node", NETOUTPUT, 1, 1);
  auto graph = builder.GetGraph();
  builder.AddControlEdge(src_node, dst_node);

  int ret = GraphUtils::CopyOutCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, MoveOutCtrlEdgesNodeIsNull) {
  auto builder = ut::GraphBuilder("test_graph0");
  NodePtr src_node;
  NodePtr dst_node;
  int ret = GraphUtils::MoveOutCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, MoveOutCtrlEdgesSuccess) {
  auto builder = ut::GraphBuilder("test_graph0");
  NodePtr src_node = builder.AddNode("src_node", NETOUTPUT, 1, 1);
  NodePtr dst_node = builder.AddNode("dst_node", NETOUTPUT, 1, 1);
  int ret = GraphUtils::MoveOutCtrlEdges(src_node, dst_node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, AppendInputNodeSuccess) {
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("Test0");
  auto builder = ut::GraphBuilder("Test1");
  const auto &node = builder.AddNode("node", "node", 1, 1);
  int ret = GraphUtils::AppendInputNode(compute_graph, node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, CopyGraphDstrGraphIsNull) {
  Graph src_graph("test0");
  Graph dst_graph("");
  int ret = GraphUtils::CopyGraph(src_graph, dst_graph);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST_F(UtestGraphUtils, CopyComputeGraphDepthGreaterThanKCopyGraphMaxRecursionDepth) {
  ComputeGraphPtr src_compute_graph = std::make_shared<ComputeGraph>("Test0");
  ComputeGraphPtr dst_compute_graph = std::make_shared<ComputeGraph>("Test1");
  std::map<ConstNodePtr, NodePtr> node_old_2_new;
  std::map<ConstOpDescPtr, OpDescPtr> op_desc_old_2_new;
  int32_t depth = 20;
  int ret = 
      GraphUtils::CopyComputeGraph(src_compute_graph, dst_compute_graph, node_old_2_new, op_desc_old_2_new, depth);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, CopyMembersSrcComputerGraphIsNull) {
  ComputeGraphPtr dst_compute_graph = std::make_shared<ComputeGraph>("Test1");
  std::unordered_map<std::string, NodePtr> all_new_nodes;
  int ret = 
      GraphUtils::CopyMembers(nullptr, dst_compute_graph, all_new_nodes);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, CopyMembersDstComputerGraphIsNull) {
  ComputeGraphPtr src_compute_graph = std::make_shared<ComputeGraph>("Test0");
  ComputeGraphPtr dst_compute_graph;
  std::unordered_map<std::string, NodePtr> all_new_nodes;
  int ret = GraphUtils::CopyMembers(src_compute_graph, dst_compute_graph, all_new_nodes);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, CloneGraph) {
  auto builder = ut::GraphBuilder("Test1");
  const auto &node0 = builder.AddNode("node0", DATA, 1, 1);
  const auto &node1 = builder.AddNode("node1", NETOUTPUT, 1, 1);
  auto graph = builder.GetGraph();
  (void) AttrUtils::SetStr(graph, ATTR_NAME_SESSION_GRAPH_ID, "0");
  std::string prefix;
  std::vector<NodePtr> input_nodes;
  std::vector<NodePtr> output_nodes;
  std::unordered_map<std::string, NodePtr> all_new_nodes;
  ComputeGraphPtr new_compute_graph = GraphUtils::CloneGraph(graph, prefix, input_nodes, output_nodes);
  EXPECT_NE(new_compute_graph, nullptr);
}

TEST_F(UtestGraphUtils, CopyTensorAttrsDstDescIsNull) {
  OpDescPtr dst_desc;
  auto builder = ut::GraphBuilder("Test1");
  const auto &src_node = builder.AddNode("src_node", DATA, 1, 1);
  int ret = GraphUtils::CopyTensorAttrs(dst_desc, src_node);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, CopyTensorAttrsSrcNodeIsNull) {
  OpDescPtr dst_desc = std::make_shared<OpDesc>("test", "test");
  NodePtr src_node;
  int ret = GraphUtils::CopyTensorAttrs(dst_desc, src_node);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, CopyTensorAttrsFail) {
  OpDescPtr dst_desc = std::make_shared<OpDesc>();
  auto builder = ut::GraphBuilder("Test1");
  const auto &src_node = builder.AddNode("src_node", DATA, 1, 1);
  int ret = GraphUtils::CopyTensorAttrs(dst_desc, src_node);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RelinkGraphEdgesNodeIsNull) {
  NodePtr node;
  std::string prefix;
  std::unordered_map<std::string, NodePtr> all_nodes;
  int ret = GraphUtils::RelinkGraphEdges(node, prefix, all_nodes);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RelinkGraphEdgesAllNodesIsNull) {
  auto builder = ut::GraphBuilder("Test1");
  const auto &node = builder.AddNode("node", DATA, 1, 1);
  std::string prefix;
  std::unordered_map<std::string, NodePtr> all_nodes;
  int ret = GraphUtils::RelinkGraphEdges(node, prefix, all_nodes);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, RelinkGraphEdgesFail) {
  auto builder = ut::GraphBuilder("Test1");
  const auto &node1 = builder.AddNode("node1", DATA, 1, 1);
  const auto &node2 = builder.AddNode("node2", DATA, 1, 1);
  std::string prefix;
  std::unordered_map<std::string, NodePtr> all_nodes;
  all_nodes.insert(make_pair("node2", node2));
  int ret = GraphUtils::RelinkGraphEdges(node1, prefix, all_nodes);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, GetRefMappingSuccess) {
  auto builder = ut::GraphBuilder("Test1");
  auto graph = builder.GetGraph();
  std::map<std::string, std::list<NodeIndexIO>> symbol_to_anchors;
  std::map<std::string, std::string> anchor_to_symbol;
  int ret = GraphUtils::GetRefMapping(graph, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, FindNodeFromAllNodesGraphIsNull) {
  ComputeGraphPtr graph;
  std::string name;
  NodePtr node = GraphUtils::FindNodeFromAllNodes(graph, name);
  EXPECT_EQ(node, nullptr);
}

TEST_F(UtestGraphUtils, FindNodeFromAllNodesSuccess) {
  auto builder = ut::GraphBuilder("Test1");
  const auto &node1 = builder.AddNode("node1", DATA, 1, 1);
  auto graph = builder.GetGraph();
  std::string name = "node1";
  NodePtr node = GraphUtils::FindNodeFromAllNodes(graph, name);
  EXPECT_EQ(node->GetName(), "node1");
}

TEST_F(UtestGraphUtils, FindNodeFromAllNodesNameIsNull) {
  auto builder = ut::GraphBuilder("Test1");
  auto graph = builder.GetGraph();
  std::string name;
  NodePtr node = GraphUtils::FindNodeFromAllNodes(graph, name);
  EXPECT_EQ(node, nullptr);
}

TEST_F(UtestGraphUtils, HandleInAnchorMappingSuccess) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("Test0");
  auto builder = ut::GraphBuilder("Test1");
  const auto &node1 = builder.AddNode("node1", NETOUTPUT, 1, 1);
  std::map<std::string, std::list<NodeIndexIO>> symbol_to_anchors;
  std::map<std::string, std::string> anchor_to_symbol;
  int ret = GraphUtils::HandleInAnchorMapping(graph, node1, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, HandleInAnchorMappingNodeTypeIsMERGE) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("Test0");
  auto builder = ut::GraphBuilder("Test1");
  const auto &node1 = builder.AddNode("node1", MERGE, 1, 1);
  std::map<std::string, std::list<NodeIndexIO>> symbol_to_anchors;
  std::map<std::string, std::string> anchor_to_symbol;
  int ret = GraphUtils::HandleInAnchorMapping(graph, node1, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, HandleSubgraphInputFail) {
  auto builder = ut::GraphBuilder("Test1");
  const auto &node1 = builder.AddNode("node1", DATA, 1, 1);
  std::map<std::string, std::list<NodeIndexIO>> symbol_to_anchors;
  std::map<std::string, std::string> anchor_to_symbol;
  int ret = GraphUtils::HandleSubgraphInput(node1, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, HandleSubgraphInputUpdateRefMappingFail) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &var1 = builder.AddNode("var1", VARIABLEV2, 1, 1);
  const auto &func = builder.AddNode("func", PARTITIONEDCALL, 4, 1);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(input1, 0, func, 0);
  builder.AddDataEdge(var1, 0, func, 1);
  builder.AddDataEdge(func, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  graph->SetParentNode(func);

  AttrUtils::SetInt(input1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  std::map<std::string, std::list<NodeIndexIO>> symbol_to_anchors;
  std::map<std::string, std::string> anchor_to_symbol;
  int ret = GraphUtils::HandleSubgraphInput(input1, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, HandleSubgraphInputSuccess) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &var1 = builder.AddNode("var1", VARIABLEV2, 1, 1);
  const auto &func = builder.AddNode("func", PARTITIONEDCALL, 4, 1);
  auto graph = builder.GetGraph();
  graph->SetParentNode(func);

  AttrUtils::SetInt(input1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  std::map<std::string, std::list<NodeIndexIO>> symbol_to_anchors;
  std::map<std::string, std::string> anchor_to_symbol;
  int ret = GraphUtils::HandleSubgraphInput(input1, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, HandleMergeInputPeerOutAnchorIsNull) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &var1 = builder.AddNode("var1", VARIABLEV2, 1, 1);
  const auto &func = builder.AddNode("func", PARTITIONEDCALL, 4, 1);
  auto graph = builder.GetGraph();
  graph->SetParentNode(func);

  AttrUtils::SetStr(input1->GetOpDesc(), ATTR_NAME_NEXT_ITERATION, "data1");
  std::map<std::string, std::list<NodeIndexIO>> symbol_to_anchors;
  std::map<std::string, std::string> anchor_to_symbol;
  int ret = GraphUtils::HandleMergeInput(input1, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, HandleMergeInputPeerOutAnchorIsNotNull) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &var1 = builder.AddNode("var1", VARIABLEV2, 1, 1);
  const auto &func = builder.AddNode("func", PARTITIONEDCALL, 4, 1);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(input1, 0, func, 0);
  builder.AddDataEdge(var1, 0, func, 1);
  builder.AddDataEdge(func, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  std::map<std::string, std::list<NodeIndexIO>> symbol_to_anchors;
  NodeIndexIO node_index_io(func, 0, kOut);
  std::list<NodeIndexIO> symbol_list;
  symbol_list.push_back(node_index_io);
  symbol_to_anchors.insert(pair<std::string, std::list<NodeIndexIO>>("var1_out_0", symbol_list));

  std::map<std::string, std::string> anchor_to_symbol;
  anchor_to_symbol.insert(pair<std::string, std::string>("data1_out_0", "var1_out_0"));
  int ret = GraphUtils::HandleMergeInput(func, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, HandleSubgraphOutput) {
  auto builder = ut::GraphBuilder("test2");
  const auto &input1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &var1 = builder.AddNode("var1", VARIABLEV2, 1, 1);
  const auto &func = builder.AddNode("func", PARTITIONEDCALL, 4, 1);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(input1, 0, func, 0);
  builder.AddDataEdge(var1, 0, func, 1);
  builder.AddDataEdge(func, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  graph->SetParentNode(func);
  AttrUtils::SetInt(input1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);

  std::map<std::string, std::list<NodeIndexIO>> symbol_to_anchors;
  NodeIndexIO node_index_io(func, 0, kOut);
  std::list<NodeIndexIO> symbol_list;
  symbol_list.push_back(node_index_io);
  symbol_to_anchors.insert(pair<std::string, std::list<NodeIndexIO>>("var1_out_0", symbol_list));

  std::map<std::string, std::string> anchor_to_symbol;
  anchor_to_symbol.insert(pair<std::string, std::string>("data1_out_0", "var1_out_0"));
  int ret = GraphUtils::HandleSubgraphOutput(func, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST_F(UtestGraphUtils, UnionSymbolMappingSuccess) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &var1 = builder.AddNode("var1", VARIABLEV2, 1, 1);
  const auto &input2 = builder.AddNode("data2", DATA, 1, 1);
  const auto &var2 = builder.AddNode("var2", VARIABLEV2, 1, 1);
  const auto &func = builder.AddNode("func", PARTITIONEDCALL, 4, 1);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(input1, 0, func, 0);
  builder.AddDataEdge(var1, 0, func, 1);
  builder.AddDataEdge(input2, 0, func, 2);
  builder.AddDataEdge(var2, 0, func, 3);
  builder.AddDataEdge(func, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  graph->SetParentNode(func);
  AttrUtils::SetInt(input1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(input2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);

  std::map<std::string, std::list<NodeIndexIO>> symbol_to_anchors;
  NodeIndexIO node_index1(input1, 0, kOut);
  NodeIndexIO node_index2(input2, 0, kOut);
  std::list<NodeIndexIO> symbol_list;
  symbol_list.push_back(node_index1);
  symbol_list.push_back(node_index2);
  symbol_to_anchors.insert(pair<std::string, std::list<NodeIndexIO>>("var1_out_0", symbol_list));
  symbol_to_anchors.insert(pair<std::string, std::list<NodeIndexIO>>("var2_out_0", symbol_list));

  std::map<std::string, std::string> anchor_to_symbol;
  anchor_to_symbol.insert(pair<std::string, std::string>("data1_out_0", "var1_out_0"));
  anchor_to_symbol.insert(pair<std::string, std::string>("data2_out_0", "var2_out_0"));

  std::string symbol;
  int ret = GraphUtils::UnionSymbolMapping(node_index1, node_index2, symbol_to_anchors, anchor_to_symbol, symbol);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, UpdateRefMappingSuccess) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &var1 = builder.AddNode("var1", VARIABLEV2, 1, 1);
  const auto &input2 = builder.AddNode("data2", DATA, 1, 1);
  const auto &var2 = builder.AddNode("var2", VARIABLEV2, 1, 1);
  const auto &func = builder.AddNode("func", PARTITIONEDCALL, 4, 1);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(input1, 0, func, 0);
  builder.AddDataEdge(var1, 0, func, 1);
  builder.AddDataEdge(input2, 0, func, 2);
  builder.AddDataEdge(var2, 0, func, 3);
  builder.AddDataEdge(func, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  graph->SetParentNode(func);
  AttrUtils::SetInt(input1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(input2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);

  std::map<std::string, std::list<NodeIndexIO>> symbol_to_anchors;
  NodeIndexIO cur_node_info(input1, 0, kOut);
  NodeIndexIO exist_node_info(input2, 0, kOut);
  std::list<NodeIndexIO> symbol_list;
  symbol_list.push_back(cur_node_info);
  symbol_list.push_back(exist_node_info);
  symbol_to_anchors.insert(pair<std::string, std::list<NodeIndexIO>>("var1_out_0", symbol_list));
  symbol_to_anchors.insert(pair<std::string, std::list<NodeIndexIO>>("var2_out_0", symbol_list));

  std::map<std::string, std::string> anchor_to_symbol;
  anchor_to_symbol.insert(pair<std::string, std::string>("data1_out_0", "var1_out_0"));
  anchor_to_symbol.insert(pair<std::string, std::string>("data2_out_0", "var2_out_0"));

  std::string symbol;
  int ret = GraphUtils::UpdateRefMapping(cur_node_info, exist_node_info, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, UpdateRefMappingSymbolToAnchorsIsNull) {
  auto builder = ut::GraphBuilder("test1");
  const auto &input1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &var1 = builder.AddNode("var1", VARIABLEV2, 1, 1);
  const auto &input2 = builder.AddNode("data2", DATA, 1, 1);
  const auto &var2 = builder.AddNode("var2", VARIABLEV2, 1, 1);
  const auto &func = builder.AddNode("func", PARTITIONEDCALL, 4, 1);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(input1, 0, func, 0);
  builder.AddDataEdge(var1, 0, func, 1);
  builder.AddDataEdge(input2, 0, func, 2);
  builder.AddDataEdge(var2, 0, func, 3);
  builder.AddDataEdge(func, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  graph->SetParentNode(func);
  AttrUtils::SetInt(input1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(input2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);

  NodeIndexIO cur_node_info(input1, 0, kOut);
  NodeIndexIO exist_node_info(input2, 0, kOut);
  std::map<std::string, std::list<NodeIndexIO>> symbol_to_anchors;
  std::map<std::string, std::string> anchor_to_symbol;
  anchor_to_symbol.insert(pair<std::string, std::string>("data1_out_0", "var1_out_0"));
  anchor_to_symbol.insert(pair<std::string, std::string>("data2_out_0", "var2_out_0"));

  std::string symbol;
  int ret = GraphUtils::UpdateRefMapping(cur_node_info, exist_node_info, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, IsRefFromInputOutDataAnchorPtrIsNull) {
  OutDataAnchorPtr out_data_anchor;
  int32_t reuse_in_index;
  bool ret = GraphUtils::IsRefFromInput(out_data_anchor, reuse_in_index);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestGraphUtils, IsRefFromInputFail) {
  auto builder = ut::GraphBuilder("test0");
  const auto &node0 = builder.AddNode("node0", "node", 1, 1);
  int32_t reuse_in_index;
  bool ret = GraphUtils::IsRefFromInput(node0->GetOutDataAnchor(0), reuse_in_index);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestGraphUtils, IsRefFromInputPassThroughOK) {
  auto builder = ut::GraphBuilder("test0");
  const auto &node0 = builder.AddNode("node0", NETOUTPUT, 1, 1);
  int32_t reuse_in_index;
  bool ret = GraphUtils::IsRefFromInput(node0->GetOutDataAnchor(0), reuse_in_index);
  EXPECT_EQ(ret, true);
}

TEST_F(UtestGraphUtils, IsRefFromInputTypeIsMergeSuccess) {
  auto builder = ut::GraphBuilder("test0");
  const auto &node0 = builder.AddNode("node0", MERGE, 1, 1);
  int32_t reuse_in_index;
  bool ret = GraphUtils::IsRefFromInput(node0->GetOutDataAnchor(0), reuse_in_index);
  EXPECT_EQ(ret, true);
}

TEST_F(UtestGraphUtils, IsRefFromInputRefOpFail) {
  auto builder = ut::GraphBuilder("test0");
  const auto &node1 = builder.AddNode("node", "node", 1, 1);
  AttrUtils::SetBool(node1->GetOpDesc(), ATTR_NAME_REFERENCE, true);

  int32_t reuse_in_index;
  bool ret = GraphUtils::IsRefFromInput(node1->GetOutDataAnchor(0), reuse_in_index);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestGraphUtils, IsNoPaddingRefFromInputSuccess) {
  auto builder = ut::GraphBuilder("test0");
  const auto &node1 = builder.AddNode("node", "node", 1, 1);
  AttrUtils::SetBool(node1->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  AttrUtils::SetBool(node1->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
  AttrUtils::SetBool(node1->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);

  int32_t reuse_in_index;
  bool ret = GraphUtils::IsNoPaddingRefFromInput(node1->GetOutDataAnchor(0), reuse_in_index);
  EXPECT_EQ(ret, true);
}

TEST_F(UtestGraphUtils, IsNodeInGraphRecursivelySuccess) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test0");
  Node node;
  node.SetOwnerComputeGraph(graph);
 
  bool ret = GraphUtils::IsNodeInGraphRecursively(graph, node);
  EXPECT_EQ(ret, true);
}

TEST_F(UtestGraphUtils, IsNodeInGraphRecursivelyFail) {
  auto builder = ut::GraphBuilder("test0");
  Node node;
  node.SetOwnerComputeGraph(builder.GetGraph());
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
  bool ret = GraphUtils::IsNodeInGraphRecursively(graph, node);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestGraphUtils, IsUnknownShapeGraphFail) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
  bool ret = GraphUtils::IsUnknownShapeGraph(graph);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestGraphUtils, IsUnknownShapeGraphGraphIsNull) {
  ComputeGraphPtr graph;
  bool ret = GraphUtils::IsUnknownShapeGraph(graph);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestGraphUtils, IsUnknownShapeGraphSuccess) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("add", "Add", 2, 1, FORMAT_NHWC, DT_FLOAT, {16, 228, 228, 3});
  auto graph = builder.GetGraph();

  auto add_node = graph->FindNode("add");
  auto out_desc = add_node->GetOpDesc()->MutableOutputDesc(0);
  out_desc->SetShape(GeShape({-1, 228, 228, 3}));

  bool ret = GraphUtils::IsUnknownShapeGraph(graph);
  EXPECT_EQ(ret, true);
}

TEST_F(UtestGraphUtils, UnfoldSubgraphSuccess) {
  ut::GraphBuilder builder = ut::GraphBuilder("test0");
  auto graph = builder.GetGraph();
  std::function<bool(const ComputeGraphPtr &)> filter;
  int ret = GraphUtils::UnfoldSubgraph(graph, filter);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, MergeInputNodesFail) {
  auto builder = ut::GraphBuilder("test0");
  const auto &node1 = builder.AddNode("node", DATA, 1, 1);
  auto graph = builder.GetGraph();
  graph->SetParentNode(node1);
  
  int ret = GraphUtils::MergeInputNodes(graph, node1);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, MergeNetOutputNodeSuccess) {
  auto builder = ut::GraphBuilder("test2");
  const auto &node1 = builder.AddNode("node", DATA, 1, 1);
  auto graph = builder.GetGraph();
  graph->SetParentNode(node1);
  
  int ret = GraphUtils::MergeNetOutputNode(graph, node1);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphUtils, RemoveJustNodeGraphImplIsNull) {
  ComputeGraph compute_graph("");
  compute_graph.impl_ = nullptr;
  auto graph_builder0 = ut::GraphBuilder("Test0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  int ret = GraphUtils::RemoveJustNode(compute_graph, node0);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, GetNodeFail) {
  UtestComputeGraphBuilder graph;
  NodePtr node_ptr = graph.GetNode("node1");
  EXPECT_EQ(node_ptr, nullptr);
}

TEST_F(UtestGraphUtils, GetAllNodeNodeSizeIs0) {
  UtestComputeGraphBuilder graph;
  std::vector<NodePtr> node_ptr = graph.GetAllNodes();
  EXPECT_EQ(node_ptr.size(), 0);
}

TEST_F(UtestGraphUtils, BuildExistNodesTest) {
  PartialGraphBuilder builder;
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";
  builder.BuildExistNodes(err, msg);
  EXPECT_EQ(err, GRAPH_SUCCESS);
  EXPECT_EQ(msg, "");

  builder.exist_nodes_.push_back(nullptr);
  builder.BuildExistNodes(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_NE(msg, "");

  builder.exist_nodes_.clear();
  auto gbuilder = ut::GraphBuilder("test2");
  auto node = gbuilder.AddNode("node", DATA, 1, 1);
  auto opdsc = std::make_shared<OpDesc>("node1", "node");
  builder.AddExistNode(node);
  builder.AddNode(opdsc);
  EXPECT_EQ(builder.exist_nodes_.size(), 1);
  builder.BuildExistNodes(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_NE(msg, "");

  err = GRAPH_SUCCESS;
  msg = "";
  builder.owner_graph_ = node->GetOwnerComputeGraph();
  builder.BuildExistNodes(err, msg);
  EXPECT_EQ(err, GRAPH_SUCCESS);
  EXPECT_EQ(msg, "");
}

TEST_F(UtestGraphUtils, PartialGraphBuilderBuildTest) {
  PartialGraphBuilder par_graph_builder;
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";
  ComputeGraphPtr computer_graph;
  computer_graph = par_graph_builder.Build(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_EQ(msg, "graph is NULL.");
  EXPECT_EQ(computer_graph, nullptr);

  auto builder = ut::GraphBuilder("test1");
  auto node = builder.AddNode("node", DATA, 1, 1);
  par_graph_builder.SetOwnerGraph(node->GetOwnerComputeGraph());
  computer_graph = par_graph_builder.Build(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_EQ(msg, "graph is NULL.");
  EXPECT_EQ(computer_graph, nullptr);
}

TEST_F(UtestGraphUtils, CompleteGraphBuilderBuilder) {
  CompleteGraphBuilder complete_builder("");
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";

  complete_builder.Build(err, msg);
  EXPECT_EQ(err, GRAPH_SUCCESS);
  EXPECT_EQ(msg, "");
}

TEST_F(UtestGraphUtils, CompleteGraphBuilderBuildGraphTargets) {
  CompleteGraphBuilder complete_builder("test1");
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";

  //node_names_ is null
  complete_builder.AddTarget("Data_1");
  complete_builder.BuildGraphTargets(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_NE(msg, "");
}

TEST_F(UtestGraphUtils, BuildNetOutputNodeWithLinkTest) {
  CompleteGraphBuilder complete_builder("test1");
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";
  auto builder = ut::GraphBuilder("test2");
  auto node = builder.AddNode("node", DATA, 1, 1);
  auto node2 = builder.AddNode("node2", NETOUTPUT, 1, 0);
  complete_builder.owner_graph_ = node->GetOwnerComputeGraph();

  OpDescPtr net_output_desc;
  std::vector<OutDataAnchorPtr> peer_out_anchors;
  complete_builder.BuildNetOutputNodeWithLink(net_output_desc, peer_out_anchors, err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_NE(msg, "");

  err = GRAPH_SUCCESS;
  msg = "";
  net_output_desc = std::make_shared<OpDesc>("test", "test");
  complete_builder.AddTarget("Data_1");
  complete_builder.BuildNetOutputNodeWithLink(net_output_desc, peer_out_anchors, err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_NE(msg, "");

  err = GRAPH_SUCCESS;
  msg = "";
  uint32_t index = 1;
  complete_builder.input_mapping_.insert(pair<uint32_t, uint32_t>(1, 0));
  auto ret_node = complete_builder.AddDataNode(index, err, msg);
  EXPECT_EQ(ret_node, complete_builder.node_names_["Data_1"]);
  complete_builder.BuildNetOutputNodeWithLink(net_output_desc, peer_out_anchors, err, msg);
  EXPECT_EQ(err, GRAPH_SUCCESS);
  EXPECT_EQ(msg, "");
}

TEST_F(UtestGraphUtils, AddDataNodeTest) {
  CompleteGraphBuilder complete_builder("test1");
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";

  auto builder = ut::GraphBuilder("test2");
  auto node = builder.AddNode("node", DATA, 1, 1);

  uint32_t index = 1;
  complete_builder.input_mapping_.insert(pair<uint32_t, uint32_t>(1, 1));
  complete_builder.owner_graph_ = node->GetOwnerComputeGraph();

  auto ret_node = complete_builder.AddDataNode(index, err, msg);
  EXPECT_EQ(err, GRAPH_SUCCESS);
  EXPECT_EQ(msg, "");
  EXPECT_EQ(ret_node, complete_builder.node_names_["Data_1"]);
}

TEST_F(UtestGraphUtils, AddNetOutputNodeTest) {
  CompleteGraphBuilder complete_builder("test1");
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";

  // graph_outputs_ and graph_targets_ is null
  complete_builder.AddNetOutputNode(err, msg);
  EXPECT_EQ(err, GRAPH_SUCCESS);
  EXPECT_EQ(msg, "");

  // node_names_ is null
  complete_builder.AddTarget("Data_1");
  complete_builder.graph_outputs_.push_back(pair<std::string, uint32_t>("Data_1", 0));
  complete_builder.AddNetOutputNode(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_NE(msg, "");

  // node is nullptr
  err = GRAPH_SUCCESS;
  msg = "";
  complete_builder.node_names_.insert(pair<std::string, NodePtr>("Data_1", nullptr));
  complete_builder.AddNetOutputNode(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_EQ(msg, "AddNetOutputNode failed: node is NULL.");
}

TEST_F(UtestGraphUtils, AddRetValNodesTest) {
  CompleteGraphBuilder complete_builder("test1");
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";

  //node_names_ is null
  complete_builder.graph_outputs_.push_back(pair<std::string, uint32_t>("Data_1", 0));
  complete_builder.AddRetValNodes(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_EQ(msg, "AddRetValNode failed: node Data_1 not exist in graph.");
 
  //node_names_ node is nullptr
  err = GRAPH_SUCCESS;
  msg = "";
  complete_builder.node_names_.insert(pair<std::string, NodePtr>("Data_1", nullptr));
  complete_builder.AddRetValNodes(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_EQ(msg, "AddRetValNode failed: node is NULL.");

  //node_names_ node is not nullptr
  auto builder = ut::GraphBuilder("test2");
  auto node = builder.AddNode("node", DATA, 1, 0);
  complete_builder.owner_graph_ = node->GetOwnerComputeGraph();
  
  complete_builder.node_names_.clear();
  complete_builder.node_names_.insert(pair<std::string, NodePtr>("Data_1", node));
  complete_builder.output_mapping_.insert(pair<uint32_t, uint32_t>(0, 0));
  err = GRAPH_SUCCESS;
  msg = "";
  complete_builder.AddRetValNodes(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_NE(msg, "");
}

TEST_F(UtestGraphUtils, BuildCtrlLinksTest) {
  PartialGraphBuilder par_builder;
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";

  auto builder = ut::GraphBuilder("test1");
  auto node = builder.AddNode("node_input", DATA, 1, 1);
  auto node2 = builder.AddNode("node_output", NETOUTPUT, 1, 1);
  par_builder.SetOwnerGraph(node->GetOwnerComputeGraph());

  par_builder.AddControlLink("node_input", "node_output");
  ComputeGraphPtr graph;
  graph = par_builder.Build(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_NE(msg, "");
  EXPECT_EQ(graph, nullptr);

  par_builder.node_names_.insert(pair<std::string, NodePtr>("node_input", nullptr));
  par_builder.node_names_.insert(pair<std::string, NodePtr>("node_output", nullptr));
  err = GRAPH_SUCCESS;
  msg = "";
  graph = par_builder.Build(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_NE(msg, "");
  EXPECT_EQ(graph, nullptr);

  par_builder.node_names_.clear();
  par_builder.node_names_.insert(pair<std::string, NodePtr>("node_input", node));
  par_builder.node_names_.insert(pair<std::string, NodePtr>("node_output", node2));
  err = GRAPH_SUCCESS;
  msg = "";
  graph = par_builder.Build(err, msg);
  EXPECT_EQ(err, GRAPH_SUCCESS);
  EXPECT_EQ(msg, "");
  EXPECT_EQ(graph, node->GetOwnerComputeGraph());
}

TEST_F(UtestGraphUtils, BuildDataLinksTest) {
  PartialGraphBuilder par_builder;
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";

  auto builder = ut::GraphBuilder("test1");
  auto node = builder.AddNode("node_input", DATA, 1, 1);
  auto node2 = builder.AddNode("node_output", NETOUTPUT, 1, 1);
  par_builder.SetOwnerGraph(node->GetOwnerComputeGraph());

  par_builder.AddDataLink("node_input", 1, "node_output", 1);
  ComputeGraphPtr graph;
  graph = par_builder.Build(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_NE(msg, "");
  EXPECT_EQ(graph, nullptr);

  par_builder.node_names_.insert(pair<std::string, NodePtr>("node_input", nullptr));
  par_builder.node_names_.insert(pair<std::string, NodePtr>("node_output", nullptr));
  err = GRAPH_SUCCESS;
  msg = "";
  graph = par_builder.Build(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_NE(msg, "");
  EXPECT_EQ(graph, nullptr);
}

TEST_F(UtestGraphUtils, PostProcessTest) {
  CompleteGraphBuilder complete_builder("test1");
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";

  auto builder = ut::GraphBuilder("test2");
  auto node1 = builder.AddNode("node1", DATA, 1, 1);
  auto owner_graph = node1->GetOwnerComputeGraph();
  complete_builder.owner_graph_ = owner_graph;
  
  auto builder2 = ut::GraphBuilder("test3");
  auto node2 = builder2.AddNode("node", "node", 1, 1);
  complete_builder.parent_node_ = node2;
  auto parent_graph = complete_builder.parent_node_->GetOwnerComputeGraph();
  
  std::string graph_id;
  AttrUtils::SetStr(parent_graph, ATTR_NAME_SESSION_GRAPH_ID, graph_id);

  AnyValue any_value;
  any_value.SetValue(1);
  complete_builder.parent_node_->GetOwnerComputeGraph()->SetAttr(ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, any_value);
  AttrUtils::SetBool(node1->GetOpDesc(), ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);
  
  complete_builder.PostProcess(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_EQ(msg, "Copy attr _dynamic_shape_partitioned failed.");
}


TEST_F(UtestGraphUtils, GetRefMappingTest) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test0");
  auto op_desc = std::make_shared<OpDesc>("node1", "node1");
  graph->AddNode(op_desc);
  std::map<std::string, std::list<NodeIndexIO>> symbol_to_anchors;
  std::map<std::string, std::string> anchor_to_symbol;
  int ret = GraphUtils::GetRefMapping(graph, symbol_to_anchors, anchor_to_symbol);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, ComputeGraphBuilderBuildNodesTest) {
  UtestComputeGraphBuilder utest_graph_builder;
  graphStatus err = GRAPH_SUCCESS;
  std::string msg = "";

  //owner_graph_ is null
  utest_graph_builder.BuildNodes(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_EQ(msg, "graph is NULL.");

  //nodes_ is null
  auto builder = ut::GraphBuilder("test1");
  auto node1 = builder.AddNode("node1", DATA, 1, 1);
  auto owner_graph = node1->GetOwnerComputeGraph();
  utest_graph_builder.owner_graph_ = owner_graph;
  err = GRAPH_SUCCESS;
  msg = "";
  utest_graph_builder.nodes_.push_back(nullptr);
  utest_graph_builder.BuildNodes(err, msg);
  EXPECT_EQ(err, GRAPH_FAILED);
  EXPECT_EQ(msg, "op_desc is NULL.");
}

TEST_F(UtestGraphUtils, DumpGEGraph) {
  auto ge_tensor = std::make_shared<GeTensor>();
  uint8_t data_buf[4096] = {0};
  data_buf[0] = 7;
  data_buf[10] = 8;
  ge_tensor->SetData(data_buf, 4096);

  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 0, 1);
  auto const_node = builder.AddNode("Const", "Const", 0, 1);
  AttrUtils::SetTensor(const_node->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, ge_tensor);
  auto add_node = builder.AddNode("Add", "Add", 2, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data_node, 0, add_node, 0);
  builder.AddDataEdge(const_node, 0, add_node, 0);
  builder.AddDataEdge(add_node, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  // test existed dir
  GraphUtils::DumpGEGraph(graph, "", true, "./ge_test_graph_0001.txt");
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("GeTestGraph1");
  bool state = GraphUtils::LoadGEGraph("./ge_test_graph_0001.txt", *com_graph1);
  ASSERT_EQ(state, true);
  ASSERT_EQ(com_graph1->GetAllNodesSize(), 4);

  // test not existed dir
  GraphUtils::DumpGEGraph(graph, "", true, "./test/ge_test_graph_0002.txt");
  ComputeGraphPtr com_graph2 = std::make_shared<ComputeGraph>("GeTestGraph2");
  state = GraphUtils::LoadGEGraph("./test/ge_test_graph_0002.txt", *com_graph2);
  ASSERT_EQ(state, true);

  // test input para user_graph_name, without path
  GraphUtils::DumpGEGraph(graph, "", true, "ge_test_graph_0003.txt");
  ComputeGraphPtr com_graph3 = std::make_shared<ComputeGraph>("GeTestGraph3");
  state = GraphUtils::LoadGEGraph("./ge_test_graph_0003.txt", *com_graph3);
  ASSERT_EQ(state, true);
}

TEST_F(UtestGraphUtils, FindNodeByTypeFromAllGraphs) {
  auto graph = BuildGraphWithSubGraph();
  ASSERT_NE(graph, nullptr);
  auto nodes = GraphUtils::FindNodesByTypeFromAllNodes(graph, "Data");
  EXPECT_EQ(nodes.size(), 2);
}

TEST_F(UtestGraphUtils, RemoveNodesByTypeWithoutRelinkPlaceholder) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_placeholder");
  BuildGraphWithPlaceholderAndEnd(graph);
  ASSERT_NE(graph, nullptr);
  auto ret = GraphUtils::RemoveNodesByTypeWithoutRelink(graph, "PlaceHolder");
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  auto nodes = GraphUtils::FindNodesByTypeFromAllNodes(graph, "PlaceHolder");
  EXPECT_EQ(nodes.size(), 0);
}

TEST_F(UtestGraphUtils, RemoveNodesByTypeWithoutRelinkEnd) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_end"); 
  BuildGraphWithPlaceholderAndEnd(graph);
  ASSERT_NE(graph, nullptr);
  auto ret = GraphUtils::RemoveNodesByTypeWithoutRelink(graph, "End");
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  auto nodes = GraphUtils::FindNodesByTypeFromAllNodes(graph, "End");
  EXPECT_EQ(nodes.size(), 0);
}

TEST_F(UtestGraphUtils, RemoveNodesByTypeWithoutRelinkAdd) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_end"); 
  BuildGraphWithPlaceholderAndEnd(graph);
  ASSERT_NE(graph, nullptr);
  auto ret = GraphUtils::RemoveNodesByTypeWithoutRelink(graph, "Add");
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  auto nodes = GraphUtils::FindNodesByTypeFromAllNodes(graph, "Add");
  EXPECT_EQ(nodes.size(), 0);
}

TEST_F(UtestGraphUtils, RemoveNodesByTypeWithoutRelinkData) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_end"); 
  BuildGraphWithPlaceholderAndEnd(graph);
  ASSERT_NE(graph, nullptr);
  auto ret = GraphUtils::RemoveNodesByTypeWithoutRelink(graph, DATA);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  auto nodes = GraphUtils::FindNodesByTypeFromAllNodes(graph, DATA);
  EXPECT_EQ(nodes.size(), 0);
}

TEST_F(UtestGraphUtils, FindNodeByTypeFromAllGraphsNullInput) {
  ComputeGraphPtr graph = nullptr;
  auto nodes = GraphUtils::FindNodesByTypeFromAllNodes(graph, "Data");
  EXPECT_EQ(nodes.size(), 0);
}
}  // namespace ge

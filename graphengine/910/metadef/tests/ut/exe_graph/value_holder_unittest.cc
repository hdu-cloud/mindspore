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
#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/runtime/context_extend.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <numeric>
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "checker/bg_test.h"
#include "graph/utils/graph_utils.h"
#include "checker/topo_checker.h"
#include "checker/summary_checker.h"

namespace gert {
namespace bg {
namespace {
ge::NodePtr FakeNode() {
  static size_t counter = 0;
  static ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto op_desc = std::make_shared<ge::OpDesc>("FakeNode_" + std::to_string(counter++), "FakeNode");
  return graph->AddNode(op_desc);
}
size_t GetComputeNodeIndex(const ge::Node *node) {
  int64_t index;
  if (!ge::AttrUtils::GetInt(node->GetOpDesc(), kComputeNodeIndex, index)) {
    return std::numeric_limits<size_t>::max();
  }
  return static_cast<size_t>(index);
}
}
class ValueHolderUt : public BgTest {
 public:
  ge::ComputeGraphPtr FindFirstSubgraphForNodeType(const ge::ComputeGraphPtr &root_graph,
                                                   const std::string &node_type) {
    for (const auto &subgraph : root_graph->GetAllSubgraphs()) {
      auto parent_node = subgraph->GetParentNode();
      if (parent_node->GetType() == node_type) {
        return subgraph;
      }
    }
    return nullptr;
  }
  ge::NodePtr FindData(const ge::ComputeGraphPtr &graph, int32_t index) {
    for (const auto &node : graph->GetDirectNode()) {
      if (node->GetType() != "Data") {
        continue;
      }
      int32_t data_index;
      if (!ge::AttrUtils::GetInt(node->GetOpDesc(), "index", data_index)) {
        continue;
      }
      if (data_index != index) {
        continue;
      }
      return node;
    }
    return nullptr;
  }

  void ConnectFromInnerData(const ge::NodePtr &node, const std::vector<int32_t> &indexes) {
    ASSERT_EQ(node->GetInDataNodes().size(), indexes.size());
    size_t i = 0;
    for (const auto &in_node : node->GetInDataNodes()) {
      ASSERT_EQ(in_node->GetType(), "InnerData");
      int32_t index;
      ASSERT_TRUE(ge::AttrUtils::GetInt(in_node->GetOpDesc(), "index", index));
      ASSERT_EQ(index, indexes[i++]);
    }
  }
  void ConnectFromOuter(ge::NodePtr node, int32_t dst_index, const ge::NodePtr &outer, int32_t src_index) {
    while (true) {
      auto dst_anchor = node->GetInDataAnchor(dst_index);
      ASSERT_NE(dst_anchor, nullptr);
      auto src_anchor = dst_anchor->GetPeerOutAnchor();
      ASSERT_NE(src_anchor, nullptr);
      auto src_node = src_anchor->GetOwnerNode();
      ASSERT_NE(src_node, nullptr);
      if (src_node == outer) {
        return;
      }
      if (src_node->GetType() == "InnerData" || src_node->GetType() == "Data") {
        int32_t parent_index;
        ASSERT_TRUE(ge::AttrUtils::GetInt(src_node->GetOpDesc(), "index", parent_index));
        auto parent_graph = src_node->GetOwnerComputeGraph();
        ASSERT_NE(parent_graph, nullptr);
        auto parent_node = parent_graph->GetParentNode();
        ASSERT_NE(parent_node, nullptr);
        node = parent_node;
        dst_index = parent_index;
      }
    }
  }
  void StrictSubgraphs(const ge::NodePtr &node, const std::vector<std::string> &names) {
    const auto &subgraph_names_to_index = node->GetOpDesc()->GetSubgraphNameIndexes();

    ASSERT_EQ(subgraph_names_to_index.size(), names.size());
    for (const auto &name : names) {
      auto iter = subgraph_names_to_index.find(name);
      ASSERT_NE(iter, subgraph_names_to_index.end());
      auto ins_name = node->GetOpDesc()->GetSubgraphInstanceName(iter->second);
      auto root_graph = ge::GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());
      ASSERT_NE(root_graph->GetSubgraph(ins_name), nullptr);
    }
  }
};
TEST_F(ValueHolderUt, CreateConstOk) {
  ge::Format f1 = ge::FORMAT_NC1HWC0;
  auto c = ValueHolder::CreateConst(reinterpret_cast<const uint8_t *>(&f1), sizeof(f1));
  EXPECT_NE(c, nullptr);
  ASSERT_TRUE(c->IsOk());
  ASSERT_NE(c->GetNode(), nullptr);
  EXPECT_EQ(c->GetType(), ValueHolder::ValueHolderType::kConst);
  EXPECT_EQ(c->GetOutIndex(), 0);
  auto node = c->GetNode();
  EXPECT_EQ(node->GetType(), "Const");
  EXPECT_EQ(node->GetAllOutDataAnchorsSize(), 1);
  EXPECT_EQ(node->GetAllInDataAnchorsSize(), 0);
  ge::Buffer buffer;
  ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(node->GetOpDesc(), "value", buffer));
  ASSERT_EQ(buffer.GetSize(), sizeof(ge::FORMAT_NC1HWC0));
  EXPECT_EQ(*reinterpret_cast<ge::Format *>(buffer.GetData()), ge::FORMAT_NC1HWC0);
}
TEST_F(ValueHolderUt, CreateVectorConstOk) {
  int64_t shape[5] = {32, 1, 224, 224, 16};
  auto c = ValueHolder::CreateConst(reinterpret_cast<const uint8_t *>(shape), sizeof(shape));
  EXPECT_NE(c, nullptr);
  ASSERT_TRUE(c->IsOk());
  ASSERT_NE(c->GetNode(), nullptr);
  EXPECT_EQ(c->GetType(), ValueHolder::ValueHolderType::kConst);
  EXPECT_EQ(c->GetOutIndex(), 0);
  auto node = c->GetNode();
  EXPECT_EQ(node->GetType(), "Const");
  ge::Buffer buffer;
  ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(node->GetOpDesc(), "value", buffer));
  ASSERT_EQ(buffer.GetSize(), sizeof(shape));
  EXPECT_EQ(memcmp(buffer.GetData(), shape, sizeof(shape)), 0);
}
TEST_F(ValueHolderUt, CreateFeedOk) {
  auto c = ValueHolder::CreateFeed(1);
  EXPECT_NE(c, nullptr);
  ASSERT_TRUE(c->IsOk());
  ASSERT_NE(c->GetNode(), nullptr);
  EXPECT_EQ(c->GetType(), ValueHolder::ValueHolderType::kFeed);
  EXPECT_EQ(c->GetOutIndex(), 0);
  auto node = c->GetNode();
  EXPECT_EQ(node->GetType(), "Data");
  EXPECT_EQ(node->GetAllOutDataAnchorsSize(), 1);
  EXPECT_EQ(node->GetAllInDataAnchorsSize(), 0);
  int32_t index;
  ASSERT_TRUE(ge::AttrUtils::GetInt(node->GetOpDesc(), "index", index));
  EXPECT_EQ(index, 1);
}
TEST_F(ValueHolderUt, CreateErrorOk) {
  auto holder = ValueHolder::CreateError("This is a test error information, int %d, %s", 10240, "Test msg");
  ASSERT_NE(holder, nullptr);
  EXPECT_FALSE(holder->IsOk());
}
TEST_F(ValueHolderUt, CreateDataOutOk) {
  ge::Format f1 = ge::FORMAT_NC1HWC0;
  auto const1 = ValueHolder::CreateConst(reinterpret_cast<const uint8_t *>(&f1), sizeof(f1));
  auto data1 = ValueHolder::CreateFeed(0);
  ASSERT_NE(const1, nullptr);
  ASSERT_NE(data1, nullptr);

  std::vector<ValueHolderPtr> inputs = {data1, const1};
  auto holders = ValueHolder::CreateDataOutput("TestNode", inputs, 3);

  ASSERT_EQ(holders.size(), 3);
  ASSERT_TRUE(holders[0]->IsOk());
  ASSERT_TRUE(holders[1]->IsOk());
  ASSERT_TRUE(holders[2]->IsOk());
  EXPECT_EQ(holders[0]->GetType(), ValueHolder::ValueHolderType::kOutput);
  EXPECT_EQ(holders[1]->GetType(), ValueHolder::ValueHolderType::kOutput);
  EXPECT_EQ(holders[2]->GetType(), ValueHolder::ValueHolderType::kOutput);

  ASSERT_NE(const1->GetGraph(), nullptr);
  ASSERT_NE(const1->GetNode()->GetOwnerComputeGraph(), nullptr);
  ASSERT_NE(data1->GetGraph(), nullptr);
  ASSERT_NE(data1->GetNode()->GetOwnerComputeGraph(), nullptr);
  ASSERT_NE(holders[0]->GetNode(), nullptr);
  ASSERT_NE(holders[0]->GetGraph(), nullptr);
  ASSERT_NE(holders[0]->GetNode()->GetOwnerComputeGraph(), nullptr);
  ASSERT_NE(holders[1]->GetNode(), nullptr);
  ASSERT_NE(holders[1]->GetGraph(), nullptr);
  ASSERT_NE(holders[1]->GetNode()->GetOwnerComputeGraph(), nullptr);
  ASSERT_NE(holders[2]->GetNode(), nullptr);
  ASSERT_NE(holders[2]->GetGraph(), nullptr);
  ASSERT_NE(holders[2]->GetNode()->GetOwnerComputeGraph(), nullptr);

  // check node is ok
  auto node = holders[0]->GetNode();
  ASSERT_EQ(node->GetType(), "TestNode");
  ASSERT_EQ(node->GetAllInDataAnchorsSize(), 2);
  ASSERT_EQ(node->GetAllOutDataAnchorsSize(), 3);

  // all holders point to the same node
  ASSERT_EQ(holders[0]->GetNode(), holders[1]->GetNode());
  ASSERT_EQ(holders[0]->GetNode(), holders[2]->GetNode());

  // all holders have correct out-index
  EXPECT_EQ(holders[0]->GetOutIndex(), 0);
  EXPECT_EQ(holders[1]->GetOutIndex(), 1);
  EXPECT_EQ(holders[2]->GetOutIndex(), 2);

  // all nodes(contains data and const) in the same graph
  EXPECT_EQ(holders[0]->GetNode()->GetOwnerComputeGraph(), const1->GetNode()->GetOwnerComputeGraph());
  EXPECT_EQ(holders[0]->GetNode()->GetOwnerComputeGraph(), data1->GetNode()->GetOwnerComputeGraph());

  // all holders holds the same graph
  EXPECT_EQ(holders[0]->GetGraph(), holders[1]->GetGraph());
  EXPECT_EQ(holders[0]->GetGraph(), holders[2]->GetGraph());
  EXPECT_EQ(holders[0]->GetGraph(), const1->GetGraph());
  EXPECT_EQ(holders[0]->GetGraph(), data1->GetGraph());

  // check graph is ok
  auto graph = holders[0]->GetGraph();
  ASSERT_EQ(graph->GetAllNodesSize(), 3);
  CheckGraphGenerally(*graph);

  auto const1_g = graph->FindFirstNodeMatchType("Const");
  auto data1_g = graph->FindFirstNodeMatchType("Data");
  auto node_g = graph->FindFirstNodeMatchType("TestNode");
  ASSERT_NE(const1_g, nullptr);
  ASSERT_NE(data1_g, nullptr);
  ASSERT_NE(node_g, nullptr);

  EXPECT_EQ(node_g->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode(), data1_g);
  EXPECT_EQ(node_g->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode(), const1_g);
}

/*
 *            --------> node3
 *          /           /   \
 *     node1        node2   const3
 *     / \          /   \
 * data1 const1  data2 const2
 */
TEST_F(ValueHolderUt, CreateDataOutOk2) {
  ge::Format fmt = ge::FORMAT_NC1HWC0;
  auto const1 = ValueHolder::CreateConst(&fmt, sizeof(fmt));
  auto data1 = ValueHolder::CreateFeed(0);
  auto node1 = ValueHolder::CreateSingleDataOutput("Node1", {const1, data1});

  auto const2 = ValueHolder::CreateConst(&fmt, sizeof(fmt));
  auto data2 = ValueHolder::CreateFeed(0);
  auto node2 = ValueHolder::CreateSingleDataOutput("Node1", {const2, data2});

  auto const3 = ValueHolder::CreateConst(&fmt, sizeof(fmt));
  auto n2_holder = ValueHolder::CreateVoid("Node2", {node1, node2, const3});

  for (const auto &holder : {const1, data1, node1, const2, data2, node2, const3, n2_holder}) {
    ASSERT_NE(holder, nullptr);
    ASSERT_TRUE(holder->IsOk());
    ASSERT_NE(holder->GetNode(), nullptr);
    ASSERT_NE(holder->GetGraph(), nullptr);
  }
  EXPECT_EQ(node1->GetNode()->GetType(), "Node1");
  EXPECT_EQ(node2->GetNode()->GetType(), "Node1");
  EXPECT_EQ(n2_holder->GetNode()->GetType(), "Node2");

  ASSERT_EQ(node1->GetNode()->GetAllOutDataAnchorsSize(), 1);
  ASSERT_EQ(node1->GetNode()->GetAllInDataAnchorsSize(), 2);
  EXPECT_EQ(node1->GetNode()->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode().get(), const1->GetNode());
  EXPECT_EQ(node1->GetNode()->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode().get(), data1->GetNode());

  ASSERT_EQ(n2_holder->GetNode()->GetAllOutDataAnchorsSize(), 0);
  ASSERT_EQ(n2_holder->GetNode()->GetAllInDataAnchorsSize(), 3);
  EXPECT_EQ(n2_holder->GetNode()->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode().get(), node1->GetNode());
  EXPECT_EQ(n2_holder->GetNode()->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode().get(), node2->GetNode());
  EXPECT_EQ(n2_holder->GetNode()->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode().get(), const3->GetNode());
}
TEST_F(ValueHolderUt, MergeIsolateNodeToGraphOk) {
  ge::Format f1 = ge::FORMAT_NC1HWC0;
  auto const1 = ValueHolder::CreateConst(reinterpret_cast<const uint8_t *>(&f1), sizeof(f1));
  auto data1 = ValueHolder::CreateFeed(0);
  auto node1 = ValueHolder::CreateDataOutput("Node1", {data1, const1}, 2);
  ASSERT_NE(const1, nullptr);
  ASSERT_NE(data1, nullptr);
  ASSERT_EQ(node1.size(), 2);
  ASSERT_NE(node1[0], nullptr);
  ASSERT_NE(node1[1], nullptr);

  ASSERT_NE(const1->GetNode()->GetOwnerComputeGraph(), nullptr);
  EXPECT_EQ(data1->GetNode()->GetOwnerComputeGraph(), const1->GetNode()->GetOwnerComputeGraph());
  EXPECT_EQ(node1[0]->GetNode()->GetOwnerComputeGraph(), const1->GetNode()->GetOwnerComputeGraph());
  EXPECT_EQ(node1[1]->GetNode()->GetOwnerComputeGraph(), const1->GetNode()->GetOwnerComputeGraph());

  ASSERT_NE(const1->GetGraph(), nullptr);
  EXPECT_EQ(const1->GetGraph(), const1->GetNode()->GetOwnerComputeGraph().get());
  EXPECT_EQ(data1->GetGraph(), const1->GetGraph());
  EXPECT_EQ(node1[0]->GetGraph(), const1->GetGraph());
  EXPECT_EQ(node1[1]->GetGraph(), const1->GetGraph());
}

/*
 *
 *           node3
 *          /     \   |
 *     node1       node2
 *     / \         /   \
 * data1 const1  data2 const2
 */
TEST_F(ValueHolderUt, MergeTwoGraphOk1) {
  ge::Format f1 = ge::FORMAT_NC1HWC0;
  auto const1 = ValueHolder::CreateConst(reinterpret_cast<const uint8_t *>(&f1), sizeof(f1));
  auto data1 = ValueHolder::CreateFeed(0);
  auto node1 = ValueHolder::CreateDataOutput("Node1", {data1, const1}, 1);

  auto const2 = ValueHolder::CreateConst(reinterpret_cast<const uint8_t *>(&f1), sizeof(f1));
  auto data2 = ValueHolder::CreateFeed(0);
  auto node2 = ValueHolder::CreateDataOutput("Node2", {data2, const2}, 2);

  auto node3 = ValueHolder::CreateSingleDataOutput("Node3", {node1[0], node2[0]});
  ASSERT_NE(node3, nullptr);

  EXPECT_NE(data1->GetNode()->GetOwnerComputeGraph(), nullptr);
  EXPECT_NE(const1->GetNode()->GetOwnerComputeGraph(), nullptr);
  EXPECT_NE(node1[0]->GetNode()->GetOwnerComputeGraph(), nullptr);

  EXPECT_NE(data2->GetNode()->GetOwnerComputeGraph(), nullptr);
  EXPECT_NE(const2->GetNode()->GetOwnerComputeGraph(), nullptr);
  EXPECT_NE(node2[0]->GetNode()->GetOwnerComputeGraph(), nullptr);
  EXPECT_NE(node2[1]->GetNode()->GetOwnerComputeGraph(), nullptr);

  EXPECT_NE(node3->GetNode()->GetOwnerComputeGraph(), nullptr);

  ASSERT_NE(const1->GetNode()->GetOwnerComputeGraph(), nullptr);
  EXPECT_EQ(const1->GetNode()->GetOwnerComputeGraph()->GetAllNodesSize(), 7);

  EXPECT_EQ(const1->GetGraph(), data1->GetGraph());
  EXPECT_EQ(const1->GetGraph(), node1[0]->GetGraph());
  EXPECT_EQ(const1->GetGraph(), data2->GetGraph());
  EXPECT_EQ(const1->GetGraph(), const2->GetGraph());
  EXPECT_EQ(const1->GetGraph(), node2[0]->GetGraph());
  EXPECT_EQ(const1->GetGraph(), node2[1]->GetGraph());
  EXPECT_EQ(const1->GetGraph(), node3->GetGraph());
}
/*
 *
 *                node4
 *               /   \
 *           node3    \
 *          /     \   |
 *     node1       node2
 *     / \         /   \
 * data1 const1  data2 const2
 */
TEST_F(ValueHolderUt, MergeTwoGraphOk) {
  ge::Format f1 = ge::FORMAT_NC1HWC0;
  auto const1 = ValueHolder::CreateConst(reinterpret_cast<const uint8_t *>(&f1), sizeof(f1));
  auto data1 = ValueHolder::CreateFeed(0);
  auto node1 = ValueHolder::CreateDataOutput("Node1", {data1, const1}, 1);
  ASSERT_NE(const1, nullptr);
  ASSERT_NE(data1, nullptr);
  ASSERT_EQ(node1.size(), 1);
  ASSERT_NE(node1[0], nullptr);
  ASSERT_NE(const1->GetNode()->GetOwnerComputeGraph(), nullptr);

  auto const2 = ValueHolder::CreateConst(reinterpret_cast<const uint8_t *>(&f1), sizeof(f1));
  auto data2 = ValueHolder::CreateFeed(0);
  auto node2 = ValueHolder::CreateDataOutput("Node2", {data2, const2}, 2);
  ASSERT_NE(const2, nullptr);
  ASSERT_NE(data2, nullptr);
  ASSERT_EQ(node2.size(), 2);
  ASSERT_NE(node2[0], nullptr);
  ASSERT_NE(node2[1], nullptr);

  EXPECT_NE(const2->GetNode()->GetOwnerComputeGraph(), nullptr);
  EXPECT_NE(node2[0]->GetNode()->GetOwnerComputeGraph(), nullptr);
  EXPECT_NE(node2[1]->GetNode()->GetOwnerComputeGraph(), nullptr);

  auto node3 = ValueHolder::CreateSingleDataOutput("Node3", {node1[0], node2[0]});
  ASSERT_NE(node3, nullptr);

  auto node4 = ValueHolder::CreateVoid("Node4", {node3, node2[1]});
  ASSERT_NE(node4, nullptr);

  EXPECT_NE(data1->GetNode()->GetOwnerComputeGraph(), nullptr);
  EXPECT_NE(node1[0]->GetNode()->GetOwnerComputeGraph(), nullptr);
  EXPECT_NE(data2->GetNode()->GetOwnerComputeGraph(), nullptr);
  EXPECT_NE(const2->GetNode()->GetOwnerComputeGraph(), nullptr);
  EXPECT_NE(node2[0]->GetNode()->GetOwnerComputeGraph(), nullptr);
  EXPECT_NE(node2[1]->GetNode()->GetOwnerComputeGraph(), nullptr);
  EXPECT_NE(node3->GetNode()->GetOwnerComputeGraph(), nullptr);
  EXPECT_NE(node4->GetNode()->GetOwnerComputeGraph(), nullptr);

  ASSERT_NE(const1->GetNode(), nullptr);
  ASSERT_NE(const1->GetNode()->GetOwnerComputeGraph(), nullptr);
  EXPECT_EQ(const1->GetNode()->GetOwnerComputeGraph()->GetAllNodesSize(), 8);

  EXPECT_EQ(const1->GetGraph(), data1->GetGraph());
  EXPECT_EQ(const1->GetGraph(), node1[0]->GetGraph());
  EXPECT_EQ(const1->GetGraph(), data2->GetGraph());
  EXPECT_EQ(const1->GetGraph(), const2->GetGraph());
  EXPECT_EQ(const1->GetGraph(), node2[0]->GetGraph());
  EXPECT_EQ(const1->GetGraph(), node2[1]->GetGraph());
  EXPECT_EQ(const1->GetGraph(), node3->GetGraph());
  EXPECT_EQ(const1->GetGraph(), node4->GetGraph());
}
TEST_F(ValueHolderUt, CreateVoidOk) {
  ge::Format f1 = ge::FORMAT_NC1HWC0;
  auto const1 = ValueHolder::CreateConst(reinterpret_cast<const uint8_t *>(&f1), sizeof(f1));
  auto data1 = ValueHolder::CreateFeed(0);
  ASSERT_NE(const1, nullptr);
  ASSERT_NE(data1, nullptr);

  std::vector<ValueHolderPtr> inputs = {data1, const1};
  auto holder = ValueHolder::CreateVoid("TestNode", inputs);

  ASSERT_NE(holder, nullptr);

  ASSERT_NE(const1->GetGraph(), nullptr);
  ASSERT_NE(const1->GetNode()->GetOwnerComputeGraph(), nullptr);
  ASSERT_NE(data1->GetGraph(), nullptr);
  ASSERT_NE(data1->GetNode()->GetOwnerComputeGraph(), nullptr);
  ASSERT_NE(holder->GetNode(), nullptr);
  ASSERT_NE(holder->GetGraph(), nullptr);
  ASSERT_NE(holder->GetNode()->GetOwnerComputeGraph(), nullptr);

  // check graph is ok
  auto graph = holder->GetGraph();
  ASSERT_EQ(graph->GetAllNodesSize(), 3);
  CheckGraphGenerally(*graph);

  auto const1_g = graph->FindFirstNodeMatchType("Const");
  auto data1_g = graph->FindFirstNodeMatchType("Data");
  auto node_g = graph->FindFirstNodeMatchType("TestNode");
  ASSERT_NE(const1_g, nullptr);
  ASSERT_NE(data1_g, nullptr);
  ASSERT_NE(node_g, nullptr);

  EXPECT_EQ(node_g->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode(), data1_g);
  EXPECT_EQ(node_g->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode(), const1_g);
}

TEST_F(ValueHolderUt, AddDependencyOk) {
  auto data1 = ValueHolder::CreateFeed(0);
  auto data2 = ValueHolder::CreateFeed(1);
  ValueHolder::AddDependency(data1, data2);

  auto node1 = ValueHolder::CreateSingleDataOutput("Node1", {data1});
  auto node2 = ValueHolder::CreateSingleDataOutput("Node1", {data1});
  ValueHolder::AddDependency(node1, node2);

  ASSERT_NE(data1, nullptr);
  ASSERT_NE(data2, nullptr);
  ASSERT_NE(node1, nullptr);
  ASSERT_NE(node2, nullptr);

  ASSERT_NE(data1->GetNode(), nullptr);
  ASSERT_NE(data2->GetNode(), nullptr);
  ASSERT_NE(node1->GetNode(), nullptr);
  ASSERT_NE(node2->GetNode(), nullptr);

  ASSERT_EQ(data1->GetNode()->GetOwnerComputeGraph(), data2->GetNode()->GetOwnerComputeGraph());
  ASSERT_EQ(data1->GetNode()->GetOwnerComputeGraph(), node1->GetNode()->GetOwnerComputeGraph());
  ASSERT_EQ(data1->GetNode()->GetOwnerComputeGraph(), node2->GetNode()->GetOwnerComputeGraph());

  ASSERT_EQ(data1->GetNode()->GetOutControlNodes().size(), 1);
  ASSERT_EQ(data2->GetNode()->GetInControlNodes().size(), 1);
  EXPECT_EQ(data1->GetNode()->GetOutControlAnchor()->GetPeerInControlAnchors().begin()->get()->GetOwnerNode().get(),
            data2->GetNode());

  ASSERT_EQ(node1->GetNode()->GetOutControlNodes().size(), 1);
  ASSERT_EQ(node2->GetNode()->GetInControlNodes().size(), 1);
  EXPECT_EQ(node1->GetNode()->GetOutControlAnchor()->GetPeerInControlAnchors().begin()->get()->GetOwnerNode().get(),
            node2->GetNode());
}

/*
 *           KernelLaunch
 *               |
 *             Tiling
 *             /    \
 *    InferShape   CompileInfo
 *     /   \          |
 * shape1  shape2   json
 */
TEST_F(ValueHolderUt, CurrentNodeOk) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  auto shape1 = ValueHolder::CreateFeed(0);
  auto shape2 = ValueHolder::CreateFeed(1);
  auto json1 = ValueHolder::CreateConst("{}", 3);

  ValueHolder::SetCurrentComputeNode(node);
  auto frame = ValueHolder::GetCurrentFrame();
  ASSERT_NE(frame, nullptr);
  ASSERT_EQ(frame->GetCurrentComputeNode(), node);
  auto shape = ValueHolder::CreateSingleDataOutput("InferShape", {shape1, shape2});
  auto compile_info = ValueHolder::CreateSingleDataOutput("TilingParse", {json1});
  auto tiling_ret = ValueHolder::CreateSingleDataOutput("Tiling", {shape, compile_info});
  auto holder = ValueHolder::CreateVoid("KernelLaunch", {tiling_ret});

  ASSERT_NE(shape1, nullptr);
  ASSERT_NE(shape2, nullptr);
  ASSERT_NE(json1, nullptr);
  ASSERT_NE(shape, nullptr);
  ASSERT_NE(compile_info, nullptr);
  ASSERT_NE(tiling_ret, nullptr);
  ASSERT_NE(holder, nullptr);

  int64_t compute_node_index_none;
  ASSERT_FALSE(ge::AttrUtils::GetInt(shape1->GetNode()->GetOpDesc(), "ComputeNodeIndex", compute_node_index_none));
  ASSERT_FALSE(ge::AttrUtils::GetInt(shape2->GetNode()->GetOpDesc(), "ComputeNodeIndex", compute_node_index_none));
  ASSERT_FALSE(ge::AttrUtils::GetInt(json1->GetNode()->GetOpDesc(), "ComputeNodeIndex", compute_node_index_none));

  int64_t compute_node_index_shape, compute_node_index_compile_ifo, compute_node_index_tiling_ret,
      compute_node_index_holder;
  ASSERT_TRUE(ge::AttrUtils::GetInt(shape->GetNode()->GetOpDesc(), "ComputeNodeIndex", compute_node_index_shape));
  ASSERT_TRUE(
      ge::AttrUtils::GetInt(compile_info->GetNode()->GetOpDesc(), "ComputeNodeIndex", compute_node_index_compile_ifo));
  ASSERT_TRUE(
      ge::AttrUtils::GetInt(tiling_ret->GetNode()->GetOpDesc(), "ComputeNodeIndex", compute_node_index_tiling_ret));
  ASSERT_TRUE(ge::AttrUtils::GetInt(holder->GetNode()->GetOpDesc(), "ComputeNodeIndex", compute_node_index_holder));
  EXPECT_EQ(compute_node_index_shape, compute_node_index_compile_ifo);
  EXPECT_EQ(compute_node_index_shape, compute_node_index_tiling_ret);
  EXPECT_EQ(compute_node_index_shape, compute_node_index_holder);

  size_t frame_current_node_index;
  frame->GetCurrentNodeIndex(frame_current_node_index);
  EXPECT_EQ(compute_node_index_shape, frame_current_node_index);
}
/*
 *    hello
 *    /  \
 * data0 data1
 */
TEST_F(ValueHolderUt, CreateExeGraphOk) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);

  ValueHolder::SetCurrentComputeNode(node);
  auto hello = ValueHolder::CreateSingleDataOutput("hello", {data0, data1});

  ValueHolder::AddRelevantInputNode(node);
  ASSERT_NE(graph, nullptr);
}
/*
 *    hello
 *    /  \
 * data0 data1
 */
TEST_F(ValueHolderUt, CreateExeGraphWithTargetsOk) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);

  ValueHolder::SetCurrentComputeNode(node);
  auto hello = ValueHolder::CreateVoid("hello", {data0, data1});
  ASSERT_NE(graph, nullptr);
}
/*
 *                      c
 * Atomic-LaunchKernel ----> LaunchKernel
 *          |                 /
 *    Atomic-tiling      Tiling
 *        /    \        /    \
 * TilingParse InferShape   TilingParse
 *    |         /   \          |
 *   json1   shape1  shape2   json2
 */
TEST_F(ValueHolderUt, ScopedCurrentNodeOk) {
  auto graph = std::make_shared<ge::ComputeGraph>("graph");

  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputOptional);
  auto node = graph->AddNode(op_desc);

  auto clean_op_desc = std::make_shared<ge::OpDesc>("node-AtomicClean", "DynamicAtomicAddrClean");
  clean_op_desc->AddInputDesc("workspace", tensor_desc);
  clean_op_desc->AddInputDesc("clean1", tensor_desc);
  clean_op_desc->AddInputDesc("clean2", tensor_desc);
  clean_op_desc->AppendIrInput("workspace", ge::kIrInputRequired);
  clean_op_desc->AppendIrInput("clean", ge::kIrInputDynamic);
  auto clean_node = graph->AddNode(clean_op_desc);

  auto shape1 = ValueHolder::CreateFeed(0);
  auto shape2 = ValueHolder::CreateFeed(1);
  auto json1 = ValueHolder::CreateConst("{}", 2);
  auto json2 = ValueHolder::CreateConst("{}", 3);

  ValueHolder::SetCurrentComputeNode(node);
  auto frame = ValueHolder::GetCurrentFrame();
  ASSERT_NE(frame, nullptr);
  ASSERT_EQ(frame->GetCurrentComputeNode(), node);
  auto shape = ValueHolder::CreateSingleDataOutput("InferShape", {shape1, shape2});

  size_t node1_index;
  ValueHolderPtr compile_info1, tiling_ret1, holder1;
  {
    auto guarder = ValueHolder::SetScopedCurrentComputeNode(clean_node);
    compile_info1 = ValueHolder::CreateSingleDataOutput("TilingParse", {json1});
    tiling_ret1 = ValueHolder::CreateSingleDataOutput("Tiling", {shape, compile_info1});
    holder1 = ValueHolder::CreateVoid("AtomicKernelLaunch", {tiling_ret1});
    EXPECT_TRUE(frame->GetCurrentNodeIndex(node1_index));
  }

  auto compile_info2 = ValueHolder::CreateSingleDataOutput("TilingParse", {json2});
  auto tiling_ret2 = ValueHolder::CreateSingleDataOutput("Tiling", {shape, compile_info2});
  auto holder2 = ValueHolder::CreateVoid("KernelLaunch", {tiling_ret2});

  ValueHolder::AddDependency(holder1, holder2);

  ASSERT_NE(shape1, nullptr);
  ASSERT_NE(shape2, nullptr);
  ASSERT_NE(json1, nullptr);
  ASSERT_NE(shape, nullptr);
  ASSERT_NE(compile_info1, nullptr);
  ASSERT_NE(tiling_ret1, nullptr);
  ASSERT_NE(holder1, nullptr);
  ASSERT_NE(compile_info2, nullptr);
  ASSERT_NE(tiling_ret2, nullptr);
  ASSERT_NE(holder2, nullptr);

  int64_t compute_node_index_none;
  ASSERT_FALSE(ge::AttrUtils::GetInt(shape1->GetNode()->GetOpDesc(), "ComputeNodeIndex", compute_node_index_none));
  ASSERT_FALSE(ge::AttrUtils::GetInt(shape2->GetNode()->GetOpDesc(), "ComputeNodeIndex", compute_node_index_none));
  ASSERT_FALSE(ge::AttrUtils::GetInt(json1->GetNode()->GetOpDesc(), "ComputeNodeIndex", compute_node_index_none));

  int64_t shape_index, compile_info1_index, tiling_ret1_index, holder1_index, compile_info2_index, tiling_ret2_index,
      holder2_index;
  ASSERT_TRUE(ge::AttrUtils::GetInt(shape->GetNode()->GetOpDesc(), "ComputeNodeIndex", shape_index));

  ASSERT_TRUE(ge::AttrUtils::GetInt(compile_info1->GetNode()->GetOpDesc(), "ComputeNodeIndex", compile_info1_index));
  ASSERT_TRUE(ge::AttrUtils::GetInt(tiling_ret1->GetNode()->GetOpDesc(), "ComputeNodeIndex", tiling_ret1_index));
  ASSERT_TRUE(ge::AttrUtils::GetInt(holder1->GetNode()->GetOpDesc(), "ComputeNodeIndex", holder1_index));

  ASSERT_TRUE(ge::AttrUtils::GetInt(compile_info2->GetNode()->GetOpDesc(), "ComputeNodeIndex", compile_info2_index));
  ASSERT_TRUE(ge::AttrUtils::GetInt(tiling_ret2->GetNode()->GetOpDesc(), "ComputeNodeIndex", tiling_ret2_index));
  ASSERT_TRUE(ge::AttrUtils::GetInt(holder2->GetNode()->GetOpDesc(), "ComputeNodeIndex", holder2_index));

  EXPECT_EQ(shape_index, compile_info2_index);
  EXPECT_EQ(shape_index, tiling_ret2_index);
  EXPECT_EQ(shape_index, holder2_index);

  EXPECT_NE(shape_index, compile_info1_index);
  EXPECT_EQ(compile_info1_index, tiling_ret1_index);
  EXPECT_EQ(compile_info1_index, holder1_index);
}

TEST_F(ValueHolderUt, CreateExeGraphNoOutpus) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);

  ValueHolder::SetCurrentComputeNode(node);
  auto hello = ValueHolder::CreateVoid("hello", {data0, data1});
}

TEST_F(ValueHolderUt, CreateExeGraphNoFrame) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);

  ValueHolder::SetCurrentComputeNode(node);
  auto hello = ValueHolder::CreateVoid("hello", {data0, data1});
}
/*
 *    hello
 *    /  \
 * data0 data1
 */
TEST_F(ValueHolderUt, GetCurrentGraphOk) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);

  ValueHolder::SetCurrentComputeNode(node);
  auto hello = ValueHolder::CreateVoid("hello", {data0, data1});

  EXPECT_NE(hello->GetCurrentFrame(), nullptr);
  EXPECT_NE(hello->GetCurrentGraph(), nullptr);
}
/*
 *       ref
 *     +------+
 *     |      |
 *   launch   |
 *     |      |
 *   tiling   |
 *     |      |
 *    alloc----
 *    /  \
 * data0 data1
 */
TEST_F(ValueHolderUt, RefFromOk) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);

  ValueHolder::SetCurrentComputeNode(node);
  auto alloc_outs = ValueHolder::CreateDataOutput("alloc", {data0, data1}, 3);
  auto tiling_outs = ValueHolder::CreateDataOutput("tiling", {data0, data1}, 2);
  tiling_outs[1]->RefFrom(alloc_outs[1]);

  auto launch = ValueHolder::CreateSingleDataOutput("launch", {tiling_outs[0], tiling_outs[1]});
  launch->RefFrom(alloc_outs[2]);
}
/*
 *    hello
 *    /  \
 * data0 data1
 */
TEST_F(ValueHolderUt, AddNullOutputs) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);

  ValueHolder::SetCurrentComputeNode(node);
  auto hello = ValueHolder::CreateSingleDataOutput("hello", {data0, data1});

  EXPECT_NE(hello->GetCurrentFrame(), nullptr);
  EXPECT_NE(hello->GetCurrentGraph(), nullptr);
}
/*
 *    hello
 *    /  \
 * data0 data1
 */
TEST_F(ValueHolderUt, AddNullTargets) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);

  ValueHolder::SetCurrentComputeNode(node);
  auto hello = ValueHolder::CreateSingleDataOutput("hello", {data0, data1});

  EXPECT_NE(hello->GetCurrentFrame(), nullptr);
  EXPECT_NE(hello->GetCurrentGraph(), nullptr);
}
TEST_F(ValueHolderUt, Guard_AddFlagToNode) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto allocator0 = ValueHolder::CreateSingleDataOutput("CreateAllocator", {data0});
  auto allocator_destroyer = ValueHolder::CreateVoidGuarder("DestroyAllocator", allocator0, {});
  ASSERT_NE(allocator_destroyer, nullptr);
  auto graph = ValueHolder::PopGraphFrame()->GetExeGraph();

  auto node = graph->FindFirstNodeMatchType("DestroyAllocator");
  ASSERT_NE(node, nullptr);
  int64_t index;
  EXPECT_TRUE(ge::AttrUtils::GetInt(node->GetOpDesc(), kReleaseResourceIndex, index));
  EXPECT_EQ(index, 0);
}
TEST_F(ValueHolderUt, Guarder_AddDependencyAutomately_ConnectDataEdgeToResource) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto allocator0 = ValueHolder::CreateSingleDataOutput("CreateAllocator", {data0});
  auto allocator_destroyer = ValueHolder::CreateVoidGuarder("DestroyAllocator", allocator0, {});
  ASSERT_NE(allocator_destroyer, nullptr);

  size_t alloc_size = 1024;
  auto size = ValueHolder::CreateConst(&alloc_size, sizeof(alloc_size));
  auto alloc_mem0 = ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator0, size});
  auto alloc_mem1 = ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator0, size});
  auto graph = ValueHolder::PopGraphFrame()->GetExeGraph();

  CheckGraphGenerally(*graph);

  ASSERT_NE(alloc_mem0, nullptr);
  ASSERT_NE(alloc_mem1, nullptr);
  HasControlEdge(*graph, *alloc_mem0->GetNode(), *allocator_destroyer->GetNode());
  HasControlEdge(*graph, *alloc_mem1->GetNode(), *allocator_destroyer->GetNode());
}/*
*     NetOutput
*        |
*      Bar -c-> foo0_guarder
*      / \    /
* data1   foo0
*          |
*        data0
*/
TEST_F(ValueHolderUt, Guarder_AddDependencyFromTheSameLevelNode_ConnectFromSrcToSubgraphNodes) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto foo0 = ValueHolder::CreateSingleDataOutput("Foo", {data0});
  auto guarder = ValueHolder::CreateVoidGuarder("FooGuarder", foo0, {});
  ASSERT_NE(guarder, nullptr);
  auto data1 = ValueHolder::CreateFeed(1);
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar", {data1});

  ValueHolder::PushGraphFrame(bar1, "BarGraph");
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {foo0, data1});
  auto bar_frame = ValueHolder::PopGraphFrame({foo1}, {});

  auto frame = ValueHolder::PopGraphFrame({bar1}, {});
  ASSERT_NE(frame, nullptr);
  ASSERT_NE(frame->GetExeGraph(), nullptr);

  EXPECT_EQ(frame->GetExeGraph()->TopologicalSorting(), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(NodeTopoChecker(bar1).OutChecker().CtrlToByType("FooGuarder").IsOk());
  EXPECT_EQ(NodeTopoChecker(bar1).StrictConnectFrom({{"Data"}, {"Foo"}}), "success");
}
TEST_F(ValueHolderUt, Guarder_DoNotAddDependency_ConnectDataEdgeToNetOutput) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto foo0 = ValueHolder::CreateSingleDataOutput("Foo", {data0});
  auto guarder = ValueHolder::CreateVoidGuarder("FooGuarder", foo0, {});
  ASSERT_NE(guarder, nullptr);

  auto bar0 = ValueHolder::CreateSingleDataOutput("Bar", {foo0});

  auto frame = ValueHolder::PopGraphFrame({foo0}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExeGraph();
  ASSERT_NE(graph, nullptr);

  EXPECT_EQ(NodeTopoChecker(foo0).StrictConnectTo(0, {{"NetOutput"}, {"FooGuarder"}, {"Bar"}}), "success");
  HasControlEdge(*graph, *bar0->GetNode(), *guarder->GetNode());
  ASSERT_EQ(guarder->GetNode()->GetInControlNodes().size(), 1);
}
TEST_F(ValueHolderUt, AddDependencyForGuard_RleaseBy) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto allocator0 = ValueHolder::CreateSingleDataOutput("CreateAllocator", {data0});
  auto allocator_destroyer = ValueHolder::CreateVoidGuarder("DestroyAllocator", allocator0, {});
  ASSERT_NE(allocator_destroyer, nullptr);

  size_t alloc_size = 1024;
  auto size = ValueHolder::CreateConst(&alloc_size, sizeof(alloc_size));
  auto alloc_mem0 = ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator0, size});
  auto free_mem0 = ValueHolder::CreateVoidGuarder("FreeMemory", {alloc_mem0}, {});
  auto graph = ValueHolder::PopGraphFrame()->GetExeGraph();
  CheckGraphGenerally(*graph);

  ASSERT_NE(free_mem0, nullptr);
  ASSERT_NE(alloc_mem0, nullptr);
  HasControlEdge(*graph, *alloc_mem0->GetNode(), *allocator_destroyer->GetNode());

  allocator0->ReleaseAfter(free_mem0);
  HasControlEdge(*graph, *free_mem0->GetNode(), *allocator_destroyer->GetNode());
}
TEST_F(ValueHolderUt, RleaseBy_NoGuarder) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto allocator0 = ValueHolder::CreateSingleDataOutput("CreateAllocator", {data0});

  size_t alloc_size = 1024;
  auto size = ValueHolder::CreateConst(&alloc_size, sizeof(alloc_size));
  auto alloc_mem0 = ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator0, size});
  auto graph = ValueHolder::PopGraphFrame()->GetExeGraph();

  CheckGraphGenerally(*graph);

  ASSERT_NE(alloc_mem0, nullptr);

  allocator0->ReleaseAfter(alloc_mem0);

  EXPECT_EQ(allocator0->GetNode()->GetOutAllNodes().size(), 1);
  EXPECT_EQ(allocator0->GetNode()->GetInAllNodes().size(), 1);

  EXPECT_EQ(alloc_mem0->GetNode()->GetOutAllNodes().size(), 0);
  EXPECT_EQ(alloc_mem0->GetNode()->GetInAllNodes().size(), 2);
}
TEST_F(ValueHolderUt, PushFrame_ChildFrameIsNotRoot) {
  ValueHolder::PopGraphFrame();
  auto root_frame = ValueHolder::PushGraphFrame();
  EXPECT_TRUE(root_frame->IsRootFrame());
  auto feed0 = ValueHolder::CreateFeed(0);
  auto child_frame = ValueHolder::PushGraphFrame(feed0, "subgraph_name0");
  ASSERT_NE(child_frame, nullptr);
  EXPECT_FALSE(child_frame->IsRootFrame());
}
TEST_F(ValueHolderUt, PushFrame_ComputeNodeIndexTheSame) {
  auto compute_node1 = FakeNode();
  auto compute_node2 = FakeNode();
  auto compute_node3 = FakeNode();

  ValueHolder::PopGraphFrame();
  auto root_frame = ValueHolder::PushGraphFrame();
  EXPECT_TRUE(root_frame->IsRootFrame());
  ValueHolder::SetCurrentComputeNode(compute_node1);
  auto feed0 = ValueHolder::CreateFeed(0);
  auto feed1 = ValueHolder::CreateFeed(1);
  auto foo0 = ValueHolder::CreateSingleDataOutput("Foo", {feed0, feed1});

  ValueHolder::SetCurrentComputeNode(compute_node2);
  auto bar0 = ValueHolder::CreateSingleDataOutput("Bar", {foo0});

  ValueHolder::PushGraphFrame(foo0, "subgraph_name0");
  auto sub_bar1 = ValueHolder::CreateSingleDataOutput("Bar", {feed0});
  ValueHolder::PopGraphFrame();
  ValueHolder::PopGraphFrame();

  EXPECT_EQ(GetComputeNodeIndex(sub_bar1->GetNode()), GetComputeNodeIndex(foo0->GetNode()));
  EXPECT_NE(GetComputeNodeIndex(bar0->GetNode()), GetComputeNodeIndex(foo0->GetNode()));
}

TEST_F(ValueHolderUt, PlacementDefault0) {
  auto data0 = ValueHolder::CreateFeed(0);
  EXPECT_EQ(data0->GetPlacement(), 0);
}
TEST_F(ValueHolderUt, SetGetPlacementOk) {
  auto data0 = ValueHolder::CreateFeed(0);
  data0->SetPlacement(1);
  EXPECT_EQ(data0->GetPlacement(), 1);
  data0->SetPlacement(2);
  EXPECT_EQ(data0->GetPlacement(), 2);
}
TEST_F(ValueHolderUt, BuildGraphWithDataOutput) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {data0, data1});
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar", {data0, data1});
  auto frame = ValueHolder::PopGraphFrame({foo1, bar1}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExeGraph();
  ASSERT_NE(graph, nullptr);

  CheckGraphGenerally(*graph);

  EXPECT_EQ(graph->GetAllNodesSize(), 5);

  auto nodes = ge::GraphUtils::FindNodesByTypeFromAllNodes(graph, "NetOutput");
  ASSERT_EQ(nodes.size(), 1);
  auto netoutput = nodes[0];
  ASSERT_NE(netoutput, nullptr);
  EXPECT_EQ(netoutput->GetInNodes().size(), 2);
  ASSERT_EQ(netoutput->GetInDataNodes().size(), 2);
  EXPECT_EQ((*netoutput->GetInDataNodes().begin())->GetType(), "Foo");
  EXPECT_EQ((*(netoutput->GetInDataNodes().begin() + 1))->GetType(), "Bar");
}
TEST_F(ValueHolderUt, BuildGraphWithCtrlOutput) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {data0, data1});
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar", {data0, data1});
  auto launch = ValueHolder::CreateVoid("Launch", {foo1, bar1});
  auto frame = ValueHolder::PopGraphFrame({}, {launch});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExeGraph();
  ASSERT_NE(graph, nullptr);

  CheckGraphGenerally(*graph);

  EXPECT_EQ(graph->GetAllNodesSize(), 6);

  auto nodes = ge::GraphUtils::FindNodesByTypeFromAllNodes(graph, "NetOutput");
  ASSERT_EQ(nodes.size(), 1);
  auto netoutput = nodes[0];
  ASSERT_NE(netoutput, nullptr);
  EXPECT_EQ(netoutput->GetInNodes().size(), 1);
  ASSERT_EQ(netoutput->GetInControlNodes().size(), 1);
  EXPECT_EQ((*netoutput->GetInControlNodes().begin())->GetType(), "Launch");
}
TEST_F(ValueHolderUt, AppendOutputOk) {
  auto foo = ValueHolder::CreateVoid("Foo", {});
  EXPECT_EQ(foo->GetNode()->GetAllOutDataAnchorsSize(), 0);

  auto outputs = foo->AppendOutputs(5);
  EXPECT_EQ(outputs.size(), 5);
  EXPECT_EQ(foo->GetNode()->GetAllOutDataAnchorsSize(), 5);

  auto bar = ValueHolder::CreateSingleDataOutput("Bar", outputs);
  EXPECT_NE(bar, nullptr);
  EXPECT_EQ(bar->GetNode()->GetAllInDataAnchorsSize(), 5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_NE(bar->GetNode()->GetInDataAnchor(i)->GetPeerOutAnchor(), nullptr);
    EXPECT_EQ(bar->GetNode()->GetInDataAnchor(i)->GetPeerOutAnchor()->GetIdx(), i);
    EXPECT_EQ(bar->GetNode()->GetInDataAnchor(i)->GetPeerOutAnchor()->GetOwnerNode(),
              foo->GetNode()->shared_from_this());
  }
}
TEST_F(ValueHolderUt, AppendOutputToNodeWithOutputs) {
  auto foo = ValueHolder::CreateDataOutput("Foo", {}, 3)[0];
  EXPECT_EQ(foo->GetNode()->GetAllOutDataAnchorsSize(), 3);

  auto outputs = foo->AppendOutputs(5);
  ASSERT_EQ(outputs.size(), 5);
  EXPECT_EQ(foo->GetNode()->GetAllOutDataAnchorsSize(), 8);

  auto bar = ValueHolder::CreateSingleDataOutput("Bar", outputs);
  EXPECT_NE(bar, nullptr);
  EXPECT_EQ(bar->GetNode()->GetAllInDataAnchorsSize(), 5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_NE(bar->GetNode()->GetInDataAnchor(i)->GetPeerOutAnchor(), nullptr);
    EXPECT_EQ(bar->GetNode()->GetInDataAnchor(i)->GetPeerOutAnchor()->GetIdx(), i + 3);
    EXPECT_EQ(bar->GetNode()->GetInDataAnchor(i)->GetPeerOutAnchor()->GetOwnerNode(),
              foo->GetNode()->shared_from_this());
  }
}
TEST_F(ValueHolderUt, ConnectFromAncestor_CreateInnerData_ParentGraph) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);
  auto foo = ValueHolder::CreateSingleDataOutput("Foo", {data0});
  ValueHolder::PushGraphFrame(foo, "Foo");
  auto sub_foo = ValueHolder::CreateSingleDataOutput("SubFoo", {data1, data2});
  ValueHolder::PopGraphFrame({sub_foo}, {});

  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExeGraph();
  ASSERT_NE(graph, nullptr);
  CheckGraphGenerally(*graph);

  EXPECT_EQ(SummaryChecker(graph).StrictDirectNodeTypes({{"Data", 3}, {"Foo", 1}}), "success");
  EXPECT_EQ(graph->GetAllSubgraphs().size(), 1);

  auto foo_node = graph->FindFirstNodeMatchType("Foo");
  ASSERT_NE(foo_node, nullptr);
  EXPECT_EQ(NodeTopoChecker(foo).StrictConnectFrom({data0, data1, data2}), "success");
  StrictSubgraphs(foo_node, {"Foo"});
  auto subgraph_name = foo_node->GetOpDesc()->GetSubgraphInstanceName(0);
  auto subgraph = graph->GetSubgraph(subgraph_name);
  ASSERT_NE(subgraph, nullptr);
  auto ret = gert::SummaryChecker(subgraph).StrictAllNodeTypes({{"InnerData", 2}, {"SubFoo", 1}, {"InnerNetOutput", 1}});
  EXPECT_EQ(ret, "success") << ret;

  auto sub_foo_node = subgraph->FindFirstNodeMatchType("SubFoo");
  ASSERT_NE(sub_foo_node, nullptr);
  ConnectFromInnerData(sub_foo_node, {1, 2});
  EXPECT_EQ(NodeTopoChecker(sub_foo_node).StrictConnectTo(0, {{"InnerNetOutput"}}), "success");
}

TEST_F(ValueHolderUt, ConnectFromAncestor_InnerDataWithGuarderOutside) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);
  auto foo = ValueHolder::CreateSingleDataOutput("Foo", {data0});
  auto data1_guarder = ValueHolder::CreateVoidGuarder("FreeMemory", data1, {});
  ValueHolder::PushGraphFrame(foo, "Foo");
  auto sub_foo = ValueHolder::CreateSingleDataOutput("SubFoo", {data1, data2});

  auto sub_frame = ValueHolder::PopGraphFrame({sub_foo}, {});
  auto subgraph = sub_frame->GetExeGraph();
  ASSERT_NE(subgraph, nullptr);
  auto innerdata_node = subgraph->FindFirstNodeMatchType("InnerData");
  std::string guarder_type_outside;
  (void) ge::AttrUtils::GetStr(innerdata_node->GetOpDesc(), kNodeWithGuarderOutside, guarder_type_outside);
  EXPECT_EQ(!guarder_type_outside.empty(), true);
  EXPECT_EQ(guarder_type_outside, "FreeMemory");
}

TEST_F(ValueHolderUt, ConnectFromAncestor_InnerDataWithGuarderOutside_In_Subgraph_Nesting) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);
  auto foo = ValueHolder::CreateSingleDataOutput("Foo", {data0});
  auto data2_guarder = ValueHolder::CreateVoidGuarder("FreeFftsMem", data2, {});

  ValueHolder::PushGraphFrame(foo, "Foo");
  auto sub_foo = ValueHolder::CreateSingleDataOutput("SubFoo", {data1});

  ValueHolder::PushGraphFrame(sub_foo, "SubFoo");
  auto sub_sub_foo = ValueHolder::CreateSingleDataOutput("SubFoo", {data2});

  auto sub_sub_frame = ValueHolder::PopGraphFrame({sub_sub_foo}, {});
  auto sub_sub_graph = sub_sub_frame->GetExeGraph();

  auto innerdata_node = sub_sub_graph->FindFirstNodeMatchType("InnerData");
  std::string guarder_type_outside;
  (void) ge::AttrUtils::GetStr(innerdata_node->GetOpDesc(), kNodeWithGuarderOutside, guarder_type_outside);
  EXPECT_EQ(!guarder_type_outside.empty(), true);
  EXPECT_EQ(guarder_type_outside, "FreeFftsMem");
}

/*
 * +-----------------------------+
 * |Foo                          |
 * |   +---------------------+   |
 * |   |SubFoo               |   |
 * |   |    NetOutput        |   |
 * |   |        |            |   |
 * |   |      Sub2Foo3       |   |
 * |   |       /    \        |   |
 * |   | Sub2Foo1  Sub2Foo2  |   |
 * |   |   |       /   |     |   |
 * |   +---+-----+-----+-----+   |
 * |       |     |     |         |
 * +-------+-----+-----+---------+
 *    /    |     |     |
 * data0 data1 data2 data3
 */
TEST_F(ValueHolderUt, ConnectFromAncestor_CreateInnerDataRecursively_AncestorGraph) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);
  auto data3 = ValueHolder::CreateFeed(3);

  auto foo = ValueHolder::CreateSingleDataOutput("Foo", {data0});
  ValueHolder::PushGraphFrame(foo, "Foo");

  auto sub_foo = ValueHolder::CreateSingleDataOutput("SubFoo", {data1});
  ValueHolder::PushGraphFrame(sub_foo, "Foo");

  auto sub2_foo1 = ValueHolder::CreateSingleDataOutput("Sub2Foo1", {data1});
  auto sub2_foo2 = ValueHolder::CreateSingleDataOutput("Sub2Foo2", {data3, data2});
  auto sub2_foo3 = ValueHolder::CreateSingleDataOutput("Sub2Foo3", {sub2_foo1, sub2_foo2});

  ValueHolder::PopGraphFrame({sub2_foo3}, {});
  ValueHolder::PopGraphFrame({sub_foo}, {});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExeGraph();
  ASSERT_NE(graph, nullptr);
  CheckGraphGenerally(*graph);

  // Check elements on root graph
  EXPECT_EQ(SummaryChecker(graph).StrictDirectNodeTypes({{"Data", 4}, {"Foo", 1}}), "success");
  EXPECT_EQ(graph->GetAllSubgraphs().size(), 2);
  auto foo_node = graph->FindFirstNodeMatchType("Foo");
  ASSERT_NE(foo_node, nullptr);
  EXPECT_EQ(NodeTopoChecker(foo_node).StrictConnectFrom({data0, data1, data3, data2}), "success");
  StrictSubgraphs(foo_node, {"Foo"});
  ASSERT_EQ(foo_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);
  auto foo_graph = graph->GetSubgraph(foo_node->GetOpDesc()->GetSubgraphInstanceName(0));
  ASSERT_NE(foo_graph, nullptr);

  // Check elements on foo graph
  ASSERT_EQ(SummaryChecker(foo_graph).StrictDirectNodeTypes({{"InnerData", 3}, {"SubFoo", 1}, {"InnerNetOutput", 1}}),
            "success");
  auto sub_foo_node = foo_graph->FindFirstNodeMatchType("SubFoo");
  ASSERT_NE(sub_foo_node, nullptr);
  ASSERT_EQ(NodeTopoChecker(sub_foo_node).StrictConnectFrom({{"InnerData"}, {"InnerData"}, {"InnerData"}}), "success");
  ASSERT_EQ(NodeTopoChecker(sub_foo_node).StrictConnectTo(0, {{"InnerNetOutput"}}), "success");
  StrictSubgraphs(sub_foo_node, {"Foo"});
  auto subfoo_graph = graph->GetSubgraph(sub_foo_node->GetOpDesc()->GetSubgraphInstanceName(0));
  ASSERT_NE(subfoo_graph, nullptr);

  // Check elements on SubFoo graph
  auto ret = gert::SummaryChecker(subfoo_graph)
                 .StrictAllNodeTypes(
                     {{"InnerData", 3}, {"Sub2Foo1", 1}, {"Sub2Foo2", 1}, {"Sub2Foo3", 1}, {"InnerNetOutput", 1}});
  auto sub2_foo1_node = subfoo_graph->FindFirstNodeMatchType("Sub2Foo1");
  ASSERT_NE(sub2_foo1_node, nullptr);
  EXPECT_EQ(NodeTopoChecker(sub2_foo1_node).StrictConnectFrom({{"InnerData"}}), "success");
  ConnectFromOuter(sub2_foo1_node, 0, FindData(graph, 1), 0);

  auto sub2_foo2_node = subfoo_graph->FindFirstNodeMatchType("Sub2Foo2");
  ASSERT_NE(sub2_foo2_node, nullptr);
  EXPECT_EQ(NodeTopoChecker(sub2_foo2_node).StrictConnectFrom({{"InnerData"}, {"InnerData"}}), "success");
  ConnectFromOuter(sub2_foo2_node, 0, FindData(graph, 3), 0);
  ConnectFromOuter(sub2_foo2_node, 1, FindData(graph, 2), 0);

  auto sub2_foo3_node = subfoo_graph->FindFirstNodeMatchType("Sub2Foo3");
  ASSERT_NE(sub2_foo3_node, nullptr);
  ASSERT_EQ(NodeTopoChecker(sub2_foo3_node).StrictConnectFrom({sub2_foo1_node, sub2_foo2_node}), "success");
}

/*
 * +---------------------------------------+
 * |Foo                                    |
 * |   +-------------------------------+   |
 * |   |SubFoo                         |   |
 * |   |         NetOutput             |   |
 * |   |             |                 |   |
 * |   |          Sub2Foo3             |   |
 * |   |       /     |      \          |   |
 * |   | Sub2Foo1  Sub2Foo2   Sub2Foo4 |   |
 * |   |   |       /      \ /          |   |
 * |   +---+-----+---------+-----------+   |
 * |       |     |         |               |
 * +-------+-----+---------+---------------+
 *    /    |     |         |
 * data0 data1 data2     data3
 */
TEST_F(ValueHolderUt, ConnectFromAncestor_DeDuplicate_SameSrc) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);
  auto data3 = ValueHolder::CreateFeed(3);

  auto foo = ValueHolder::CreateSingleDataOutput("Foo", {data0});
  ValueHolder::PushGraphFrame(foo, "Foo");

  auto sub_foo = ValueHolder::CreateSingleDataOutput("SubFoo", {data1});
  ValueHolder::PushGraphFrame(sub_foo, "Foo");

  auto sub2_foo1 = ValueHolder::CreateSingleDataOutput("Sub2Foo1", {data1});
  auto sub2_foo2 = ValueHolder::CreateSingleDataOutput("Sub2Foo2", {data3, data2});
  auto sub2_foo4 = ValueHolder::CreateSingleDataOutput("Sub2Foo4", {data3});
  auto sub2_foo3 = ValueHolder::CreateSingleDataOutput("Sub2Foo3", {sub2_foo1, sub2_foo2, sub2_foo4});

  ValueHolder::PopGraphFrame({sub2_foo3}, {});
  ValueHolder::PopGraphFrame({sub_foo}, {});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExeGraph();
  ASSERT_NE(graph, nullptr);
  CheckGraphGenerally(*graph);

  auto foo_node = graph->FindFirstNodeMatchType("Foo");
  ASSERT_NE(foo_node, nullptr);
  ASSERT_EQ(NodeTopoChecker(foo_node).StrictConnectFrom({{"Data"}, {"Data"}, {"Data"}, {"Data"}}), "success");

  auto sub_foo_node = ge::GraphUtils::FindNodesByTypeFromAllNodes(graph, "SubFoo")[0];
  ASSERT_NE(sub_foo_node, nullptr);
  ASSERT_EQ(NodeTopoChecker(sub_foo_node).StrictConnectFrom({{"InnerData"}, {"InnerData"}, {"InnerData"}}), "success");
  auto sub_foo_graph = FindFirstSubgraphForNodeType(graph, "SubFoo");
  ASSERT_NE(sub_foo_graph, nullptr);
  auto sub2_foo2_node = sub_foo_graph->FindFirstNodeMatchType("Sub2Foo2");
  ASSERT_NE(sub2_foo2_node, nullptr);
  ASSERT_EQ(NodeTopoChecker(sub2_foo2_node).StrictConnectFrom({{"InnerData"}, {"InnerData"}}), "success");
  ConnectFromOuter(sub2_foo2_node, 0, FindData(graph, 3), 0);
  ConnectFromOuter(sub2_foo2_node, 1, FindData(graph, 2), 0);

  auto sub2_foo4_node = sub_foo_graph->FindFirstNodeMatchType("Sub2Foo4");
  ASSERT_EQ(NodeTopoChecker(sub2_foo4_node).StrictConnectFrom({{"InnerData"}}), "success");
  ConnectFromOuter(sub2_foo4_node, 0, FindData(graph, 3), 0);

  auto inner_data_from_3 = sub2_foo2_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  ASSERT_EQ(NodeTopoChecker(inner_data_from_3).StrictConnectTo(0, {sub2_foo2_node, sub2_foo4_node}), "success");
}
TEST_F(ValueHolderUt, PopFrame_CreateControlEdge_Targets) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto foo = ValueHolder::CreateSingleDataOutput("Foo", {data0, data1});
  auto frame = ValueHolder::PopGraphFrame({}, {data0, foo});

  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExeGraph();
  ASSERT_NE(graph, nullptr);

  ASSERT_EQ(SummaryChecker(graph).StrictAllNodeTypes({{"Data", 2}, {"Foo", 1}, {"NetOutput", 1}}), "success");

  auto netoutput = graph->FindFirstNodeMatchType("NetOutput");
  ASSERT_NE(netoutput, nullptr);
  ASSERT_EQ(NodeTopoChecker(netoutput).StrictConnectFrom({{"Data", -1}, {"Foo", -1}}), "success");
  EXPECT_EQ(netoutput->GetInDataNodes().size(), 0);
  EXPECT_EQ(netoutput->GetInControlNodes().size(), 2);
}
TEST_F(ValueHolderUt, PopFrame_CreateNetOuptut_PopRootGraph) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto foo = ValueHolder::CreateSingleDataOutput("Foo", {data0, data1});
  auto frame = ValueHolder::PopGraphFrame({data0, foo}, {});

  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExeGraph();
  ASSERT_NE(graph, nullptr);

  ASSERT_EQ(SummaryChecker(graph).StrictAllNodeTypes({{"Data", 2}, {"Foo", 1}, {"NetOutput", 1}}), "success");

  auto netoutput = graph->FindFirstNodeMatchType("NetOutput");
  ASSERT_NE(netoutput, nullptr);
  EXPECT_EQ(netoutput->GetName(), "NetOutput");
  ASSERT_EQ(NodeTopoChecker(netoutput).StrictConnectFrom({data0, foo}), "success");

  auto foo_node = graph->FindFirstNodeMatchType("Foo");
  ASSERT_NE(foo_node, nullptr);
  ASSERT_EQ(NodeTopoChecker(foo_node).StrictConnectFrom({data0, data1}), "success");
  ASSERT_EQ(NodeTopoChecker(foo_node).StrictConnectTo(0, {netoutput}), "success");
}
TEST_F(ValueHolderUt, PopFrame_CreateInnerNetOuptut_PopSubgraph) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto foo = ValueHolder::CreateSingleDataOutput("Foo", {data0, data1});
  ValueHolder::PushGraphFrame(foo, "Foo");
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {data0});
  auto bar2 = ValueHolder::CreateSingleDataOutput("Bar2", {data1});
  auto frame = ValueHolder::PopGraphFrame({bar1}, {bar2});

  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExeGraph();
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(SummaryChecker(graph).StrictAllNodeTypes({{"InnerData", 2}, {"Bar1", 1}, {"Bar2", 1}, {"InnerNetOutput", 1}}), "success");
  auto netoutput = graph->FindFirstNodeMatchType("InnerNetOutput");
  ASSERT_NE(netoutput, nullptr);
  ASSERT_EQ(NodeTopoChecker(netoutput).StrictConnectFrom({{"Bar1"}, {"Bar2"}}), "success");
}
/*
 * +-----------------------------+
 * |Foo                          |
 * |   +---------------------+   |
 * |   |SubFoo               |   |
 * |   |    NetOutput        |   |
 * |   |  /   |      \       |   |
 * |   | | foo5      |       |   |
 * |   | |  |        |       |   |
 * |   +-+--+--------+-------+   |
 * |     |  |        |           |
 * |    /  Foo2    Foo3          |
 * |   |  /       /     \        |
 * +---+---------+--------+------+
 *     |         |        |
 *  data0      data1    data2
 */
TEST_F(ValueHolderUt, PopFrame_CraeteInnerData_OutputsUseParentHolder) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {data0, data1, data2});
  ValueHolder::PushGraphFrame(foo1, "Foo");

  auto foo2 = ValueHolder::CreateSingleDataOutput("Foo2", {data0, data1});
  auto foo3 = ValueHolder::CreateDataOutput("Foo3", {data1, data2}, 3);

  auto foo4 = ValueHolder::CreateSingleDataOutput("Foo4", {foo2, foo3[0]});
  ValueHolder::PushGraphFrame(foo4, "Foo");
  auto foo5 = ValueHolder::CreateSingleDataOutput("Foo5", {foo2});

  ValueHolder::PopGraphFrame({data0, foo3[1], foo5}, {});
  ValueHolder::PopGraphFrame({}, {});

  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExeGraph();
  ASSERT_NE(graph, nullptr);

  auto foo3_nodes = ge::GraphUtils::FindNodesByTypeFromAllNodes(const_cast<ge::ComputeGraphPtr &>(graph), "Foo3");
  ASSERT_EQ(foo3_nodes.size(), 1);

  auto foo4_graph = FindFirstSubgraphForNodeType(graph, "Foo4");
  ASSERT_NE(foo4_graph, nullptr);
  ASSERT_EQ(SummaryChecker(foo4_graph).StrictAllNodeTypes({{"InnerData", 3}, {"Foo5", 1}, {"InnerNetOutput", 1}}),
            "success");
  auto netoutput = foo4_graph->FindFirstNodeMatchType("InnerNetOutput");
  ASSERT_EQ(NodeTopoChecker(netoutput).StrictConnectFrom({{"InnerData"}, {"InnerData"}, {"Foo5"}}), "success");
  ConnectFromOuter(netoutput, 0, FindData(graph, 0), 0);
  ConnectFromOuter(netoutput, 1, foo3_nodes[0], 1);
}

/*
 * +--------------------------------------------------------+
 * |Foo-Node                                                |
 * |   +---------------------+    +---------------------+   |
 * |   |Foo-Subgraph1        |    |Foo-Subgraph2        |   |
 * |   |   NetOutput         |    |          NetOutput  |   |
 * |   |      |              |    |     ERROR   |       |   |
 * |   |     Bar1            |    | Bar1 --->  Bar2     |   |
 * |   |   /    \            |    |          /    \     |   |
 * |   +--0------1-----------+    +---------0------1----+   |
 * +------0------1--------2---------------------------------+
 *        |      |        |
 *     data0   data1    data2
 */
TEST_F(ValueHolderUt, PopFrame_Failed_OutpusUseGraphNotAncestor) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {data0, data1, data2});
  ValueHolder::PushGraphFrame(foo1, "Foo1");
  auto bar1 = ValueHolder::CreateDataOutput("Bar1", {data0, data1}, 3);
  ValueHolder::PopGraphFrame({bar1[0]}, {});

  ValueHolder::PushGraphFrame(foo1, "Foo2");
  auto bar2 = ValueHolder::CreateSingleDataOutput("Bar2", {bar1[0], data0, data1});
  ASSERT_EQ(bar2, nullptr);
  bar2 = ValueHolder::CreateSingleDataOutput("Bar2", {bar1[1], data0, data1});
  ASSERT_EQ(bar2, nullptr);
  ValueHolder::PopGraphFrame();
}

TEST_F(ValueHolderUt, CreateConstDataOk) {
  ge::Format f1 = ge::FORMAT_NC1HWC0;
  auto const_data1 = ValueHolder::CreateConstData(0);
  auto data1 = ValueHolder::CreateFeed(0);
  ASSERT_NE(const_data1, nullptr);
  ASSERT_NE(data1, nullptr);

  std::vector<ValueHolderPtr> inputs = {data1, const_data1};
  auto holder = ValueHolder::CreateVoid("TestNode", inputs);

  ASSERT_NE(holder, nullptr);

  ASSERT_NE(const_data1->GetGraph(), nullptr);
  ASSERT_NE(const_data1->GetNode()->GetOwnerComputeGraph(), nullptr);
  ASSERT_NE(data1->GetGraph(), nullptr);
  ASSERT_NE(data1->GetNode()->GetOwnerComputeGraph(), nullptr);
  ASSERT_NE(holder->GetNode(), nullptr);
  ASSERT_NE(holder->GetGraph(), nullptr);
  ASSERT_NE(holder->GetNode()->GetOwnerComputeGraph(), nullptr);

  // check graph is ok
  auto graph = holder->GetGraph();
  ASSERT_EQ(graph->GetAllNodesSize(), 3);
  CheckGraphGenerally(*graph);

  auto const1_g = graph->FindFirstNodeMatchType("ConstData");
  auto data1_g = graph->FindFirstNodeMatchType("Data");
  auto node_g = graph->FindFirstNodeMatchType("TestNode");
  ASSERT_NE(const1_g, nullptr);
  ASSERT_NE(data1_g, nullptr);
  ASSERT_NE(node_g, nullptr);

  EXPECT_EQ(node_g->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode(), data1_g);
  EXPECT_EQ(node_g->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode(), const1_g);
}
}  // namespace bg
}  // namespace gert

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

#ifndef AIR_CXX_TESTS_UT_GE_RUNTIME_V2_COMMON_BG_TEST_H_
#define AIR_CXX_TESTS_UT_GE_RUNTIME_V2_COMMON_BG_TEST_H_
#include <gtest/gtest.h>
#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "exe_graph/runtime/continuous_buffer.h"
#include "exe_graph/runtime/context_extend.h"
namespace gert {
class BgTest : public testing::Test {
 protected:
  void SetUp() override {
    Test::SetUp();
    bg::ValueHolder::PushGraphFrame();
  }
  void TearDown() override {
    Test::TearDown();
    while (bg::ValueHolder::PopGraphFrame())
      ;
  }

 public:
  void CheckGraphGenerally(const ge::ComputeGraph &graph) {
    CheckNamesUniq(graph);
    CheckOwners(graph);
    CheckSubgraphExists(graph);
    CheckDataIndexOnAllSubgraphs(graph);
  }

  void CheckExeGraphGenerally(const ge::ComputeGraph &graph) {
    CheckGraphGenerally(graph);
    CheckKernelExtendInfoOk(graph);
  }
  void CheckComputeNodeInfoOk(const ge::ComputeGraph &root_graph,
                              const std::map<bg::ValueHolderPtr, ge::NodePtr> &holders_to_compute_node) {
    ge::Buffer buffer;
    ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(root_graph.shared_from_this(), kComputeNodeInfo, buffer));
    auto compute_nodes_info = reinterpret_cast<const ContinuousBuffer *>(buffer.GetData());
    ASSERT_NE(compute_nodes_info, nullptr);

    ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(root_graph.shared_from_this(), kBuffer, buffer));
    auto exe_buffer = reinterpret_cast<const ContinuousBuffer *>(buffer.GetData());
    ASSERT_NE(exe_buffer, nullptr);

    std::map<const ge::Node *, ge::NodePtr> exe_nodes_to_compute_node;
    for (const auto &holder_to_cnode : holders_to_compute_node) {
      exe_nodes_to_compute_node[holder_to_cnode.first->GetNode()] = holder_to_cnode.second;
    }

    for (const auto &exe_node : root_graph.GetAllNodes()) {
      auto enode_to_cnode = exe_nodes_to_compute_node.find(exe_node.get());
      if (enode_to_cnode == exe_nodes_to_compute_node.end()) {
        continue;
      }
      auto cnode = enode_to_cnode->second;

      int64_t index;
      ASSERT_TRUE(ge::AttrUtils::GetInt(exe_node->GetOpDesc(), kComputeNodeIndex, index))
          << "The node " << exe_node->GetName() << " Type " << exe_node->GetType()
          << " does not have attr ComputeNodeIndex";

      auto compute_node_info = compute_nodes_info->Get<ComputeNodeInfo>(index);
      ASSERT_NE(compute_node_info, nullptr)
          << "The node " << exe_node->GetName() << " Type " << exe_node->GetType() << " ComputeNodeIndex invalid, "
          << index << ", total num " << compute_nodes_info->GetNum();

      auto name_index = reinterpret_cast<size_t>(compute_node_info->GetNodeName());
      auto name = exe_buffer->Get<char>(name_index);
      ASSERT_NE(name, nullptr) << "The node " << exe_node->GetName() << " Type " << exe_node->GetType()
                               << " does not have a valid name index";
      EXPECT_STREQ(cnode->GetName().c_str(), name);

      auto type_index = reinterpret_cast<size_t>(compute_node_info->GetNodeType());
      auto type = exe_buffer->Get<char>(type_index);
      ASSERT_NE(type, nullptr) << "The node " << exe_node->GetName() << " Type " << exe_node->GetType()
                               << " does not have a valid type index";
      EXPECT_STREQ(cnode->GetType().c_str(), type);

      // todo check input, output, attrs...
    }
  }

  void HasControlEdge(const ge::ComputeGraph &graph, const ge::Node &src_node, const ge::Node &dst_node) {
    auto src_anchor = src_node.GetOutControlAnchor();
    auto dst_anchor = dst_node.GetInControlAnchor();
    EXPECT_TRUE(src_anchor->IsLinkedWith(dst_anchor));
  }

 private:
  void CheckDataIndex(const ge::ComputeGraph &graph) {
    std::map<int32_t, std::string> indexes_to_name;
    for (const auto &node : graph.GetDirectNode()) {
      if (node->GetType() == "Data" || node->GetType() == "InnerData") {
        int32_t index;
        ASSERT_TRUE(ge::AttrUtils::GetInt(node->GetOpDesc(), "index", index))
            << "Can not get index attr on data " << node->GetName();
        ASSERT_TRUE(indexes_to_name.emplace(index, node->GetName()).second)
            << "Duplicated index on data " << node->GetName() << " and data " << indexes_to_name[index] << ", on graph "
            << graph.GetName();
      }
    }
  }
  void CheckDataIndexOnAllSubgraphs(const ge::ComputeGraph &graph) {
    CheckDataIndex(graph);
    for (const auto &subgraph : graph.GetAllSubgraphs()) {
      CheckDataIndex(*subgraph);
    }
  }
  void CheckKernelExtendInfoOk(const ge::ComputeGraph &root_graph) {
    ge::Buffer buffer;
    ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(root_graph.shared_from_this(), kKernelExtendInfo, buffer));
    auto extend_info_buffer = reinterpret_cast<const ContinuousBuffer *>(buffer.GetData());
    ASSERT_NE(extend_info_buffer, nullptr);

    ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(root_graph.shared_from_this(), kBuffer, buffer));
    auto exe_buffer = reinterpret_cast<const ContinuousBuffer *>(buffer.GetData());
    ASSERT_NE(exe_buffer, nullptr);

    std::set<std::string> nodes_does_have_extend_info = {"NetOutput"};
    for (const auto &node : root_graph.GetAllNodes()) {
      if (nodes_does_have_extend_info.count(node->GetType()) > 0) {
        continue;
      }
      int64_t index;
      EXPECT_TRUE(ge::AttrUtils::GetInt(node->GetOpDesc(), kKernelExtendIndex, index))
          << "The node " << node->GetName() << " Type " << node->GetType() << " does not have a KernelExtendInfoIndex";
      ASSERT_LT(static_cast<size_t>(index), extend_info_buffer->GetNum())
          << "The node " << node->GetName() << " Type " << node->GetType()
          << " KernelExtendInfoIndex out of range: " << index << " > " << extend_info_buffer->GetNum();
      auto extend_info = extend_info_buffer->Get<KernelExtendInfo>(index);
      ASSERT_NE(extend_info, nullptr);

      auto name_index = reinterpret_cast<bg::BufferPool::BufId>(extend_info->GetKernelName());
      auto name = exe_buffer->Get<char>(name_index);
      ASSERT_NE(name, nullptr) << "The node " << node->GetName() << " Type " << node->GetType()
                               << " does not have a valid name index";
      auto type_index = reinterpret_cast<bg::BufferPool::BufId>(extend_info->GetKernelType());
      auto type = exe_buffer->Get<char>(type_index);
      ASSERT_NE(type, nullptr) << "The node " << node->GetName() << " Type " << node->GetType()
                               << " does not have a valid type index";

      EXPECT_STREQ(node->GetName().c_str(), name);
      EXPECT_STREQ(node->GetType().c_str(), type);
    }
  }
  void CheckOwners(const ge::ComputeGraph &root_graph) {
    auto &non_const_graph = const_cast<ge::ComputeGraph &>(root_graph);
    for (const auto &node : root_graph.GetDirectNode()) {
      EXPECT_EQ(node->GetOwnerComputeGraph(), non_const_graph.shared_from_this());
      EXPECT_EQ(non_const_graph.GetParentNode(), nullptr);
      EXPECT_EQ(non_const_graph.GetParentGraph(), nullptr);
    }
    for (const auto &subgraph : root_graph.GetAllSubgraphs()) {
      auto owner_node = subgraph->GetParentNode();
      ASSERT_NE(owner_node, nullptr);
      EXPECT_EQ(owner_node->GetOwnerComputeGraph(), subgraph->GetParentGraph());
      for (const auto &node : subgraph->GetDirectNode()) {
        EXPECT_EQ(node->GetOwnerComputeGraph(), subgraph)
            << "The node " << node->GetName() << "(" << node->GetType() << ")"
            << " owner graph name " << node->GetOwnerComputeGraph()->GetName() << ", expect " << subgraph->GetName();
      }
    }
  }
  void CheckSubgraphExists(const ge::ComputeGraph &root_graph) {
    for (const auto &node : root_graph.GetAllNodes()) {
      for (const auto &subgraph_name : node->GetOpDesc()->GetSubgraphInstanceNames()) {
        EXPECT_NE(root_graph.GetSubgraph(subgraph_name), nullptr)
            << "Subgraph " << subgraph_name << " does not exists on root graph, node name " << node->GetName();
      }
    }
  }
  void CheckSubgraphAndParentNodeIoNum(const ge::ComputeGraph &root_graph) {
    for (const auto &node : root_graph.GetAllNodes()) {
      for (const auto &subgraph_name : node->GetOpDesc()->GetSubgraphInstanceNames()) {
        auto subgraph = root_graph.GetSubgraph(subgraph_name);
        std::vector<ge::NodePtr> data_nodes;
        ge::NodePtr netoutput_node;
        for (const auto &subgraph_node : subgraph->GetDirectNode()) {
          if (subgraph_node->GetType() == "Data") {
            data_nodes.emplace_back(subgraph_node);
            continue;
          }
          if (subgraph_node->GetType() == "NetOutput") {
            ASSERT_EQ(netoutput_node, nullptr);
            netoutput_node = subgraph_node;
          }
        }
        ASSERT_NE(netoutput_node, nullptr);
        auto parent_node = subgraph->GetParentNode();
        ASSERT_NE(parent_node, nullptr);
        EXPECT_EQ(parent_node->GetAllInDataAnchorsSize(), data_nodes.size());
        EXPECT_EQ(parent_node->GetAllOutDataAnchorsSize(), netoutput_node->GetAllInDataAnchorsSize());
      }
    }
  }
  void CheckNamesUniq(const ge::ComputeGraph &graph) {
    std::set<std::string> node_names;
    for (const auto &node : graph.GetAllNodes()) {
      node_names.emplace(node->GetName());
    }
    EXPECT_EQ(node_names.size(), graph.GetAllNodesSize());
  }
};
}  // namespace gert
#endif  //AIR_CXX_TESTS_UT_GE_RUNTIME_V2_COMMON_BG_TEST_H_

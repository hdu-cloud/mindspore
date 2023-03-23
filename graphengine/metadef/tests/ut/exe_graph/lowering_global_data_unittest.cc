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
#include "exe_graph/lowering/lowering_global_data.h"
#include "exe_graph/lowering/frame_selector.h"
#include <gtest/gtest.h>
#include "checker/bg_test.h"
namespace gert {
namespace {
ge::NodePtr BuildTestNode() {
  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  return graph->AddNode(op_desc);
}
}
class LoweringGlobalDataUT : public BgTest {};
TEST_F(LoweringGlobalDataUT, SetGetStreamOk) {
  LoweringGlobalData gd;

  EXPECT_EQ(gd.GetStream(), nullptr);

  auto holder = bg::ValueHolder::CreateFeed(0);
  auto holder1 = holder;
  gd.SetStream(std::move(holder1));
  EXPECT_EQ(gd.GetStream(), holder);
}
TEST_F(LoweringGlobalDataUT, SetGetCompileResultOk) {
  LoweringGlobalData gd;

  auto node = BuildTestNode();
  ASSERT_NE(node, nullptr);

  EXPECT_EQ(gd.FindCompiledResult(node), nullptr);

  gd.AddCompiledResult(node, {});
  ASSERT_NE(gd.FindCompiledResult(node), nullptr);
  EXPECT_TRUE(gd.FindCompiledResult(node)->GetTaskDefs().empty());
}

TEST_F(LoweringGlobalDataUT, SetGetKnownSubgraphModelOk) {
  LoweringGlobalData gd;

  auto node = BuildTestNode();
  ASSERT_NE(node, nullptr);

  EXPECT_EQ(gd.FindKnownSubgraphModel(node), nullptr);

  gd.AddKnownSubgraphModel(node, reinterpret_cast<void *>(0x123));
  EXPECT_EQ(gd.FindKnownSubgraphModel(node), reinterpret_cast<void *>(0x123));
}

TEST_F(LoweringGlobalDataUT, SetGetAllocatorOk) {
  LoweringGlobalData gd;
  EXPECT_EQ(gd.GetAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput}), nullptr);
  auto holder = bg::ValueHolder::CreateFeed(0);
  ASSERT_NE(holder, nullptr);
  gd.SetAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput}, holder);
  EXPECT_EQ(gd.GetAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput}), holder);
}

TEST_F(LoweringGlobalDataUT, GetOrCreateAllocatorOk) {
  LoweringGlobalData gd;
  auto allocator1 = gd.GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(allocator1, nullptr);
  EXPECT_EQ(allocator1, gd.GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput}));
}

TEST_F(LoweringGlobalDataUT, GetOrCreateAllocatorFromExternalAllocatorOk) {
  LoweringGlobalData gd;
  auto holder = bg::ValueHolder::CreateFeed(0);
  auto holder1 = holder;
  gd.SetExternalAllocator(std::move(holder1));
  auto allocator1 = gd.GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(allocator1, nullptr);
}

TEST_F(LoweringGlobalDataUT, GetSessionIdSameWithOneGlobalData) {
  LoweringGlobalData gd;

  auto session_id1 = gd.GetSessionId();
  EXPECT_NE(session_id1, std::numeric_limits<uint64_t>::max());
  auto session_id2 = gd.GetSessionId();
  EXPECT_EQ(session_id1, session_id2);
}

TEST_F(LoweringGlobalDataUT, GetSessionIdDifferentWithDiffGlobalData) {
  LoweringGlobalData gd1;

  auto session_id1 = gd1.GetSessionId();
  EXPECT_NE(session_id1, std::numeric_limits<uint64_t>::max());
  LoweringGlobalData gd2;
  auto session_id2 = gd2.GetSessionId();
  EXPECT_EQ(session_id2, session_id1 + 1U);
}

TEST_F(LoweringGlobalDataUT, GetOrCreateAllocator_AlwaysCreateOnRootFrame_CallInSubgraph) {
  LoweringGlobalData gd;

  auto data0 = bg::ValueHolder::CreateFeed(0);
  auto foo1 = bg::ValueHolder::CreateSingleDataOutput("Foo", {data0});

  bg::ValueHolder::PushGraphFrame(foo1, "FooGraph");
  auto allocator1 = gd.GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(allocator1, nullptr);
  ASSERT_NE(bg::ValueHolder::PopGraphFrame(), nullptr);

  auto root_frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);

  ASSERT_EQ(allocator1->GetNode()->GetOwnerComputeGraph(), root_frame->GetExeGraph());
}

TEST_F(LoweringGlobalDataUT, GetOrCreateUniqueValueHolderOk) {
  LoweringGlobalData gd;
  auto builder = [&]() -> bg::ValueHolderPtr {
    auto resource_holder = bg::FrameSelector::OnMainRoot([&]() -> std::vector<bg::ValueHolderPtr> {
      std::string name = "aicpu_resource";
      auto name_holder = bg::ValueHolder::CreateConst(name.c_str(), name.size(), true);
      auto create_container_holder = bg::ValueHolder::CreateSingleDataOutput("CreateStepContainer", {name_holder});
      bg::ValueHolder::CreateVoidGuarder("DestroyStepContainer", create_container_holder, {});
      return {create_container_holder};
    });
    return resource_holder[0];
  };
  auto holder_0 = gd.GetOrCreateUniqueValueHolder("aicpu_container_0", builder);
  EXPECT_NE(holder_0, nullptr);

  auto clear_builder = [&]() -> bg::ValueHolderPtr {
    return bg::ValueHolder::CreateVoid("ClearStepContainer", {holder_0});
  };
  auto clear_holder = bg::FrameSelector::OnMainRootLast(clear_builder);
  EXPECT_NE(clear_holder, nullptr);
  std::string create_resource_name = holder_0->GetNode()->GetOpDesc()->GetName();
  EXPECT_EQ(create_resource_name.find("CreateStepContainer"), 0);

  auto last_exec_nodes = bg::ValueHolder::GetLastExecNodes();
  EXPECT_EQ(last_exec_nodes.size(), 1);
  EXPECT_NE(last_exec_nodes[0], nullptr);
  std::string clear_resource_name = last_exec_nodes[0]->GetNode()->GetOpDesc()->GetName();
  EXPECT_EQ(clear_resource_name.find("ClearStepContainer"), 0);

  // use same key: aicpu_container_0, check unique
  auto holder_1 = gd.GetOrCreateUniqueValueHolder("aicpu_container_0", builder);
  EXPECT_EQ(last_exec_nodes.size(), 1);
  last_exec_nodes.clear();
}

TEST_F(LoweringGlobalDataUT, OnMainRootLastOk) {
  LoweringGlobalData gd; 
  uint64_t global_container_id = 0;
  auto builder = [&]() -> bg::ValueHolderPtr {
    uint64_t container_id = global_container_id++;
    auto container_id_holder = bg::ValueHolder::CreateConst(&container_id, sizeof(uint64_t));
    uint64_t session_id = gd.GetSessionId();
    auto session_id_holder = bg::ValueHolder::CreateConst(&session_id, sizeof(uint64_t));
    auto resource_holder = bg::FrameSelector::OnMainRoot([&]() -> std::vector<bg::ValueHolderPtr> {
      auto create_session_holder = bg::ValueHolder::CreateSingleDataOutput(
        "CreateSession", {session_id_holder});
      bg::ValueHolder::CreateVoidGuarder("DestroySession", create_session_holder, {});
      auto clear_builder = [&]() -> bg::ValueHolderPtr {
        return bg::ValueHolder::CreateVoid("ClearStepContainer", {session_id_holder, container_id_holder});
      };
      auto clear_holder = bg::FrameSelector::OnMainRootLast(clear_builder);
      EXPECT_NE(clear_holder, nullptr);
      return {container_id_holder};
    });
    return resource_holder[0];
  };
  auto holder_0 = gd.GetOrCreateUniqueValueHolder("aicpu_container_0", builder);
  EXPECT_NE(holder_0, nullptr);

  auto last_exec_nodes = bg::ValueHolder::GetLastExecNodes();
  EXPECT_EQ(last_exec_nodes.size(), 1);
  EXPECT_NE(last_exec_nodes[0], nullptr);
  std::string clear_resource_name = last_exec_nodes[0]->GetNode()->GetOpDesc()->GetName();
  EXPECT_EQ(clear_resource_name.find("ClearStepContainer"), 0);

  // use same key: aicpu_container_0, check unique
  auto holder_1 = gd.GetOrCreateUniqueValueHolder("aicpu_container_0", builder);
  EXPECT_EQ(last_exec_nodes.size(), 1);
  last_exec_nodes.clear();
}
}  // namespace gert
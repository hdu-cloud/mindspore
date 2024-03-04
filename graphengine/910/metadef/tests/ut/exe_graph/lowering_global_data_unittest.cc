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
#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/runtime/execute_graph_types.h"
#include "checker/summary_checker.h"
#include "checker/topo_checker.h"
#include "exe_graph/lowering/lowering_opt.h"
#include "graph/utils/graph_utils.h"

namespace gert {
namespace {
ge::NodePtr BuildTestNode() {
  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  return graph->AddNode(op_desc);
}
}  // namespace
class LoweringGlobalDataUT : public BgTest {
 protected:
  void SetUp() override {
    BgTest::SetUp();
  }

  void InitTestFrames() {
    root_frame = bg::ValueHolder::GetCurrentFrame();
    auto init_node = bg::ValueHolder::CreateVoid("Init", {});
    bg::ValueHolder::PushGraphFrame(init_node, "Init");
    init_frame = bg::ValueHolder::PopGraphFrame();

    auto de_init_node = bg::ValueHolder::CreateVoid("DeInit", {});
    bg::ValueHolder::PushGraphFrame(de_init_node, "DeInit");
    de_init_frame = bg::ValueHolder::PopGraphFrame();

    auto main_node = bg::ValueHolder::CreateVoid(GetExecuteGraphTypeStr(ExecuteGraphType::kMain), {});
    bg::ValueHolder::PushGraphFrame(main_node, "Main");
  }
  void InitTestFramesWithStream(LoweringGlobalData &global_data) {
    root_frame = bg::ValueHolder::GetCurrentFrame();
    auto init_node = bg::ValueHolder::CreateVoid("Init", {});
    bg::ValueHolder::PushGraphFrame(init_node, "Init");
    global_data.SetStream(bg::ValueHolder::CreateFeed(-1), ExecuteGraphType::kInit);
    init_frame = bg::ValueHolder::PopGraphFrame();

    auto de_init_node = bg::ValueHolder::CreateVoid("DeInit", {});
    bg::ValueHolder::PushGraphFrame(de_init_node, "DeInit");
    de_init_frame = bg::ValueHolder::PopGraphFrame();

    auto main_node = bg::ValueHolder::CreateVoid(GetExecuteGraphTypeStr(ExecuteGraphType::kMain), {});
    bg::ValueHolder::PushGraphFrame(main_node, "Main");
    global_data.SetStream(bg::ValueHolder::CreateFeed(-1), ExecuteGraphType::kMain);
  }
  bg::GraphFrame *root_frame;
  std::unique_ptr<bg::GraphFrame> init_frame;
  std::unique_ptr<bg::GraphFrame> de_init_frame;
};
TEST_F(LoweringGlobalDataUT, SetGetStreamOk) {
  LoweringGlobalData gd;

  EXPECT_EQ(gd.GetStream(), nullptr);

  auto holder = bg::ValueHolder::CreateFeed(0);
  auto holder1 = holder;
  gd.SetStream(std::move(holder1));
  EXPECT_EQ(gd.GetStream(), holder);
}
TEST_F(LoweringGlobalDataUT, GetStream_HolderOnInit_GetOnInit) {
  LoweringGlobalData gd;
  EXPECT_EQ(gd.GetStream(), nullptr);

  InitTestFramesWithStream(gd);
  std::vector<bg::ValueHolderPtr> graph_out;
  std::vector<bg::ValueHolderPtr> node_out;
  auto ret = bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> { return {gd.GetStream()}; },
                                           graph_out, node_out);

  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  ASSERT_NE(graph_out[0], nullptr);
  ASSERT_EQ(graph_out[0]->GetNode()->GetOwnerComputeGraph()->GetParentNode()->GetType(), "Init");
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

TEST_F(LoweringGlobalDataUT, SetGetKnownSubgraphModel) {
  LoweringGlobalData gd;

  std::string graph_name = "graph";

  EXPECT_EQ(gd.GetGraphStaticCompiledModel(graph_name), nullptr);

  gd.AddStaticCompiledGraphModel(graph_name, reinterpret_cast<void *>(0x123));
  EXPECT_EQ(gd.GetGraphStaticCompiledModel(graph_name), reinterpret_cast<void *>(0x123));
}

TEST_F(LoweringGlobalDataUT, GetOrCreateAllocatorOk) {
  InitTestFrames();
  LoweringGlobalData gd;
  auto allocator1 = gd.GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(allocator1, nullptr);
  EXPECT_EQ(allocator1, gd.GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput}));
}

TEST_F(LoweringGlobalDataUT, GetOrCreateAllocator_InitRootCreateSync1) {
  InitTestFrames();
  LoweringGlobalData gd;
  auto holder = gd.GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(holder, nullptr);

  ASSERT_EQ(gd.GetAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput}), holder);

  std::vector<bg::ValueHolderPtr> on_init;
  std::vector<bg::ValueHolderPtr> on_root;
  auto ret = bg::FrameSelector::OnInitRoot(
      [&]() -> std::vector<bg::ValueHolderPtr> {
        auto allocator = gd.GetAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
        return {allocator};
      },
      on_init, on_root);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  ASSERT_EQ(on_init.size(), 1U);
  ASSERT_EQ(on_root.size(), 1U);
  ASSERT_NE(on_init[0], nullptr);
  ASSERT_NE(on_root[0], nullptr);
}
TEST_F(LoweringGlobalDataUT, GetOrCreateAllocator_InitRootCreateSync2) {
  InitTestFrames();
  LoweringGlobalData gd;
  std::vector<bg::ValueHolderPtr> on_init;
  std::vector<bg::ValueHolderPtr> on_root;
  auto ret = bg::FrameSelector::OnInitRoot(
      [&]() -> std::vector<bg::ValueHolderPtr> {
        return {gd.GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput})};
      },
      on_init, on_root);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  ASSERT_NE(on_init[0], nullptr);

  ASSERT_NE(gd.GetAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput}), nullptr);
}
TEST_F(LoweringGlobalDataUT, GetOrCreateAllocator_CreateSelectAllocator_MainExternalAllocatorSet) {
  InitTestFrames();
  LoweringGlobalData gd;
  gd.SetStream(bg::ValueHolder::CreateFeed(-1));
  gd.SetExternalAllocator(bg::ValueHolder::CreateFeed(-2));

  auto allocator1 = gd.GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(allocator1, nullptr);
  EXPECT_EQ(allocator1->GetNode()->GetType(), "SelectAllocator");
  EXPECT_EQ(NodeTopoChecker(allocator1).StrictConnectFrom(
              {{"InnerData"}, {"InnerData"}, {"Data"}, {"InnerData"}, {"Data"}}),
            "success");
  auto create_allocator_node = init_frame->GetExeGraph()->FindFirstNodeMatchType("CreateAllocator");
  ASSERT_NE(create_allocator_node, nullptr);
  ConnectFromInitToMain(create_allocator_node.get(), 0, allocator1->GetNode(), 3);

  bg::ValueHolderPtr init_allocator = nullptr;
  bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    init_allocator = gd.GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
    return {};
  });
  ASSERT_NE(init_allocator, nullptr);
  EXPECT_EQ(init_allocator->GetNode()->GetType(), "CreateAllocator");
  EXPECT_EQ(NodeTopoChecker(init_allocator).StrictConnectFrom({{"Const"}, {"Const"}}), "success");
}
TEST_F(LoweringGlobalDataUT, GetOrCreateAllocator_CreateSelectAllocator_ExternalAllocatorSet) {
  InitTestFrames();
  LoweringGlobalData gd;
  gd.SetStream(bg::ValueHolder::CreateFeed(-1));
  gd.SetExternalAllocator(bg::ValueHolder::CreateFeed(-2));
  bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    gd.SetStream(bg::ValueHolder::CreateFeed(-1), ExecuteGraphType::kInit);
    gd.SetExternalAllocator(bg::ValueHolder::CreateFeed(-2), ExecuteGraphType::kInit);
    return {};
  });

  auto allocator1 = gd.GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(allocator1, nullptr);
  EXPECT_EQ(allocator1->GetNode()->GetType(), "SelectAllocator");
  EXPECT_EQ(NodeTopoChecker(allocator1).StrictConnectFrom(
              {{"InnerData"}, {"InnerData"}, {"Data"}, {"InnerData"}, {"Data"}}),
            "success");
  auto create_allocator_node = init_frame->GetExeGraph()->FindFirstNodeMatchType("CreateAllocator");
  ASSERT_NE(create_allocator_node, nullptr);
  ConnectFromInitToMain(create_allocator_node.get(), 0, allocator1->GetNode(), 3);

  bg::ValueHolderPtr init_allocator = nullptr;
  bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    init_allocator = gd.GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
    return {};
  });
  ASSERT_NE(init_allocator, nullptr);
  EXPECT_EQ(init_allocator->GetNode()->GetType(), "SelectAllocator");
  EXPECT_EQ(NodeTopoChecker(init_allocator).StrictConnectFrom(
              {{"Const"}, {"Const"}, {"Data"}, {"CreateAllocator"}, {"Data"}}),
            "success");
}

TEST_F(LoweringGlobalDataUT, GetOrCreateAllocator_ExternalAllocatorSet_UseAlwaysExternalAllocatorOption) {
  InitTestFrames();
  LoweringGlobalData gd;
  gd.SetStream(bg::ValueHolder::CreateFeed(-1));
  gd.SetExternalAllocator(bg::ValueHolder::CreateFeed(-2));
  LoweringOption opt;
  opt.always_external_allocator = true;
  gd.SetLoweringOption(opt);
  bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    gd.SetStream(bg::ValueHolder::CreateFeed(-1), ExecuteGraphType::kInit);
    gd.SetExternalAllocator(bg::ValueHolder::CreateFeed(-2), ExecuteGraphType::kInit);
    return {};
  });

  auto allocator1 = gd.GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(allocator1, nullptr);
  EXPECT_EQ(allocator1->GetNode()->GetType(), "SelectAllocator");

  auto create_allocator_node = init_frame->GetExeGraph()->FindFirstNodeMatchType("CreateAllocator");
  // 外置allocator后，图中就不存在CreateAllocator节点了
  ASSERT_EQ(create_allocator_node, nullptr);

  auto get_allocator_node = init_frame->GetExeGraph()->FindFirstNodeMatchType("GetAllocator");
  // 外置allocator后，init
  ASSERT_NE(get_allocator_node, nullptr);

  bg::ValueHolderPtr init_allocator = nullptr;
  bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    init_allocator = gd.GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
    return {};
  });
  ASSERT_NE(init_allocator, nullptr);
  EXPECT_EQ(init_allocator->GetNode()->GetType(), "GetAllocator");
  EXPECT_EQ(NodeTopoChecker(init_allocator).StrictConnectFrom(
            {{"Const"}, {"Const"}, {"Data"}}),
            "success");
}

TEST_F(LoweringGlobalDataUT, GetOrCreateAllocator_AlwaysReturnOnRootFrame_CallInSubgraph) {
  InitTestFrames();
  LoweringGlobalData gd;

  auto data0 = bg::ValueHolder::CreateFeed(0);
  auto foo1 = bg::ValueHolder::CreateSingleDataOutput("Foo", {data0});

  bg::ValueHolder::PushGraphFrame(foo1, "FooGraph");
  auto allocator1 = gd.GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(allocator1, nullptr);
  ASSERT_NE(bg::ValueHolder::PopGraphFrame(), nullptr);

  ASSERT_EQ(allocator1->GetNode()->GetOwnerComputeGraph(), root_frame->GetExeGraph());
}

TEST_F(LoweringGlobalDataUT, GetOrCreateAllocator_AlwaysCreateOnInitFrame_CallInSubgraph) {
  InitTestFrames();
  LoweringGlobalData gd;

  auto data0 = bg::ValueHolder::CreateFeed(0);
  auto foo1 = bg::ValueHolder::CreateSingleDataOutput("Foo", {data0});

  bg::ValueHolder::PushGraphFrame(foo1, "FooGraph");
  auto allocator1 = gd.GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(allocator1, nullptr);

  ASSERT_NE(bg::ValueHolder::PopGraphFrame(), nullptr);

  ASSERT_EQ(SummaryChecker(init_frame->GetExeGraph())
                .StrictAllNodeTypes({{"CreateAllocator", 1}, {"Const", 2}, {"InnerNetOutput", 1}}),
            "success");
}
TEST_F(LoweringGlobalDataUT, GetOrCreateAllocator_ReturnOnInit_WhenGetOnInit) {
  InitTestFrames();
  LoweringGlobalData gd;

  auto data0 = bg::ValueHolder::CreateFeed(0);
  auto foo1 = bg::ValueHolder::CreateSingleDataOutput("Foo", {data0});

  bg::ValueHolder::PushGraphFrame(foo1, "FooGraph");
  auto allocator1 = gd.GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(allocator1, nullptr);
  ASSERT_EQ(allocator1->GetNode()->GetType(), "Init");

  std::vector<bg::ValueHolderPtr> graph_out;
  std::vector<bg::ValueHolderPtr> node_out;
  auto ret = bg::FrameSelector::OnInitRoot(
      [&]() -> std::vector<bg::ValueHolderPtr> {
        return {gd.GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput})};
      },
      graph_out, node_out);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  ASSERT_EQ(graph_out.size(), 1);
  auto init_node = graph_out[0]->GetNode()->GetOwnerComputeGraph()->GetParentNode();
  ASSERT_NE(init_node, nullptr);
  ASSERT_EQ(init_node->GetType(), "Init");
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
    uint64_t session_id = 0;
    auto session_id_holder = bg::ValueHolder::CreateConst(&session_id, sizeof(uint64_t));
    auto resource_holder = bg::FrameSelector::OnMainRoot([&]() -> std::vector<bg::ValueHolderPtr> {
      auto create_session_holder = bg::ValueHolder::CreateSingleDataOutput("CreateSession", {session_id_holder});
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

TEST_F(LoweringGlobalDataUT, SinkWeightInfoTest) {
  LoweringGlobalData gd;
  size_t weight_info = 1;
  gd.SetModelWeightSize(weight_info);
  auto result = gd.GetModelWeightSize();
  EXPECT_EQ(result, weight_info);
}

TEST_F(LoweringGlobalDataUT, GetValueHolersSizeTest) {
  LoweringGlobalData gd;
  gd.SetValueHolders("test1", nullptr);
  EXPECT_EQ(gd.GetValueHoldersSize("test1"), 1);
  EXPECT_EQ(gd.GetValueHoldersSize("test2"), 0);
  gd.SetValueHolders("test1", nullptr);
  EXPECT_EQ(gd.GetValueHoldersSize("test1"), 2);

  gd.SetUniqueValueHolder("test3", nullptr);
  EXPECT_EQ(gd.GetValueHoldersSize("test3"), 1);
  gd.SetUniqueValueHolder("test3", nullptr);
  EXPECT_EQ(gd.GetValueHoldersSize("test3"), 1);
}
}  // namespace gert
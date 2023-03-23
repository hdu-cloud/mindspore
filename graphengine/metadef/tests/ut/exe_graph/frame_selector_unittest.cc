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
#include "exe_graph/lowering/frame_selector.h"
#include <gtest/gtest.h>
#include "checker/bg_test.h"
#include "checker/summary_checker.h"
#include "checker/topo_checker.h"
#include "exe_graph/runtime/execute_graph_types.h"
namespace gert {
namespace bg {
class FrameSelectorUT : public BgTest {
 public:
  void InitTestFrames() {
    auto init_node = ValueHolder::CreateVoid("Init", {});
    ValueHolder::PushGraphFrame(init_node, "Init");
    ValueHolder::PopGraphFrame();

    auto de_init_node = ValueHolder::CreateVoid("DeInit", {});
    ValueHolder::PushGraphFrame(de_init_node, "DeInit");
    ValueHolder::PopGraphFrame();

    auto main_node = ValueHolder::CreateVoid(GetExecuteGraphTypeStr(ExecuteGraphType::kMain), {});
    ValueHolder::PushGraphFrame(main_node, "Main");
  }
};
/*
 * +-----------------------+
 * |FooGraph               |
 * |                       |
 * |   InnerNetOutput      |
 * |      |                |
 * |     foo2 <--+         |
 * |    /         \        |
 * |  c1           \       |
 * +---+-----+------+------+
 *     |     |      |
 *   data0 data1   bar1
 *                  |
 *                 c0
 */
TEST_F(FrameSelectorUT, SelectMainRoot_CreateOnRoot_NoMainGraph) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);

  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data0, data1});
  ValueHolder::PushGraphFrame(foo1, "FooGraph");

  // on FooGraph
  auto c1 = ValueHolder::CreateConst("ConstData", 10, true);

  // on RootGraph
  auto bars = FrameSelector::OnMainRoot([&]() -> std::vector<ValueHolderPtr> {
    auto c0 = ValueHolder::CreateConst("ConstData", 10, true);
    auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {c0});
    return {bar1};
  });
  ASSERT_EQ(bars.size(), 1);
  ASSERT_NE(bars[0], nullptr);

  // on FooGraph
  auto foo2 = ValueHolder::CreateSingleDataOutput("Foo2", {c1, bars[0]});
  auto foo2_graph = ValueHolder::PopGraphFrame({foo2}, {});
  ASSERT_NE(foo2_graph, nullptr);

  auto frame = ValueHolder::PopGraphFrame();

  ASSERT_EQ(
      SummaryChecker(frame->GetExeGraph()).StrictDirectNodeTypes({{"Data", 2}, {"Const", 1}, {"Foo1", 1}, {"Bar1", 1}}),
      "success");
  ASSERT_EQ(SummaryChecker(foo2_graph->GetExeGraph())
                .StrictDirectNodeTypes({{"InnerData", 1}, {"Const", 1}, {"Foo2", 1}, {"InnerNetOutput", 1}}),
            "success");

  ASSERT_EQ(NodeTopoChecker(bars[0]).StrictConnectTo(0, {{"Foo1", 2}}), "success");
  ASSERT_EQ(NodeTopoChecker(bars[0]).StrictConnectFrom({{"Const"}}), "success");
}
TEST_F(FrameSelectorUT, SelectMainRoot_CreateOnMainRoot_CurrentFrameIsMainRoot) {
  InitTestFrames();

  auto data0 = ValueHolder::CreateFeed(0);

  // on RootGraph
  auto bars = FrameSelector::OnMainRoot([&]() -> std::vector<ValueHolderPtr> {
    auto c0 = ValueHolder::CreateConst("ConstData", 10, true);
    auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {c0, data0});
    return {bar1};
  });
  ASSERT_EQ(bars.size(), 1);
  ASSERT_NE(bars[0], nullptr);

  auto main_frame = ValueHolder::PopGraphFrame();
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(main_frame, nullptr);
  ASSERT_NE(root_frame, nullptr);

  ASSERT_EQ(SummaryChecker(root_frame->GetExeGraph()).StrictDirectNodeTypes({{"Init", 1}, {"Main", 1}, {"DeInit", 1}}), "success");
  ASSERT_EQ(SummaryChecker(main_frame->GetExeGraph()).StrictDirectNodeTypes({{"Data", 1}, {"Const", 1}, {"Bar1", 1}}), "success");

  ASSERT_EQ(NodeTopoChecker(bars[0]).StrictConnectFrom({{"Const"}, {"Data"}}), "success");
  ASSERT_TRUE(bars[0]->GetNode()->GetOutNodes().empty());
}

/*
 * +-----------------------+
 * |FooGraph               |
 * |                       |
 * |   InnerNetOutput      |
 * |      |                |
 * |     foo2 <--+         |
 * |    /         \        |
 * |  c1           \       |
 * +---+-----+------+------+
 *     |     |      |
 *   data0 data1   bar1
 *                  |
 *                 c0
 */
TEST_F(FrameSelectorUT, SelectMainRoot_CreateOnMainRoot_CurrentFrameIsMainSubgraphs) {
  InitTestFrames();

  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);

  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data0, data1});
  ValueHolder::PushGraphFrame(foo1, "FooGraph");

  // on FooGraph
  auto c1 = ValueHolder::CreateConst("ConstData", 10, true);

  // on MainRootGraph
  auto bars = FrameSelector::OnMainRoot([&]() -> std::vector<ValueHolderPtr> {
    auto c0 = ValueHolder::CreateConst("ConstData", 10, true);
    auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {c0});
    return {bar1};
  });
  ASSERT_EQ(bars.size(), 1);
  ASSERT_NE(bars[0], nullptr);

  // on FooGraph
  auto foo2 = ValueHolder::CreateSingleDataOutput("Foo2", {c1, bars[0]});
  auto foo2_graph = ValueHolder::PopGraphFrame({foo2}, {});
  ASSERT_NE(foo2_graph, nullptr);

  auto frame = ValueHolder::PopGraphFrame();

  ASSERT_EQ(
      SummaryChecker(frame->GetExeGraph()).StrictDirectNodeTypes({{"Data", 2}, {"Const", 1}, {"Foo1", 1}, {"Bar1", 1}}),
      "success");
  ASSERT_EQ(SummaryChecker(foo2_graph->GetExeGraph())
                .StrictDirectNodeTypes({{"InnerData", 1}, {"Const", 1}, {"Foo2", 1}, {"InnerNetOutput", 1}}),
            "success");

  ASSERT_EQ(NodeTopoChecker(bars[0]).StrictConnectTo(0, {{"Foo1", 2}}), "success");
  ASSERT_EQ(NodeTopoChecker(bars[0]).StrictConnectFrom({{"Const"}}), "success");
}
TEST_F(FrameSelectorUT, SelectMainRoot_Failed_BuilderIsNullptr) {
  ASSERT_EQ(FrameSelector::OnMainRoot(nullptr).size(), 0);
}
TEST_F(FrameSelectorUT, SelectMainRoot_Failed_ConnectFromSubgraph) {
  InitTestFrames();

  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data0, data1});
  ValueHolder::PushGraphFrame(foo1, "FooGraph");

  // on FooGraph
  auto c1 = ValueHolder::CreateConst("ConstData", 10, true);

  // on RootGraph
  auto bars = FrameSelector::OnMainRoot([&]() -> std::vector<ValueHolderPtr> {
    auto c0 = ValueHolder::CreateConst("ConstData", 10, true);
    auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {c1});
    return {bar1};
  });
  ASSERT_EQ(bars.size(), 1);
  ASSERT_EQ(bars[0], nullptr);
}
}  // namespace bg
}  // namespace gert
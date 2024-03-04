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
#include "exe_graph/lowering/generate_exe_graph.h"
#include <gtest/gtest.h>
#include "exe_graph/lowering/value_holder.h"
#include "checker/bg_test.h"
#include "checker/topo_checker.h"
namespace gert {
using namespace bg;
namespace {
std::vector<ValueHolderPtr> StubInferShape(const ge::NodePtr &node, const std::vector<ValueHolderPtr> &shapes) {
  return ValueHolder::CreateDataOutput("InferShape", shapes, 10);
}
std::vector<ValueHolderPtr> StubAllocOutputMemory(TensorPlacement placement, const ge::NodePtr &node,
                                                  const std::vector<ValueHolderPtr> &output_sizes,
                                                  LoweringGlobalData &global_data) {
  return ValueHolder::CreateDataOutput("AllocOutputMemory", output_sizes, output_sizes.size());
}
std::vector<ValueHolderPtr> StubCalcTensorSize(const ge::NodePtr &node,
                                               const std::vector<ValueHolderPtr> &output_shapes) {
  return ValueHolder::CreateDataOutput("CalcTensorSize", output_shapes, output_shapes.size());
}
}  // namespace
class GenerateExeGraphUT : public BgTest {
 protected:
  void SetUp() override {
    BgTest::SetUp();
    bg::GenerateExeGraph::AddBuilderImplement({nullptr, nullptr, nullptr});
  }
};
TEST_F(GenerateExeGraphUT, NoImpl_Failed_InferShape) {
  ASSERT_TRUE(bg::GenerateExeGraph::InferShape(nullptr, {bg::ValueHolder::CreateFeed(0)}).empty());
}
TEST_F(GenerateExeGraphUT, NoImpl_Failed_AllocOutputMemory) {
  LoweringGlobalData gd;
  ASSERT_TRUE(
      bg::GenerateExeGraph::AllocOutputMemory(kOnDeviceHbm, nullptr, {bg::ValueHolder::CreateFeed(0)}, gd).empty());
}
TEST_F(GenerateExeGraphUT, NoImpl_Failed_CalcTensorSize) {
  LoweringGlobalData gd;
  ASSERT_TRUE(bg::GenerateExeGraph::CalcTensorSize(nullptr, {bg::ValueHolder::CreateFeed(0)}).empty());
}
TEST_F(GenerateExeGraphUT, StubImpl_GraphCorrect_InferShape) {
  bg::GenerateExeGraph::AddBuilderImplement({StubInferShape, nullptr, nullptr});
  auto input_shape = bg::ValueHolder::CreateFeed(0);
  auto shapes = bg::GenerateExeGraph::InferShape(nullptr, {input_shape});
  ASSERT_EQ(shapes.size(), 10);
  ASSERT_EQ(shapes[0]->GetNode()->GetType(), "InferShape");
  ASSERT_EQ(NodeTopoChecker(shapes[0]).StrictConnectFrom({{input_shape}}), "success");
}
TEST_F(GenerateExeGraphUT, StubImpl_GraphCorrect_AllocOutputMemory) {
  bg::GenerateExeGraph::AddBuilderImplement({nullptr, StubAllocOutputMemory, nullptr});
  auto input_shape0 = bg::ValueHolder::CreateFeed(0);
  auto input_shape1 = bg::ValueHolder::CreateFeed(1);

  LoweringGlobalData gd;
  auto shapes = bg::GenerateExeGraph::AllocOutputMemory(kOnDeviceHbm, nullptr, {input_shape0, input_shape1}, gd);

  ASSERT_EQ(shapes.size(), 2);
  ASSERT_EQ(shapes[0]->GetNode()->GetType(), "AllocOutputMemory");
  ASSERT_EQ(NodeTopoChecker(shapes[0]).StrictConnectFrom({{input_shape0, input_shape1}}), "success");
}
TEST_F(GenerateExeGraphUT, StubImpl_GraphCorrect_CalcTensorSize) {
  bg::GenerateExeGraph::AddBuilderImplement({nullptr, nullptr, StubCalcTensorSize});
  auto input_shape0 = bg::ValueHolder::CreateFeed(0);
  auto input_shape1 = bg::ValueHolder::CreateFeed(1);

  LoweringGlobalData gd;
  auto shapes = bg::GenerateExeGraph::CalcTensorSize(nullptr, {input_shape0, input_shape1});

  ASSERT_EQ(shapes.size(), 2);
  ASSERT_EQ(shapes[0]->GetNode()->GetType(), "CalcTensorSize");
  ASSERT_EQ(NodeTopoChecker(shapes[0]).StrictConnectFrom({{input_shape0, input_shape1}}), "success");
}
}  // namespace gert
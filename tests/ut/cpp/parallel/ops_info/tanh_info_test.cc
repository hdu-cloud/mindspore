/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include <string>
#include <list>
#include <vector>
#include "common/common_test.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/ops_info/activation_info.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {

class TanhInfo;
using TanhInfoPtr = std::shared_ptr<TanhInfo>;
TanhInfoPtr tanh;

class TestTanhInfo : public UT::Common {
 public:
  TestTanhInfo() {}
  void SetUp();
  void TearDown() {}
};

void TestTanhInfo::SetUp() {
  RankList dev_list;

  for (int32_t i = 0; i < 130; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(128);
  stage_map.push_back(2);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");

  mindspore::HashMap<std::string, ValuePtr> attr;

  Shapes inputs_shape = {{2, 4, 8, 16}};
  Shapes outputs_shape = {{2, 4, 8, 16}};

  tanh = std::make_shared<TanhInfo>("tanh_info", inputs_shape, outputs_shape, attr);
}

TEST_F(TestTanhInfo, InferDevMatrixShape1) {
  Strategies inputs = {{2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  tanh->Init(strategy, nullptr);
  Shape dev_matrix_shape = tanh->dev_matrix_shape();

  Shape expect = {2, 4, 1, 16};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestTanhInfo, InferSliceShape1) {
  Strategies str = {{2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, str);

  tanh->Init(strategy, nullptr);
  std::vector<TensorInfo> inputs = tanh->inputs_tensor_info();
  std::vector<TensorInfo> outputs = tanh->outputs_tensor_info();

  Shape input_slice_shape_expect = {1, 1, 8, 1};
  Shape output_slice_shape_expect = {1, 1, 8, 1};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape input_slice_shape = input_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(input_slice_shape, input_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

TEST_F(TestTanhInfo, GetTensorLayout1) {
  Strategies str = {{2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, str);

  tanh->Init(strategy, nullptr);
  std::vector<TensorInfo> inputs = tanh->inputs_tensor_info();
  std::vector<TensorInfo> outputs = tanh->outputs_tensor_info();

  TensorMap input_expect = {3, 2, 1, 0};
  TensorMap output_expect = {3, 2, 1, 0};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Map input_tensor_map = input_tensor_info.tensor_layout().origin_tensor_map();
  Map output_tensor_map = output_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(input_tensor_map.array(), input_expect);
  ASSERT_EQ(output_tensor_map.array(), output_expect);
}

TEST_F(TestTanhInfo, GetForwardOp1) {
  Strategies inputs = {{2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  tanh->Init(strategy, nullptr);
  OperatorVector forward_op = tanh->forward_op();
  size_t size = forward_op.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestTanhInfo, GetMirrorOPs1) {
  Strategies inputs = {{2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  tanh->Init(strategy, nullptr);
  MirrorOps mirror_ops = tanh->mirror_ops();

  size_t size = mirror_ops.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestTanhInfo, CheckStrategy1) {
  // Success: {{2,4,1,16}}
  Strategies inputs = {{2, 2, 8, 16}, {2, 4, 16, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = tanh->Init(strategy, nullptr);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestTanhInfo, CheckStrategy2) {
  // Success: {{2,4,1,16}}
  Strategies inputs = {{2, 4, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = tanh->Init(strategy, nullptr);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestTanhInfo, CheckStrategy3) {
  // Success: {{2,4,1,16}}
  Strategies inputs = {{2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = tanh->Init(strategy, nullptr);
  ASSERT_EQ(ret, SUCCESS);
}

}  // namespace parallel
}  // namespace mindspore

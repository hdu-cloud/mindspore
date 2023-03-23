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

#include "register/node_converter_registry.h"
#include <gtest/gtest.h>

class NodeConverterRegistryUnittest : public testing::Test {};

namespace TestNodeConverterRegistry {
gert::LowerResult TestFunc(const ge::NodePtr &node, const gert::LowerInput &lower_input) {
  return {};
}
gert::LowerResult TestFunc2(const ge::NodePtr &node, const gert::LowerInput &lower_input) {
  return {};
}

TEST_F(NodeConverterRegistryUnittest, RegisterSuccess_DefaultPlacement) {
  EXPECT_EQ(gert::NodeConverterRegistry::GetInstance().FindNodeConverter("RegisterSuccess1"), nullptr);
  REGISTER_NODE_CONVERTER("RegisterSuccess1", TestFunc);
  EXPECT_EQ(gert::NodeConverterRegistry::GetInstance().FindNodeConverter("RegisterSuccess1"), TestFunc);
  auto reg_data1 = gert::NodeConverterRegistry::GetInstance().FindRegisterData("RegisterSuccess1");
  ASSERT_NE(reg_data1, nullptr);
  EXPECT_EQ(reg_data1->converter, TestFunc);
  EXPECT_EQ(reg_data1->require_placement, -1);
}

TEST_F(NodeConverterRegistryUnittest, RegisterSuccess_WithPlacement) {
  EXPECT_EQ(gert::NodeConverterRegistry::GetInstance().FindNodeConverter("RegisterSuccess2"), nullptr);
  REGISTER_NODE_CONVERTER_PLACEMENT("RegisterSuccess2", 10, TestFunc2);
  EXPECT_EQ(gert::NodeConverterRegistry::GetInstance().FindNodeConverter("RegisterSuccess2"), TestFunc2);
  auto reg_data1 = gert::NodeConverterRegistry::GetInstance().FindRegisterData("RegisterSuccess2");
  ASSERT_NE(reg_data1, nullptr);
  EXPECT_EQ(reg_data1->converter, TestFunc2);
  EXPECT_EQ(reg_data1->require_placement, 10);
}
}

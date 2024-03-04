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
#include <gtest/gtest.h>
#define private public
#include "runtime/kernel_run_context_builder.h"
#undef private
namespace gert {
class KernelRunContextBuilderUT : public testing::Test {};

TEST_F(KernelRunContextBuilderUT, SetBufferPoolOk) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test0", "test1");
  KernelRunContextBuilder builder;
  auto holder = builder.Build(op_desc);
  auto compute_node_info = reinterpret_cast<const ComputeNodeInfo *>(
      holder.context_->GetComputeNodeExtend());
  EXPECT_EQ(std::string(compute_node_info->GetNodeName()), "test0");
  EXPECT_EQ(std::string(compute_node_info->GetNodeType()), "test1");
}

TEST_F(KernelRunContextBuilderUT, SetInputsOutputsOk) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test0", "test1");
  KernelRunContextBuilder builder;
  gert::StorageShape shape1({1,2,3,4}, {1,2,3,4});
  gert::StorageShape shape2({2,2,3,4}, {2,2,3,4});
  gert::StorageShape shape3({3,2,3,4}, {3,2,3,4});
  auto holder = builder.Inputs({{&shape1, nullptr}, {&shape2, nullptr}}).Outputs({&shape3}).Build(op_desc);
  auto context = holder.context_;
  EXPECT_EQ(context->GetInputNum(), 2);
  EXPECT_EQ(context->GetOutputNum(), 1);
  EXPECT_TRUE(context->GetInputPointer<StorageShape>(0) == &shape1);
  EXPECT_TRUE(context->GetInputPointer<StorageShape>(1) == &shape2);
  EXPECT_TRUE(context->GetOutputPointer<StorageShape>(0) == &shape3);
}

TEST_F(KernelRunContextBuilderUT, SetInputsOutputsDataTypeOk) {
ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test0", "test1");
KernelRunContextBuilder builder;

ge::DataType in_datatype_1 = ge::DT_INT4;
ge::DataType in_datatype_2 = ge::DT_INT8;
ge::DataType out_datatype = ge::DT_INT8;
auto holder = builder.Inputs({{reinterpret_cast<void *>(in_datatype_1), nullptr}, {reinterpret_cast<void *>(in_datatype_2), nullptr}}).Outputs({reinterpret_cast<void *>(out_datatype)}).Build(op_desc);
auto context = holder.context_;
EXPECT_EQ(context->GetInputNum(), 2);
EXPECT_EQ(context->GetOutputNum(), 1);
EXPECT_TRUE(*context->GetInputPointer<ge::DataType>(0) == in_datatype_1);
EXPECT_TRUE(*context->GetInputPointer<ge::DataType>(1) == in_datatype_2);
EXPECT_TRUE(*context->GetOutputPointer<ge::DataType>(0) == out_datatype);
}

TEST_F(KernelRunContextBuilderUT, BuildContextHolderSuccessWhenOpLossAttrs) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test0", "test1");
  op_desc->AppendIrAttrName("attr1");
  KernelRunContextBuilder builder;
  auto holder = builder.Build(op_desc);
  ASSERT_NE(holder.context_holder_, nullptr);
  EXPECT_NE(holder.compute_node_extend_holder_, nullptr);
}
}  // namespace gert

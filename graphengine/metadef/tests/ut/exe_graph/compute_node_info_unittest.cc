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
#include "exe_graph/runtime/compute_node_info.h"
#include <gtest/gtest.h>
#include "faker/kernel_run_context_faker.h"
namespace gert {
class ComputeNodeInfoUT : public testing::Test {};
TEST_F(ComputeNodeInfoUT, GetInputFormatOk) {
  auto context_holder = KernelRunContextFaker()
                            .IrInputNum(2)
                            .NodeIoNum(2, 1)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
                            .Build();

  auto context = context_holder.GetContext<ExtendedKernelContext>();
  ASSERT_NE(context, nullptr);
  auto compute_node_info = context->GetComputeNodeInfo();
  ASSERT_NE(compute_node_info, nullptr);

  auto td = compute_node_info->GetInputTdInfo(0);
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetStorageFormat(), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(td->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td->GetFormat().GetStorageFormat(), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(td->GetFormat().GetOriginFormat(), ge::FORMAT_NCHW);

  td = compute_node_info->GetInputTdInfo(1);
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetStorageFormat(), ge::FORMAT_FRACTAL_Z);
  EXPECT_EQ(td->GetOriginFormat(), ge::FORMAT_HWCN);
  EXPECT_EQ(td->GetFormat().GetStorageFormat(), ge::FORMAT_FRACTAL_Z);
  EXPECT_EQ(td->GetFormat().GetOriginFormat(), ge::FORMAT_HWCN);

  EXPECT_EQ(compute_node_info->GetInputTdInfo(2), nullptr);
}
TEST_F(ComputeNodeInfoUT, GetOutputFormatOk) {
  auto context_holder = KernelRunContextFaker()
                            .IrInputNum(2)
                            .NodeIoNum(2, 1)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .Build();

  auto context = context_holder.GetContext<ExtendedKernelContext>();
  ASSERT_NE(context, nullptr);
  auto compute_node_info = context->GetComputeNodeInfo();
  ASSERT_NE(compute_node_info, nullptr);

  auto td = compute_node_info->GetOutputTdInfo(0);
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetStorageFormat(), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(td->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td->GetFormat().GetStorageFormat(), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(td->GetFormat().GetOriginFormat(), ge::FORMAT_NCHW);

  EXPECT_EQ(compute_node_info->GetOutputTdInfo(1), nullptr);
}
TEST_F(ComputeNodeInfoUT, GetNodeNameTypeOk) {
  auto context_holder = KernelRunContextFaker()
                            .IrInputNum(2)
                            .NodeIoNum(2, 1)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
                            .Build();

  auto context = context_holder.GetContext<ExtendedKernelContext>();
  ASSERT_NE(context, nullptr);
  auto compute_node_info = context->GetComputeNodeInfo();
  ASSERT_NE(compute_node_info, nullptr);

  EXPECT_STREQ(compute_node_info->GetNodeName(), "node");
  EXPECT_STREQ(compute_node_info->GetNodeType(), "node");
}
TEST_F(ComputeNodeInfoUT, GetInputInfoOk) {
  auto context_holder = KernelRunContextFaker()
                            .IrInputNum(2)
                            .NodeIoNum(2, 1)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
                            .Build();

  auto context = context_holder.GetContext<ExtendedKernelContext>();
  ASSERT_NE(context, nullptr);
  auto compute_node_info = context->GetComputeNodeInfo();
  ASSERT_NE(compute_node_info, nullptr);

  EXPECT_EQ(compute_node_info->GetIrInputsNum(), 2);
  EXPECT_EQ(compute_node_info->GetInputsNum(), 2);
  EXPECT_EQ(compute_node_info->GetOutputsNum(), 1);
}
TEST_F(ComputeNodeInfoUT, GetInputInstanceOk) {
  auto context_holder = KernelRunContextFaker()
                            .IrInputNum(2)
                            .NodeIoNum(2, 1)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
                            .Build();

  auto context = context_holder.GetContext<ExtendedKernelContext>();
  ASSERT_NE(context, nullptr);
  auto compute_node_info = context->GetComputeNodeInfo();
  ASSERT_NE(compute_node_info, nullptr);

  auto ins = compute_node_info->GetInputInstanceInfo(0);
  ASSERT_NE(ins, nullptr);
  EXPECT_EQ(ins->GetInstanceNum(), 1);
  EXPECT_EQ(ins->GetInstanceStart(), 0);

  ins = compute_node_info->GetInputInstanceInfo(1);
  ASSERT_NE(ins, nullptr);
  EXPECT_EQ(ins->GetInstanceNum(), 1);
  EXPECT_EQ(ins->GetInstanceStart(), 1);

  EXPECT_EQ(compute_node_info->GetInputInstanceInfo(2), nullptr);
}
TEST_F(ComputeNodeInfoUT, GetDynamicInputInstanceOk) {
  auto context_holder = KernelRunContextFaker()
                            .IrInstanceNum({2, 0, 1})
                            .NodeIoNum(3, 1)
                            .Build();

  auto context = context_holder.GetContext<ExtendedKernelContext>();
  ASSERT_NE(context, nullptr);
  auto compute_node_info = context->GetComputeNodeInfo();
  ASSERT_NE(compute_node_info, nullptr);

  auto ins = compute_node_info->GetInputInstanceInfo(0);
  ASSERT_NE(ins, nullptr);
  EXPECT_EQ(ins->GetInstanceNum(), 2);
  EXPECT_EQ(ins->GetInstanceStart(), 0);

  ins = compute_node_info->GetInputInstanceInfo(1);
  ASSERT_NE(ins, nullptr);
  EXPECT_EQ(ins->GetInstanceNum(), 0);
  EXPECT_EQ(ins->GetInstanceStart(), 2);

  ins = compute_node_info->GetInputInstanceInfo(2);
  ASSERT_NE(ins, nullptr);
  EXPECT_EQ(ins->GetInstanceNum(), 1);
  EXPECT_EQ(ins->GetInstanceStart(), 2);

  EXPECT_EQ(compute_node_info->GetInputInstanceInfo(3), nullptr);
}
TEST_F(ComputeNodeInfoUT, GetAttrsOk) {
  auto context_holder = KernelRunContextFaker()
                            .IrInstanceNum({2, 0, 1})
                            .NodeIoNum(3, 1)
                            .NodeAttrs({
                                {"i", ge::AnyValue::CreateFrom(static_cast<int64_t>(10))},
                                {"li", ge::AnyValue::CreateFrom(std::vector<int64_t>({10,20,30}))}
                            })
                            .Build();

  auto context = context_holder.GetContext<ExtendedKernelContext>();
  ASSERT_NE(context, nullptr);
  auto compute_node_info = context->GetComputeNodeInfo();
  ASSERT_NE(compute_node_info, nullptr);

  auto attrs = compute_node_info->GetAttrs();
  ASSERT_NE(attrs, nullptr);

  EXPECT_EQ(attrs->GetAttrNum(), 2);
  ASSERT_NE(attrs->GetAttrPointer<int64_t>(0), nullptr);
  EXPECT_EQ(*attrs->GetAttrPointer<int64_t>(0), 10);

  ASSERT_NE(attrs->GetAttrPointer<ContinuousVector>(1), nullptr);
  auto vec = attrs->GetAttrPointer<ContinuousVector>(1);
  EXPECT_EQ(reinterpret_cast<const int64_t *>(vec->GetData())[0], 10);
  EXPECT_EQ(reinterpret_cast<const int64_t *>(vec->GetData())[1], 20);
  EXPECT_EQ(reinterpret_cast<const int64_t *>(vec->GetData())[2], 30);
}

TEST_F(ComputeNodeInfoUT, GetAttrsEmptyAxes) {
auto context_holder = KernelRunContextFaker()
    .IrInstanceNum({2, 0, 1})
    .NodeIoNum(3, 1)
    .NodeAttrs({
                   {"i", ge::AnyValue::CreateFrom(static_cast<int64_t>(10))},
                   {"axes", ge::AnyValue::CreateFrom(std::vector<int64_t>({}))}
               })
    .Build();

auto context = context_holder.GetContext<ExtendedKernelContext>();
ASSERT_NE(context, nullptr);
auto compute_node_info = context->GetComputeNodeInfo();
ASSERT_NE(compute_node_info, nullptr);

auto attrs = compute_node_info->GetAttrs();
ASSERT_NE(attrs, nullptr);
EXPECT_EQ(attrs->GetAttrNum(), 2);
EXPECT_EQ(*attrs->GetAttrPointer<int64_t>(0), 10);
auto vec = attrs->GetAttrPointer<ContinuousVector>(1);
EXPECT_NE(vec, nullptr);
EXPECT_EQ(vec->GetSize(), 0);
}
}  // namespace gert
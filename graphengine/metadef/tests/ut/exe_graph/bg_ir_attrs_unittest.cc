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
#include "exe_graph/lowering/bg_kernel_context_extend.h"
#include <gtest/gtest.h>
#include <memory>
#include "graph/compute_graph.h"
#include "graph/utils/node_utils.h"
#include "exe_graph/runtime/context_extend.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/lowering/bg_ir_attrs.h"
#include "graph/debug/ge_attr_define.h"
#include "transformer/inc/expand_dimension.h"

namespace gert {
class BgIrAttrsUT : public testing::Test {};
// 构造tensorAttr，其shape小于size，测试AppendTensorAtrr函数能够正常内存拷贝
TEST_F(BgIrAttrsUT, ShapeSmallerThanSizeOfTensorAttr) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc ge_td;
  ge_td.SetOriginFormat(ge::FORMAT_NHWC);
  ge_td.SetFormat(ge::FORMAT_NHWC);
  ge_td.SetDataType(ge::DT_FLOAT16);
  ge_td.SetOriginShape(ge::GeShape({10, 10}));
  ge_td.SetShape(ge::GeShape({10, 10}));
  ge::GeTensor ge_tensor(ge_td);
  std::vector<uint16_t> fake_data(12 * 12);
  for (size_t i = 0; i < fake_data.size(); ++i) {
    fake_data[i] = static_cast<uint16_t>(i % std::numeric_limits<uint16_t>::max());
  }
  ge_tensor.SetData(reinterpret_cast<uint8_t *>(fake_data.data()), fake_data.size() * 2);
  ge::AttrUtils::SetTensor(op_desc, "a1", ge_tensor);
  op_desc->AppendIrAttrName("a1");

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  bg::BufferPool buffer_pool;
  auto ret = bg::CreateComputeNodeInfo(node, buffer_pool);
  ASSERT_NE(ret, nullptr);
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
  auto attrs = compute_node_info->GetAttrs();
  ASSERT_NE(attrs, nullptr);
  EXPECT_EQ(attrs->GetAttrNum(), 1);

  auto gert_tensor = attrs->GetAttrPointer<gert::Tensor>(0);
  EXPECT_EQ(attrs->GetTensor(0), gert_tensor);
  ASSERT_NE(gert_tensor, nullptr);
  EXPECT_EQ(gert_tensor->GetOriginShape(), gert::Shape({10, 10}));
  EXPECT_EQ(gert_tensor->GetStorageShape(), gert::Shape({10, 10}));
  EXPECT_EQ(gert_tensor->GetOriginFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(gert_tensor->GetStorageFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(gert_tensor->GetDataType(), ge::DT_FLOAT16);
  auto gert_tensor_ptr = gert_tensor->GetData<uint16_t>();
  EXPECT_NE(gert_tensor_ptr, nullptr);
  for (size_t i = 0; i < 10 * 10; ++i) {
    EXPECT_EQ(gert_tensor_ptr[i], static_cast<uint16_t>(i % std::numeric_limits<uint16_t>::max()));
  }
}
TEST_F(BgIrAttrsUT, CreateDataTypeAttrBuffer) {
  auto op_desc = std::make_shared<ge::OpDesc>("foo", "Foo");
  op_desc->AppendIrAttrName("dtype");
  EXPECT_EQ(op_desc->GetIrAttrNames().size(), 1U);
  EXPECT_EQ(op_desc->GetIrAttrNames().at(0), "dtype");
  ge::AttrUtils::SetDataType(op_desc, "dtype", ge::DT_INT32);
  auto node = ge::NodeUtils::CreatNodeWithoutGraph(op_desc);
  size_t attr_size;
  auto attr_buffer = bg::CreateAttrBuffer(node, attr_size);
  auto base = reinterpret_cast<size_t*>(attr_buffer.get());
  EXPECT_EQ(base[0], 1U);
  EXPECT_EQ(base[1], 2 * sizeof(size_t));
  EXPECT_EQ(*reinterpret_cast<ge::DataType*>(&base[2]), ge::DT_INT32);
}

TEST_F(BgIrAttrsUT, CreateAttrBufferSuccessOpLossAttr) {
  auto op_desc = std::make_shared<ge::OpDesc>("foo", "Foo");
  op_desc->AppendIrAttrName("dtype");
  EXPECT_EQ(op_desc->GetIrAttrNames().size(), 1U);
  EXPECT_EQ(op_desc->GetIrAttrNames().at(0), "dtype");
  // ge::AttrUtils::SetDataType(op_desc, "dtype", ge::DT_INT32);
  auto node = ge::NodeUtils::CreatNodeWithoutGraph(op_desc);
  size_t attr_size;
  auto attr_buffer = bg::CreateAttrBuffer(node, attr_size);
  EXPECT_EQ(attr_size, 8);
  auto base = reinterpret_cast<size_t*>(attr_buffer.get());
  EXPECT_EQ(base[0], 0U);
  EXPECT_EQ(base[1], 0U);
}
}  // namespace gert

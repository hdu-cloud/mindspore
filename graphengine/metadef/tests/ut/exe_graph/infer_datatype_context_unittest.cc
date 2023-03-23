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
#include "exe_graph/runtime/infer_datatype_context.h"
#include <gtest/gtest.h>
#include "faker/kernel_run_context_faker.h"
#include "exe_graph/runtime/storage_shape.h"
namespace gert {
class InferDataTypeContextUT : public testing::Test {};
TEST_F(InferDataTypeContextUT, GetInputDataTypeOk) {
  ge::DataType in_datatype1 = ge::DT_INT8;
  ge::DataType in_datatype2 = ge::DT_INT8;
  ge::DataType out_datatype = ge::DT_FLOAT16;
  auto context_holder = InferDataTypeContextFaker()
                            .IrInputNum(2)
                            .NodeIoNum(2, 1)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
                            .InputDataTypes({&in_datatype1, &in_datatype2})
                            .OutputDataTypes({&out_datatype})
                            .Build();
  auto context = context_holder.GetContext<InferDataTypeContext>();
  ASSERT_NE(context, nullptr);

  EXPECT_EQ(context->GetInputDataType(0), in_datatype1);
  EXPECT_EQ(context->GetInputDataType(1), in_datatype2);
}

TEST_F(InferDataTypeContextUT, GetDynamicInputDataTypeOk) {
  ge::DataType in_datatype1 = ge::DT_INT8;
  ge::DataType in_datatype2 = ge::DT_INT4;
  ge::DataType in_datatype3 = ge::DT_INT8;
  ge::DataType in_datatype4 = ge::DT_INT4;
  ge::DataType out_datatype = ge::DT_FLOAT16;
  auto context_holder = InferDataTypeContextFaker()
                            .IrInstanceNum({1, 2, 0, 1})
                            .NodeIoNum(4, 1)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                            .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                            .InputDataTypes({&in_datatype1, &in_datatype2, &in_datatype3, &in_datatype4})
                            .OutputDataTypes({&out_datatype})
                            .Build();
  auto context = context_holder.GetContext<InferDataTypeContext>();
  ASSERT_NE(context, nullptr);

  EXPECT_EQ(context->GetOptionalInputDataType(0), in_datatype1);
  EXPECT_EQ(context->GetDynamicInputDataType(1, 0), in_datatype2);
  EXPECT_EQ(context->GetDynamicInputDataType(1, 1), in_datatype3);

  EXPECT_EQ(context->GetOptionalInputDataType(2), ge::DataType::DT_UNDEFINED);

  EXPECT_EQ(context->GetOptionalInputDataType(3), in_datatype4);
}

TEST_F(InferDataTypeContextUT, GetOutDataTypeOk) {
  ge::DataType in_datatype1 = ge::DT_INT4;
  ge::DataType in_datatype2 = ge::DT_INT8;
  ge::DataType out_datatype = ge::DT_FLOAT16;
  auto context_holder = InferDataTypeContextFaker()
                            .IrInputNum(2)
                            .NodeIoNum(2, 1)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
                            .InputDataTypes({&in_datatype1, &in_datatype2})
                            .OutputDataTypes({&out_datatype})
                            .Build();
  auto context = context_holder.GetContext<InferDataTypeContext>();
  ASSERT_NE(context, nullptr);

  EXPECT_EQ(context->GetOutputDataType(0), out_datatype);

  EXPECT_EQ(context->GetOutputDataType(1), ge::DataType::DT_UNDEFINED);
}

TEST_F(InferDataTypeContextUT, SetOutputDataTypeOk) {
  ge::DataType in_datatype1 = ge::DT_INT4;
  ge::DataType in_datatype2 = ge::DT_INT8;
  ge::DataType origin_out_datatype = ge::DT_FLOAT16;
  auto context_holder = InferDataTypeContextFaker()
      .IrInputNum(2)
      .NodeIoNum(2, 1)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
      .InputDataTypes({&in_datatype1, &in_datatype2})
      .OutputDataTypes({&origin_out_datatype})
      .Build();
  auto context = context_holder.GetContext<InferDataTypeContext>();
  ASSERT_NE(context, nullptr);

  EXPECT_EQ(context->GetOutputDataType(0), origin_out_datatype);
  EXPECT_EQ(context->SetOutputDataType(0, ge::DT_INT32), ge::GRAPH_SUCCESS);
  EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}
}  // namespace gert
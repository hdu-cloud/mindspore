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
#include "exe_graph/runtime/tiling_context.h"
#include "faker/kernel_run_context_faker.h"
#undef private

namespace gert {
class TilingContextUT : public testing::Test {};
namespace {
struct TestTilingData {
  int64_t a;
};
struct TestCompileInfo {
  int64_t a;
  int64_t b;
  std::vector<int64_t> c;
};
}  // namespace
TEST_F(TilingContextUT, GetCompileInfoOk) {
  gert::StorageShape in_shape = {{1, 16, 256}, {1, 16, 256}};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};

  // tiling data
  TestCompileInfo compile_info_holder = {10, 200, {10, 20, 30}};
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .InputShapes({&in_shape})
                    .OutputShapes({&out_shape})
                    .CompileInfo(&compile_info_holder)
                    .TilingData(param.get())
                    .Build();

  auto context = holder.GetContext<TilingContext>();
  ASSERT_NE(context, nullptr);
  auto compile_info = reinterpret_cast<const TestCompileInfo *>(context->GetCompileInfo());
  ASSERT_NE(compile_info, nullptr);
  EXPECT_EQ(compile_info->a, 10);
  EXPECT_EQ(compile_info->b, 200);
  EXPECT_EQ(compile_info->c, compile_info_holder.c);
}

TEST_F(TilingContextUT, GetInputShapeOk) {
  gert::StorageShape in_shape = {{1, 16, 256}, {1, 16, 256}};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};

  // tiling data
  TestCompileInfo compile_info_holder = {10, 200, {10, 20, 30}};
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .InputShapes({&in_shape})
                    .OutputShapes({&out_shape})
                    .CompileInfo(&compile_info_holder)
                    .TilingData(param.get())
                    .Build();

  auto context = holder.GetContext<TilingContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_NE(context->GetInputShape(0), nullptr);
  EXPECT_EQ(context->GetInputShape(0)->GetOriginShape(), in_shape.GetOriginShape());
  EXPECT_EQ(context->GetInputShape(0)->GetStorageShape(), in_shape.GetStorageShape());
  ASSERT_EQ(context->GetInputShape(1), nullptr);
}

TEST_F(TilingContextUT, GetDynamicInputShapeOk) {
  gert::StorageShape in_shape0 = {{1, 16, 256}, {1, 16, 256}};
  gert::StorageShape in_shape1 = {{2, 16, 256}, {1, 16, 256}};
  gert::StorageShape in_shape2 = {{3, 16, 256}, {1, 16, 256}};
  gert::StorageShape in_shape3 = {{4, 16, 256}, {1, 16, 256}};
  gert::StorageShape in_shape4 = {{5, 16, 256}, {1, 16, 256}};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};

  TestCompileInfo compile_info_holder = {10, 200, {10, 20, 30}};
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(5, 1)
                    .IrInstanceNum({1,2,1,0,1})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .InputShapes({&in_shape0, &in_shape1, &in_shape2, &in_shape3, &in_shape4})
                    .OutputShapes({&out_shape})
                    .CompileInfo(&compile_info_holder)
                    .TilingData(param.get())
                    .Build();

  auto context = holder.GetContext<TilingContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetDynamicInputShape(0, 0), nullptr);
  EXPECT_EQ(*context->GetDynamicInputShape(0, 0), in_shape0);
  EXPECT_EQ(context->GetDynamicInputShape(0, 1), nullptr);

  ASSERT_NE(context->GetDynamicInputShape(1, 0), nullptr);
  EXPECT_EQ(*context->GetDynamicInputShape(1, 0), in_shape1);
  ASSERT_NE(context->GetDynamicInputShape(1, 1), nullptr);
  EXPECT_EQ(*context->GetDynamicInputShape(1, 1), in_shape2);
  EXPECT_EQ(context->GetDynamicInputShape(1, 2), nullptr);

  ASSERT_NE(context->GetDynamicInputShape(2, 0), nullptr);
  EXPECT_EQ(*context->GetDynamicInputShape(2, 0), in_shape3);
  EXPECT_EQ(context->GetDynamicInputShape(2, 1), nullptr);

  EXPECT_EQ(context->GetOptionalInputShape(3), nullptr);

  ASSERT_NE(context->GetOptionalInputShape(4), nullptr);
  EXPECT_EQ(*context->GetOptionalInputShape(4), in_shape4);
}

TEST_F(TilingContextUT, GetDynamicInputDescOk) {
  gert::StorageShape in_shape0 = {{1, 16, 256}, {1, 16, 256}};
  gert::StorageShape in_shape1 = {{2, 16, 256}, {1, 16, 256}};
  gert::StorageShape in_shape2 = {{3, 16, 256}, {1, 16, 256}};
  gert::StorageShape in_shape3 = {{4, 16, 256}, {1, 16, 256}};
  gert::StorageShape in_shape4 = {{5, 16, 256}, {1, 16, 256}};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};

  TestCompileInfo compile_info_holder = {10, 200, {10, 20, 30}};
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
      .NodeIoNum(5, 1)
      .IrInstanceNum({1,2,1,0,1})
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_Z, ge::FORMAT_ND)
      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_C1HWNC0, ge::FORMAT_ND)
      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS, ge::FORMAT_ND)
      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
      .InputShapes({&in_shape0, &in_shape1, &in_shape2, &in_shape3, &in_shape4})
      .OutputShapes({&out_shape})
      .CompileInfo(&compile_info_holder)
      .TilingData(param.get())
      .Build();

  auto context = holder.GetContext<TilingContext>();
  ASSERT_NE(context, nullptr);

  auto tensor_desc1 = context->GetDynamicInputDesc(1, 0);
  ASSERT_NE(tensor_desc1, nullptr);
  EXPECT_EQ(tensor_desc1->GetOriginFormat(), ge::FORMAT_FRACTAL_Z);

  auto tensor_desc2 = context->GetDynamicInputDesc(1, 1);
  ASSERT_NE(tensor_desc2, nullptr);
  EXPECT_EQ(tensor_desc2->GetOriginFormat(), ge::FORMAT_C1HWNC0);

  auto tensor_desc3 = context->GetDynamicInputDesc(1, 2);
  ASSERT_EQ(tensor_desc3, nullptr);

  EXPECT_EQ(context->GetOptionalInputDesc(3), nullptr);

  ASSERT_NE(context->GetOptionalInputDesc(4), nullptr);
  EXPECT_EQ(context->GetOptionalInputDesc(4)->GetOriginFormat(), ge::FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS);
}

TEST_F(TilingContextUT, GetOutputShapeOk) {
  gert::StorageShape in_shape0 = {{1, 16, 256}, {1, 16, 256}};
  gert::StorageShape in_shape1 = {{2, 16, 256}, {1, 16, 256}};
  gert::StorageShape in_shape2 = {{3, 16, 256}, {1, 16, 256}};
  gert::StorageShape in_shape3 = {{4, 16, 256}, {1, 16, 256}};
  gert::StorageShape in_shape4 = {{5, 16, 256}, {1, 16, 256}};
  gert::StorageShape out_shape = {{6, 16, 256}, {1, 16, 1, 16, 16}};

  TestCompileInfo compile_info_holder = {10, 200, {10, 20, 30}};
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(5, 1)
                    .IrInstanceNum({1,2,1,0,1})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .InputShapes({&in_shape0, &in_shape1, &in_shape2, &in_shape3, &in_shape4})
                    .OutputShapes({&out_shape})
                    .CompileInfo(&compile_info_holder)
                    .TilingData(param.get())
                    .Build();

  auto context = holder.GetContext<TilingContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetOutputShape(0), nullptr);
  EXPECT_EQ(*context->GetOutputShape(0), out_shape);
  ASSERT_EQ(context->GetOutputShape(1), nullptr);
}
TEST_F(TilingContextUT, GetInputTensorOk) {
  gert::Tensor in_tensor = {{{1, 16, 256}, {1, 16, 256}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            kOnDeviceHbm,                                // placement
                            ge::DT_FLOAT16,                              // data type
                            (void *) 0xabc};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};

  // tiling data
  TestCompileInfo compile_info_holder = {10, 200, {10, 20, 30}};
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .InputShapes({reinterpret_cast<gert::StorageShape*>(&in_tensor)})
                    .OutputShapes({&out_shape})
                    .CompileInfo(&compile_info_holder)
                    .TilingData(param.get())
                    .Build();

  auto context = holder.GetContext<TilingContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_NE(context->GetInputTensor(0), nullptr);
  EXPECT_EQ(context->GetInputTensor(0)->GetOriginShape(), in_tensor.GetOriginShape());
  EXPECT_EQ(context->GetInputTensor(0)->GetStorageShape(), in_tensor.GetStorageShape());
  EXPECT_EQ(context->GetInputTensor(0)->GetOriginFormat(), in_tensor.GetOriginFormat());
  EXPECT_EQ(context->GetInputTensor(0)->GetStorageFormat(), in_tensor.GetStorageFormat());
  EXPECT_EQ(context->GetInputTensor(0)->GetExpandDimsType(), in_tensor.GetExpandDimsType());
  EXPECT_EQ(context->GetInputTensor(0)->GetDataType(), in_tensor.GetDataType());
  EXPECT_EQ(context->GetInputTensor(0)->GetAddr(), in_tensor.GetAddr());
}

TEST_F(TilingContextUT, GetDynamicInputTensorOk) {
  gert::Tensor in_tensor = {{{1, 16}, {1, 16}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            kOnDeviceHbm,                                // placement
                            ge::DT_FLOAT16,                              // data type
                            (void *) 0x234};
  gert::Tensor in_tensor1 = {{{1, 16}, {1, 16}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            kOnDeviceHbm,                                // placement
                            ge::DT_FLOAT16,                              // data type
                            (void *) 0x123};
  gert::Tensor in_tensor2 = {{{1, 16}, {1, 16}},                // shape
                             {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                             kOnDeviceHbm,                                // placement
                             ge::DT_FLOAT16,                              // data type
                             (void *) 0x456};
  gert::StorageShape out_shape = {{1, 16}, {1, 16}};

  // tiling data
  TestCompileInfo compile_info_holder = {10, 200, {10, 20, 30}};
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
      .NodeIoNum(3, 1)
      .IrInstanceNum({1, 0, 2})
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
      .InputShapes({reinterpret_cast<gert::StorageShape*>(&in_tensor),
                   reinterpret_cast<gert::StorageShape*>(&in_tensor1),
                   reinterpret_cast<gert::StorageShape*>(&in_tensor2)})
      .OutputShapes({&out_shape})
      .CompileInfo(&compile_info_holder)
      .TilingData(param.get())
      .Build();

  auto context = holder.GetContext<TilingContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetInputTensor(0), nullptr);
  EXPECT_EQ(context->GetInputTensor(0)->GetOriginShape(), in_tensor.GetOriginShape());
  EXPECT_EQ(context->GetInputTensor(0)->GetStorageShape(), in_tensor.GetStorageShape());
  EXPECT_EQ(context->GetInputTensor(0)->GetOriginFormat(), in_tensor.GetOriginFormat());
  EXPECT_EQ(context->GetInputTensor(0)->GetStorageFormat(), in_tensor.GetStorageFormat());
  EXPECT_EQ(context->GetInputTensor(0)->GetExpandDimsType(), in_tensor.GetExpandDimsType());
  EXPECT_EQ(context->GetInputTensor(0)->GetDataType(), in_tensor.GetDataType());
  EXPECT_EQ(context->GetInputTensor(0)->GetAddr(), in_tensor.GetAddr());

  ASSERT_EQ(context->GetOptionalInputTensor(1), nullptr);

  ASSERT_NE(context->GetDynamicInputTensor(2, 0), nullptr);
  EXPECT_EQ(context->GetDynamicInputTensor(2, 0)->GetOriginShape(), in_tensor1.GetOriginShape());
  EXPECT_EQ(context->GetDynamicInputTensor(2, 0)->GetStorageShape(), in_tensor1.GetStorageShape());
  EXPECT_EQ(context->GetDynamicInputTensor(2, 0)->GetOriginFormat(), in_tensor1.GetOriginFormat());
  EXPECT_EQ(context->GetDynamicInputTensor(2, 0)->GetStorageFormat(), in_tensor1.GetStorageFormat());
  EXPECT_EQ(context->GetDynamicInputTensor(2, 0)->GetExpandDimsType(), in_tensor1.GetExpandDimsType());
  EXPECT_EQ(context->GetDynamicInputTensor(2, 0)->GetDataType(), in_tensor1.GetDataType());
  EXPECT_EQ(context->GetDynamicInputTensor(2, 0)->GetAddr(), in_tensor1.GetAddr());

  ASSERT_NE(context->GetDynamicInputTensor(2, 1), nullptr);
  EXPECT_EQ(context->GetDynamicInputTensor(2, 1)->GetOriginShape(), in_tensor2.GetOriginShape());
  EXPECT_EQ(context->GetDynamicInputTensor(2, 1)->GetStorageShape(), in_tensor2.GetStorageShape());
  EXPECT_EQ(context->GetDynamicInputTensor(2, 1)->GetOriginFormat(), in_tensor2.GetOriginFormat());
  EXPECT_EQ(context->GetDynamicInputTensor(2, 1)->GetStorageFormat(), in_tensor2.GetStorageFormat());
  EXPECT_EQ(context->GetDynamicInputTensor(2, 1)->GetExpandDimsType(), in_tensor2.GetExpandDimsType());
  EXPECT_EQ(context->GetDynamicInputTensor(2, 1)->GetDataType(), in_tensor2.GetDataType());
  EXPECT_EQ(context->GetDynamicInputTensor(2, 1)->GetAddr(), in_tensor2.GetAddr());
}

TEST_F(TilingContextUT, SetTypedTilingDataOk) {
  gert::StorageShape in_shape = {{1, 16, 256}, {1, 16, 256}};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};

  // tiling data
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .InputShapes({&in_shape})
                    .OutputShapes({&out_shape})
                    .TilingData(param.get())
                    .Build();

  auto context = holder.GetContext<TilingContext>();
  auto tiling_data = context->GetTilingData<TestTilingData>();
  ASSERT_NE(tiling_data, nullptr);
  tiling_data->a = 10;
  auto root_tiling_data = reinterpret_cast<TilingData *>(param.get());

  EXPECT_EQ(root_tiling_data->GetDataSize(), sizeof(TestTilingData));
  EXPECT_EQ(root_tiling_data->GetData(), tiling_data);
}

TEST_F(TilingContextUT, SetTypedTilingDataOutOfBounds) {
  gert::StorageShape in_shape = {{1, 16, 256}, {1, 16, 256}};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};

  auto param = gert::TilingData::CreateCap(4);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .InputShapes({&in_shape})
                    .OutputShapes({&out_shape})
                    .TilingData(param.get())
                    .Build();

  auto context = holder.GetContext<TilingContext>();
  auto tiling_data = context->GetTilingData<TestTilingData>();
  EXPECT_EQ(tiling_data, nullptr);
}
TEST_F(TilingContextUT, SetAppendTilingDataOk) {
  gert::StorageShape in_shape = {{1, 16, 256}, {1, 16, 256}};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};

  // tiling data
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .InputShapes({&in_shape})
                    .OutputShapes({&out_shape})
                    .TilingData(param.get())
                    .Build();

  auto context = holder.GetContext<TilingContext>();

  // 算子tiling中可以用如下操作append
  auto tiling_data = context->GetRawTilingData();
  ASSERT_NE(tiling_data, nullptr);

  tiling_data->Append(static_cast<int64_t>(10));
  tiling_data->Append(static_cast<int64_t>(20));
  tiling_data->Append(static_cast<int64_t>(30));
  tiling_data->Append(static_cast<int32_t>(40));
  tiling_data->Append(static_cast<int16_t>(50));
  tiling_data->Append(static_cast<int8_t>(60));

  EXPECT_EQ(context->GetRawTilingData()->GetDataSize(), 31);  // 3 * 8 + 4 + 2 + 1
}

TEST_F(TilingContextUT, SetTilingKeyOk) {
  gert::StorageShape in_shape = {{1, 16, 256}, {1, 16, 256}};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};

  // tiling data
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .InputShapes({&in_shape})
                    .OutputShapes({&out_shape})
                    .TilingData(param.get())
                    .Build();

  auto context = holder.GetContext<TilingContext>();

  context->SetTilingKey(20);
  EXPECT_EQ(context->GetTilingKey(), 20);
  EXPECT_EQ(
      *reinterpret_cast<uint64_t *>(
          &(holder.holder.value_holder_[holder.kernel_input_num + TilingContext::kOutputTilingKey].any_value_.data)),
      20);
}
TEST_F(TilingContextUT, SetBlockDimOk) {
  gert::StorageShape in_shape = {{1, 16, 256}, {1, 16, 256}};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};

  // tiling data
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .InputShapes({&in_shape})
                    .OutputShapes({&out_shape})
                    .TilingData(param.get())
                    .Build();

  auto context = holder.GetContext<TilingContext>();

  context->SetBlockDim(10);
  EXPECT_EQ(context->GetBlockDim(), 10);
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(&(
                holder.holder.value_holder_[holder.kernel_input_num + TilingContext::kOutputBlockDim].any_value_.data)),
            10);
}

TEST_F(TilingContextUT, SetNeedAtomicOk) {
  gert::StorageShape in_shape = {{1, 16, 256}, {1, 16, 256}};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};

  // tiling data
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .InputShapes({&in_shape})
                    .OutputShapes({&out_shape})
                    .TilingData(param.get())
                    .Build();

  auto context = holder.GetContext<TilingContext>();

  context->SetNeedAtomic(true);
  EXPECT_TRUE(context->NeedAtomic());
  EXPECT_TRUE(*reinterpret_cast<bool *>(
      &(holder.holder.value_holder_[holder.kernel_input_num + TilingContext::kOutputAtomicCleanFlag].any_value_.data)));

  context->SetNeedAtomic(false);
  EXPECT_FALSE(context->NeedAtomic());
  EXPECT_FALSE(*reinterpret_cast<bool *>(
      &(holder.holder.value_holder_[holder.kernel_input_num + TilingContext::kOutputAtomicCleanFlag].any_value_.data)));
}

TEST_F(TilingContextUT, SetWorkspaceSizesOk) {
  gert::StorageShape in_shape = {{1, 16, 256}, {1, 16, 256}};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};

  auto workspace_size_holder = ContinuousVector::Create<size_t>(8);
  auto ws_size = reinterpret_cast<ContinuousVector *>(workspace_size_holder.get());

  // tiling data
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .InputShapes({&in_shape})
                    .OutputShapes({&out_shape})
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  auto context = holder.GetContext<TilingContext>();

  auto ws = context->GetWorkspaceSizes(1);
  ASSERT_NE(ws, nullptr);
  ws[0] = 10240;
  EXPECT_EQ(ws_size->GetSize(), 1);
  EXPECT_EQ(reinterpret_cast<const size_t *>(ws_size->GetData())[0], 10240);
}

TEST_F(TilingContextUT, SetWorkspaceSizesOutOfBounds) {
  gert::StorageShape in_shape = {{1, 16, 256}, {1, 16, 256}};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};

  auto workspace_size_holder = ContinuousVector::Create<size_t>(0);
  auto ws_size = reinterpret_cast<ContinuousVector *>(workspace_size_holder.get());

  // tiling data
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .InputShapes({&in_shape})
                    .OutputShapes({&out_shape})
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  auto context = holder.GetContext<TilingContext>();

  auto ws = context->GetWorkspaceSizes(9);
  EXPECT_EQ(ws, nullptr);
}

TEST_F(TilingContextUT, SetTilingCondOk) {
  gert::StorageShape in_shape = {{1, 16, 256}, {1, 16, 256}};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};

  // tiling data
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .InputShapes({&in_shape})
                    .OutputShapes({&out_shape})
                    .TilingData(param.get())
                    .Build();

  auto context = holder.GetContext<TilingContext>();
  context->SetTilingCond(10);
  EXPECT_EQ(context->GetTilingCond(), 10);
}
}  // namespace gert
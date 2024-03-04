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
#include "exe_graph/runtime/tiling_data.h"
#include "common/util/tiling_utils.h"
#include "faker/kernel_run_context_faker.h"
#include <gtest/gtest.h>
namespace gert {
class TilingDataUT : public testing::Test {};
namespace {
struct TestData {
  int64_t a;
  int32_t b;
  int16_t c;
  int16_t d;
};

FakeKernelContextHolder BuildTestContext() {
  auto holder = gert::KernelRunContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .NodeAttrs({{"int", ge::AnyValue::CreateFrom<int64_t>(0x7fffffffUL)},
                                {"str", ge::AnyValue::CreateFrom<std::string>("Hello!")},
                                {"bool", ge::AnyValue::CreateFrom<bool>(true)},
                                {"float", ge::AnyValue::CreateFrom<float>(10.101)},
                                {"list_int", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 3})},
                                {"list_int2", ge::AnyValue::CreateFrom<std::vector<int64_t>>({4, 5, 6})},
                                {"list_float", ge::AnyValue::CreateFrom<std::vector<float>>({1.2, 3.4, 4.5})}})
                    .Build();
  return holder;
}

}  // namespace
TEST_F(TilingDataUT, AppendSameTypesOk) {
  auto data = TilingData::CreateCap(2048);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  ASSERT_NE(tiling_data, nullptr);
  std::vector<int64_t> expect_vec;
  for (int64_t i = 0; i < 10; ++i) {
    tiling_data->Append(i);
    expect_vec.push_back(i);
  }
  ASSERT_EQ(tiling_data->GetDataSize(), 80);
  EXPECT_EQ(memcmp(tiling_data->GetData(), expect_vec.data(), tiling_data->GetDataSize()), 0);
}
TEST_F(TilingDataUT, AppendStructOk) {
  auto data = TilingData::CreateCap(2048);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  ASSERT_NE(tiling_data, nullptr);
  TestData td{.a = 1024, .b = 512, .c = 256, .d = 128};
  tiling_data->Append(td);
  ASSERT_EQ(tiling_data->GetDataSize(), sizeof(td));
  EXPECT_EQ(memcmp(tiling_data->GetData(), &td, tiling_data->GetDataSize()), 0);
}

TEST_F(TilingDataUT, AppendDifferentTypes) {
  auto data = TilingData::CreateCap(2048);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  ASSERT_NE(tiling_data, nullptr);
  std::vector<int64_t> expect_vec1;
  for (int64_t i = 0; i < 10; ++i) {
    tiling_data->Append(i);
    expect_vec1.push_back(i);
  }

  std::vector<int32_t> expect_vec2;
  for (int32_t i = 0; i < 3; ++i) {
    tiling_data->Append(i);
    expect_vec2.push_back(i);
  }

  TestData td{.a = 1024, .b = 512, .c = 256, .d = 128};
  tiling_data->Append(td);

  ASSERT_EQ(tiling_data->GetDataSize(), 80 + 12 + sizeof(TestData));
  EXPECT_EQ(memcmp(tiling_data->GetData(), expect_vec1.data(), 80), 0);
  EXPECT_EQ(memcmp(reinterpret_cast<uint8_t *>(tiling_data->GetData()) + 80, expect_vec2.data(), 12), 0);
  EXPECT_EQ(memcmp(reinterpret_cast<uint8_t *>(tiling_data->GetData()) + 92, &td, sizeof(td)), 0);
}

TEST_F(TilingDataUT, AppendOutOfBounds) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  ASSERT_NE(tiling_data, nullptr);
  std::vector<int64_t> expect_vec1;
  for (int64_t i = 0; i < 10; ++i) {
    tiling_data->Append(i);
    expect_vec1.push_back(i);
  }

  std::vector<int64_t> expect_vec2;
  for (int64_t i = 0; i < 2; ++i) {
    tiling_data->Append(i);
    expect_vec2.push_back(i);
  }
  EXPECT_NE(tiling_data->Append(static_cast<int64_t>(3)), ge::GRAPH_SUCCESS);

  ASSERT_EQ(tiling_data->GetDataSize(), 16);
  EXPECT_EQ(memcmp(tiling_data->GetData(), expect_vec1.data(), 16), 0);
}

TEST_F(TilingDataUT, AppendAttrInt32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 0, AttrDataType::kInt64, AttrDataType::kInt32),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<int32_t *>(tiling_data->GetData()), 0x7fffffff);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int32_t));
}

TEST_F(TilingDataUT, AppendAttrStrOk) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 1, AttrDataType::kString, AttrDataType::kString),
            ge::GRAPH_SUCCESS);
  std::string s1(reinterpret_cast<char *>(tiling_data->GetData()),
                 reinterpret_cast<char *>(tiling_data->GetData()) + 6);
  EXPECT_STREQ(s1.c_str(), "Hello!");
  EXPECT_EQ(tiling_data->GetDataSize(), 6);
  tiling_data->SetDataSize(0);
}

TEST_F(TilingDataUT, AppendAttrBoolOk) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 2, AttrDataType::kBool, AttrDataType::kBool),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<bool *>(tiling_data->GetData()), true);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(bool));
}

TEST_F(TilingDataUT, AppendAttrFloatOk) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 3, AttrDataType::kFloat32, AttrDataType::kFloat32),
            ge::GRAPH_SUCCESS);
  EXPECT_FLOAT_EQ(*reinterpret_cast<float *>(tiling_data->GetData()), 10.101);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(float));
}

TEST_F(TilingDataUT, AppendAttrList32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(
      tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 4, AttrDataType::kListInt64, AttrDataType::kListInt32),
      ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int32_t) * 3);
  auto ele = reinterpret_cast<int32_t *>(tiling_data->GetData());
  std::vector<int32_t> expect_data{1, 2, 3};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele[i], expect_data[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrInt64ToUint32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 0, AttrDataType::kInt64, AttrDataType::kUint32),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(tiling_data->GetData()), 0x7fffffff);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint32_t));
}

TEST_F(TilingDataUT, AppendAttrFloat32ToFloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 3, AttrDataType::kFloat32, AttrDataType::kFloat16),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint16_t *>(tiling_data->GetData()), optiling::FloatToUint16(10.101));
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t));
}

TEST_F(TilingDataUT, AppendAttrFloat32ToInt32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 3, AttrDataType::kFloat32, AttrDataType::kInt32),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(*reinterpret_cast<uint16_t *>(tiling_data->GetData()), 10);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(int32_t));
}

TEST_F(TilingDataUT, AppendAttrListInt64ToListUint32Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(
      tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 4, AttrDataType::kListInt64, AttrDataType::kListUint32),
      ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint32_t) * 3);
  auto ele1 = reinterpret_cast<uint32_t *>(tiling_data->GetData());
  std::vector<int32_t> expect_data1{1, 2, 3};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele1[i], expect_data1[i]);
  }
}

TEST_F(TilingDataUT, AppendAttrListFloat32ToListFloat16Ok) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 6, AttrDataType::kListFloat32,
                                                AttrDataType::kListFloat16),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_data->GetDataSize(), sizeof(uint16_t) * 3);
  auto ele1 = reinterpret_cast<uint16_t *>(tiling_data->GetData());
  std::vector<float> expet_data = {1.2, 3.4, 4.5};
  for (size_t i = 0UL; i < 3UL; ++i) {
    EXPECT_EQ(ele1[i], optiling::FloatToUint16(expet_data[i]));
  }
}

TEST_F(TilingDataUT, AppendAttrIndexInvalid) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 10, AttrDataType::kInt64, AttrDataType::kInt32),
            ge::GRAPH_FAILED);
}

TEST_F(TilingDataUT, AppendAttrSrcTypeInvalid) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 1, AttrDataType::kListListInt32,
                                                AttrDataType::kListListInt32),
            ge::GRAPH_FAILED);
}

TEST_F(TilingDataUT, AppendAttrDstTypeInvalid) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  auto holder = BuildTestContext();
  auto context = holder.GetContext<TilingContext>();
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(
      tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 1, AttrDataType::kInt64, AttrDataType::kListListInt32),
      ge::GRAPH_FAILED);
  EXPECT_EQ(tiling_data->AppendConvertedAttrVal(context->GetAttrs(), 1, AttrDataType::kInt64, AttrDataType::kString),
            ge::GRAPH_FAILED);
}

TEST_F(TilingDataUT, AppendListOverCapacity) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  EXPECT_NE(tiling_data, nullptr);

  tiling_data->SetDataSize(5);
  std::vector<uint64_t> append_data = {1, 2};
  EXPECT_NE(tiling_data->Append<uint64_t>(append_data.data(), append_data.size()), ge::GRAPH_SUCCESS);
}

TEST_F(TilingDataUT, AppendListOk) {
  auto data = TilingData::CreateCap(20);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  EXPECT_NE(tiling_data, nullptr);

  tiling_data->SetDataSize(5);
  std::vector<uint8_t> append_data = {1, 2, 3, 4};
  EXPECT_EQ(tiling_data->Append<uint8_t>(append_data.data(), append_data.size()), ge::GRAPH_SUCCESS);

  EXPECT_EQ(tiling_data->GetDataSize(), 9);
  for (size_t i = 0UL; i < append_data.size(); ++i) {
    EXPECT_EQ(*(reinterpret_cast<uint8_t *>(tiling_data->GetData()) + 5 + i), append_data[i]);
  }
}
}  // namespace gert
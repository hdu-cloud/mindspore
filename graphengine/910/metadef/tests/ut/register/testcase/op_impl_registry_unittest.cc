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
#define private public
#define protected public
#include "register/op_impl_registry.h"
#include "register/op_impl_registry_base.h"
#undef private
#undef protected
#include <gtest/gtest.h>
#include "exe_graph/runtime/kernel_context.h"
#include "graph/any_value.h"
#include "register/op_impl_registry_api.h"

namespace gert_test {
namespace {
ge::graphStatus TestInferShapeFunc1(gert::InferShapeContext *) {
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus TestInferShapeFunc2(gert::InferShapeContext *) {
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus TestInferShapeFunc3(gert::InferShapeContext *) {
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus TestInferShapeRangeFunc1(gert::InferShapeRangeContext *) {
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus TestInferShapeRangeFunc2(gert::InferShapeRangeContext *) {
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus TestTilingFunc1(gert::TilingContext *) {
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus TestTilingFunc2(gert::TilingContext *) {
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TestInferDataTypeFunc(gert::InferDataTypeContext *) {
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus TestTilingParseFunc1(gert::TilingParseContext *) {
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus TestTilingParseFunc2(gert::TilingParseContext *) {
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TestOpExecuteFunc(gert::OpExecuteContext *) {
  return ge::GRAPH_SUCCESS;
}

struct TilingParseCompileInfo {
  int64_t a;
  int64_t b;
  int64_t c;
  std::vector<int32_t> d;
};
}  // namespace
class OpImplRegistryUT : public testing::Test {

 protected:
  virtual void TearDown() {
    gert::OpImplRegistry::GetInstance().GetAllTypesToImpl().clear();
  }
};

TEST_F(OpImplRegistryUT, Register_impl_null) {
  gert::OpImplRegisterV2 reg("Test");
  reg.impl_.release();
  reg.HostInputs({0, 2, 128});
  reg.TilingInputsDataDependency({0, 1});
}

TEST_F(OpImplRegistryUT, Register_Success_RegisterAll) {
  auto funcs = gert::OpImplRegistry::GetInstance().GetOpImpl("TestFoo");
  ASSERT_EQ(funcs, nullptr);

  IMPL_OP(TestFoo)
      .InferShape(TestInferShapeFunc1)
      .Tiling(TestTilingFunc1)
      .TilingParse<TilingParseCompileInfo>(TestTilingParseFunc1)
      .InferShapeRange(TestInferShapeRangeFunc1)
      .InferDataType(TestInferDataTypeFunc)
      .OpExecuteFunc(TestOpExecuteFunc)
      .HostInputs({0, 2, 128})
      .PrivateAttr("A")
      .PrivateAttr("B", 10L)
      .PrivateAttr("C", std::vector<int64_t>({1, 2, 3, 4}))
      .PrivateAttr("D", "hello")
      .PrivateAttr("E", 20.0F)
      .PrivateAttr("F", true)
      .PrivateAttr("G", std::vector<float>({10.0F, 20.0F}))
      .InputsDataDependency({0, 1, 3, 5});

  IMPL_OP(TestFoo_tilingDepend)
      .TilingInputsDataDependency({0, 1, 128})
      .TilingInputsDataDependency({0});

  IMPL_OP(TestFoo_error)
      .TilingInputsDataDependency({0, 1, 128})
      .InputsDataDependency({0, 1, 3, 5});

  IMPL_OP(TestFoo_error2)
      .InputsDataDependency({0, 1, 3, 5})
      .TilingInputsDataDependency({0, 1, 2, 128});

  funcs = gert::OpImplRegistry::GetInstance().GetOpImpl("TestFoo_tilingDepend");
  EXPECT_TRUE(funcs->IsTilingInputDataDependency(0U));
  EXPECT_TRUE(funcs->IsTilingInputDataDependency(1U));
  EXPECT_FALSE(funcs->IsTilingInputDataDependency(2U));
  EXPECT_FALSE(funcs->IsTilingInputDataDependency(3U));
  EXPECT_FALSE(funcs->IsTilingInputDataDependency(4U));
  EXPECT_FALSE(funcs->IsTilingInputDataDependency(5U));
  EXPECT_FALSE(funcs->IsTilingInputDataDependency(6U));

  funcs = gert::OpImplRegistry::GetInstance().GetOpImpl("TestFoo_error");
  EXPECT_TRUE(funcs->IsTilingInputDataDependency(0U));
  EXPECT_TRUE(funcs->IsTilingInputDataDependency(1U));
  EXPECT_TRUE(funcs->IsInputDataDependency(0U));
  EXPECT_TRUE(funcs->IsInputDataDependency(1U));
  EXPECT_TRUE(funcs->IsInputDataDependency(3U));

  funcs = gert::OpImplRegistry::GetInstance().GetOpImpl("TestFoo_error2");
  EXPECT_TRUE(funcs->IsInputDataDependency(0U));
  EXPECT_TRUE(funcs->IsInputDataDependency(1U));
  EXPECT_TRUE(funcs->IsInputDataDependency(3U));
  EXPECT_TRUE(funcs->IsInputDataDependency(5U));
  EXPECT_TRUE(funcs->IsTilingInputDataDependency(0U));
  EXPECT_TRUE(funcs->IsTilingInputDataDependency(1U));
  EXPECT_TRUE(funcs->IsTilingInputDataDependency(2U));

  funcs = gert::OpImplRegistry::GetInstance().GetOpImpl("TestFoo");
  ASSERT_NE(funcs, nullptr);
  EXPECT_EQ(funcs->infer_shape, &TestInferShapeFunc1);
  EXPECT_EQ(funcs->tiling, &TestTilingFunc1);
  EXPECT_EQ(funcs->max_tiling_data_size, 2048);
  EXPECT_EQ(funcs->tiling_parse, reinterpret_cast<gert::OpImplKernelRegistry::KernelFunc>(&TestTilingParseFunc1));
  EXPECT_NE(funcs->compile_info_creator, nullptr);
  EXPECT_NE(funcs->compile_info_deleter, nullptr);
  EXPECT_EQ(funcs->infer_shape_range, &TestInferShapeRangeFunc1);
  EXPECT_EQ(funcs->infer_datatype, &TestInferDataTypeFunc);
  EXPECT_EQ(funcs->op_execute_func, &TestOpExecuteFunc);

  EXPECT_TRUE(funcs->IsInputDataDependency(0U));
  EXPECT_TRUE(funcs->IsInputDataDependency(1U));
  EXPECT_FALSE(funcs->IsInputDataDependency(2U));
  EXPECT_TRUE(funcs->IsInputDataDependency(3U));
  EXPECT_FALSE(funcs->IsInputDataDependency(4U));
  EXPECT_TRUE(funcs->IsInputDataDependency(5U));
  EXPECT_FALSE(funcs->IsInputDataDependency(6U));

  EXPECT_TRUE(funcs->IsHostInput(0U));
  EXPECT_FALSE(funcs->IsHostInput(1U));
  EXPECT_TRUE(funcs->IsHostInput(2U));
  EXPECT_FALSE(funcs->IsHostInput(3U));
  EXPECT_FALSE(funcs->IsHostInput(4U));
  EXPECT_FALSE(funcs->IsHostInput(5U));
  EXPECT_FALSE(funcs->IsHostInput(6U));

  EXPECT_EQ(funcs->private_attrs.size(), 7u);
  EXPECT_EQ(funcs->private_attrs[0].first, "A");
  EXPECT_TRUE(funcs->private_attrs[0].second.IsEmpty());
  EXPECT_EQ(funcs->private_attrs[1].first, "B");
  ASSERT_NE(funcs->private_attrs[1].second.Get<int64_t>(), nullptr);
  ASSERT_EQ(*funcs->private_attrs[1].second.Get<int64_t>(), 10);
  EXPECT_EQ(funcs->private_attrs[2].first, "C");
  ASSERT_NE(funcs->private_attrs[2].second.Get<std::vector<int64_t>>(), nullptr);
  ASSERT_EQ(*funcs->private_attrs[2].second.Get<std::vector<int64_t>>(), std::vector<int64_t>({1, 2, 3, 4}));
  EXPECT_EQ(funcs->private_attrs[3].first, "D");
  ASSERT_NE(funcs->private_attrs[3].second.Get<std::string>(), nullptr);
  ASSERT_EQ(*funcs->private_attrs[3].second.Get<std::string>(), "hello");
  EXPECT_EQ(funcs->private_attrs[4].first, "E");
  ASSERT_NE(funcs->private_attrs[4].second.Get<float>(), nullptr);
  ASSERT_FLOAT_EQ(*funcs->private_attrs[4].second.Get<float>(), 20.0F);
  EXPECT_EQ(funcs->private_attrs[5].first, "F");
  ASSERT_NE(funcs->private_attrs[5].second.Get<bool>(), nullptr);
  ASSERT_TRUE(*funcs->private_attrs[5].second.Get<bool>());
  EXPECT_EQ(funcs->private_attrs[6].first, "G");
  ASSERT_NE(funcs->private_attrs[6].second.Get<std::vector<float>>(), nullptr);
  ASSERT_EQ(funcs->private_attrs[6].second.Get<std::vector<float>>()->size(), 2U);
  EXPECT_FLOAT_EQ(funcs->private_attrs[6].second.Get<std::vector<float>>()->at(0), 10.0);
  EXPECT_FLOAT_EQ(funcs->private_attrs[6].second.Get<std::vector<float>>()->at(1), 20.0);
}

TEST_F(OpImplRegistryUT, Register_Success_RegisterMultiple) {
  auto funcs = gert::OpImplRegistry::GetInstance().GetOpImpl("TestFoo1");
  ASSERT_EQ(funcs, nullptr);
  funcs = gert::OpImplRegistry::GetInstance().GetOpImpl("TestFoo2");
  ASSERT_EQ(funcs, nullptr);
  funcs = gert::OpImplRegistry::GetInstance().GetOpImpl("TestFoo3");
  ASSERT_EQ(funcs, nullptr);

  IMPL_OP(TestFoo1)
      .InferShape(TestInferShapeFunc1)
      .Tiling(TestTilingFunc1)
      .TilingParse<TilingParseCompileInfo>(TestTilingParseFunc1);

  IMPL_OP(TestFoo2)
      .InferShape(TestInferShapeFunc2)
      .Tiling(TestTilingFunc2)
      .TilingParse<TilingParseCompileInfo>(TestTilingParseFunc2)
      .InferShapeRange(TestInferShapeRangeFunc2);

  IMPL_OP(TestFoo3)
      .InferShape(TestInferShapeFunc3)
      .InferDataType(TestInferDataTypeFunc)
      .PrivateAttr("A", std::vector<float>({10.0F, 20.0F}))
      .InputsDataDependency({0, 1, 3, 5});


  funcs = gert::OpImplRegistry::GetInstance().GetOpImpl("TestFoo1");
  ASSERT_NE(funcs, nullptr);
  EXPECT_EQ(funcs->infer_shape, &TestInferShapeFunc1);
  EXPECT_EQ(funcs->tiling, &TestTilingFunc1);
  EXPECT_EQ(funcs->tiling_parse, reinterpret_cast<gert::OpImplKernelRegistry::KernelFunc>(&TestTilingParseFunc1));
  EXPECT_NE(funcs->compile_info_creator, nullptr);
  EXPECT_NE(funcs->compile_info_deleter, nullptr);

  funcs = gert::OpImplRegistry::GetInstance().GetOpImpl("TestFoo2");
  ASSERT_NE(funcs, nullptr);
  EXPECT_EQ(funcs->infer_shape, &TestInferShapeFunc2);
  EXPECT_EQ(funcs->tiling, &TestTilingFunc2);
  EXPECT_EQ(funcs->tiling_parse, reinterpret_cast<gert::OpImplKernelRegistry::KernelFunc>(&TestTilingParseFunc2));
  EXPECT_NE(funcs->compile_info_creator, nullptr);
  EXPECT_NE(funcs->compile_info_deleter, nullptr);
  EXPECT_EQ(funcs->infer_shape_range, &TestInferShapeRangeFunc2);

  funcs = gert::OpImplRegistry::GetInstance().GetOpImpl("TestFoo3");
  ASSERT_NE(funcs, nullptr);
  EXPECT_EQ(funcs->infer_shape, &TestInferShapeFunc3);
  EXPECT_EQ(funcs->infer_datatype, &TestInferDataTypeFunc);
  EXPECT_EQ(funcs->private_attrs.size(), 1U);
  EXPECT_TRUE(funcs->IsInputDataDependency(0));
  EXPECT_TRUE(funcs->IsInputDataDependency(1));
  EXPECT_TRUE(funcs->IsInputDataDependency(3));
  EXPECT_TRUE(funcs->IsInputDataDependency(5));
}
TEST_F(OpImplRegistryUT, Register_MergeOk_OneOpMultipleTimes1) {
  auto funcs = gert::OpImplRegistry::GetInstance().GetOpImpl("TestFoo");
  ASSERT_EQ(funcs, nullptr);

  IMPL_OP(TestFoo)
      .InferShape(TestInferShapeFunc1)
      .InferShapeRange(TestInferShapeRangeFunc1)
      .InferDataType(TestInferDataTypeFunc)
      .InputsDataDependency({0, 1, 3, 5});

  IMPL_OP(TestFoo)
      .Tiling(TestTilingFunc1)
      .TilingParse<TilingParseCompileInfo>(TestTilingParseFunc1)
      .PrivateAttr("A")
      .PrivateAttr("B", 10L)
      .PrivateAttr("C", std::vector<int64_t>({1, 2, 3, 4}))
      .PrivateAttr("D", "hello")
      .PrivateAttr("E", 20.0F)
      .PrivateAttr("F", true)
      .PrivateAttr("G", std::vector<float>({10.0F, 20.0F}));

  funcs = gert::OpImplRegistry::GetInstance().GetOpImpl("TestFoo");
  ASSERT_NE(funcs, nullptr);
  EXPECT_EQ(funcs->infer_shape, &TestInferShapeFunc1);
  EXPECT_EQ(funcs->tiling, &TestTilingFunc1);
  EXPECT_EQ(funcs->max_tiling_data_size, 2048);
  EXPECT_EQ(funcs->tiling_parse, reinterpret_cast<gert::OpImplKernelRegistry::KernelFunc>(&TestTilingParseFunc1));
  EXPECT_NE(funcs->compile_info_creator, nullptr);
  EXPECT_NE(funcs->compile_info_deleter, nullptr);
  EXPECT_EQ(funcs->infer_shape_range, &TestInferShapeRangeFunc1);
  EXPECT_EQ(funcs->infer_datatype, &TestInferDataTypeFunc);
  EXPECT_TRUE(funcs->IsInputDataDependency(0U));
  EXPECT_TRUE(funcs->IsInputDataDependency(1U));
  EXPECT_FALSE(funcs->IsInputDataDependency(2U));
  EXPECT_TRUE(funcs->IsInputDataDependency(3U));
  EXPECT_FALSE(funcs->IsInputDataDependency(4U));
  EXPECT_TRUE(funcs->IsInputDataDependency(5U));
  EXPECT_FALSE(funcs->IsInputDataDependency(6U));

  ASSERT_EQ(funcs->private_attrs.size(), 7u);
  EXPECT_EQ(funcs->private_attrs[0].first, "A");
  EXPECT_TRUE(funcs->private_attrs[0].second.IsEmpty());
  EXPECT_EQ(funcs->private_attrs[1].first, "B");
  ASSERT_NE(funcs->private_attrs[1].second.Get<int64_t>(), nullptr);
  ASSERT_EQ(*funcs->private_attrs[1].second.Get<int64_t>(), 10);
  EXPECT_EQ(funcs->private_attrs[2].first, "C");
  ASSERT_NE(funcs->private_attrs[2].second.Get<std::vector<int64_t>>(), nullptr);
  ASSERT_EQ(*funcs->private_attrs[2].second.Get<std::vector<int64_t>>(), std::vector<int64_t>({1, 2, 3, 4}));
  EXPECT_EQ(funcs->private_attrs[3].first, "D");
  ASSERT_NE(funcs->private_attrs[3].second.Get<std::string>(), nullptr);
  ASSERT_EQ(*funcs->private_attrs[3].second.Get<std::string>(), "hello");
  EXPECT_EQ(funcs->private_attrs[4].first, "E");
  ASSERT_NE(funcs->private_attrs[4].second.Get<float>(), nullptr);
  ASSERT_FLOAT_EQ(*funcs->private_attrs[4].second.Get<float>(), 20.0F);
  EXPECT_EQ(funcs->private_attrs[5].first, "F");
  ASSERT_NE(funcs->private_attrs[5].second.Get<bool>(), nullptr);
  ASSERT_TRUE(*funcs->private_attrs[5].second.Get<bool>());
  EXPECT_EQ(funcs->private_attrs[6].first, "G");
  ASSERT_NE(funcs->private_attrs[6].second.Get<std::vector<float>>(), nullptr);
  ASSERT_EQ(funcs->private_attrs[6].second.Get<std::vector<float>>()->size(), 2U);
  EXPECT_FLOAT_EQ(funcs->private_attrs[6].second.Get<std::vector<float>>()->at(0), 10.0);
  EXPECT_FLOAT_EQ(funcs->private_attrs[6].second.Get<std::vector<float>>()->at(1), 20.0);
}
TEST_F(OpImplRegistryUT, Register_MergeOk_OneOpMultipleTimes2) {
  auto funcs = gert::OpImplRegistry::GetInstance().GetOpImpl("TestFoo");
  ASSERT_EQ(funcs, nullptr);

  IMPL_OP(TestFoo)
      .Tiling(TestTilingFunc1)
      .TilingParse<TilingParseCompileInfo>(TestTilingParseFunc1)
      .PrivateAttr("A")
      .PrivateAttr("B", 10L)
      .PrivateAttr("C", std::vector<int64_t>({1, 2, 3, 4}))
      .PrivateAttr("D", "hello")
      .PrivateAttr("E", 20.0F)
      .PrivateAttr("F", true)
      .PrivateAttr("G", std::vector<float>({10.0F, 20.0F}));

  IMPL_OP(TestFoo)
      .InferShape(TestInferShapeFunc1)
      .InferShapeRange(TestInferShapeRangeFunc1)
      .InferDataType(TestInferDataTypeFunc)
      .InputsDataDependency({0, 1, 3, 5});

  funcs = gert::OpImplRegistry::GetInstance().GetOpImpl("TestFoo");
  ASSERT_NE(funcs, nullptr);
  EXPECT_EQ(funcs->infer_shape, &TestInferShapeFunc1);
  EXPECT_EQ(funcs->tiling, &TestTilingFunc1);
  EXPECT_EQ(funcs->max_tiling_data_size, 2048);
  EXPECT_EQ(funcs->tiling_parse, reinterpret_cast<gert::OpImplKernelRegistry::KernelFunc>(&TestTilingParseFunc1));
  EXPECT_NE(funcs->compile_info_creator, nullptr);
  EXPECT_NE(funcs->compile_info_deleter, nullptr);
  EXPECT_EQ(funcs->infer_shape_range, &TestInferShapeRangeFunc1);
  EXPECT_EQ(funcs->infer_datatype, &TestInferDataTypeFunc);
  EXPECT_TRUE(funcs->IsInputDataDependency(0U));
  EXPECT_TRUE(funcs->IsInputDataDependency(1U));
  EXPECT_FALSE(funcs->IsInputDataDependency(2U));
  EXPECT_TRUE(funcs->IsInputDataDependency(3U));
  EXPECT_FALSE(funcs->IsInputDataDependency(4U));
  EXPECT_TRUE(funcs->IsInputDataDependency(5U));
  EXPECT_FALSE(funcs->IsInputDataDependency(6U));

  ASSERT_EQ(funcs->private_attrs.size(), 7u);
  EXPECT_EQ(funcs->private_attrs[0].first, "A");
  EXPECT_TRUE(funcs->private_attrs[0].second.IsEmpty());
  EXPECT_EQ(funcs->private_attrs[1].first, "B");
  ASSERT_NE(funcs->private_attrs[1].second.Get<int64_t>(), nullptr);
  ASSERT_EQ(*funcs->private_attrs[1].second.Get<int64_t>(), 10);
  EXPECT_EQ(funcs->private_attrs[2].first, "C");
  ASSERT_NE(funcs->private_attrs[2].second.Get<std::vector<int64_t>>(), nullptr);
  ASSERT_EQ(*funcs->private_attrs[2].second.Get<std::vector<int64_t>>(), std::vector<int64_t>({1, 2, 3, 4}));
  EXPECT_EQ(funcs->private_attrs[3].first, "D");
  ASSERT_NE(funcs->private_attrs[3].second.Get<std::string>(), nullptr);
  ASSERT_EQ(*funcs->private_attrs[3].second.Get<std::string>(), "hello");
  EXPECT_EQ(funcs->private_attrs[4].first, "E");
  ASSERT_NE(funcs->private_attrs[4].second.Get<float>(), nullptr);
  ASSERT_FLOAT_EQ(*funcs->private_attrs[4].second.Get<float>(), 20.0F);
  EXPECT_EQ(funcs->private_attrs[5].first, "F");
  ASSERT_NE(funcs->private_attrs[5].second.Get<bool>(), nullptr);
  ASSERT_TRUE(*funcs->private_attrs[5].second.Get<bool>());
  EXPECT_EQ(funcs->private_attrs[6].first, "G");
  ASSERT_NE(funcs->private_attrs[6].second.Get<std::vector<float>>(), nullptr);
  ASSERT_EQ(funcs->private_attrs[6].second.Get<std::vector<float>>()->size(), 2U);
  EXPECT_FLOAT_EQ(funcs->private_attrs[6].second.Get<std::vector<float>>()->at(0), 10.0);
  EXPECT_FLOAT_EQ(funcs->private_attrs[6].second.Get<std::vector<float>>()->at(1), 20.0);}

TEST_F(OpImplRegistryUT, Register_DefaultValue_WhenNotRegister) {
  IMPL_OP(TestFoo);

  auto funcs = gert::OpImplRegistry::GetInstance().GetOpImpl("TestFoo");
  ASSERT_NE(funcs, nullptr);
  EXPECT_EQ(funcs->infer_shape, nullptr);
  EXPECT_EQ(funcs->infer_shape_range, nullptr);
  EXPECT_EQ(funcs->infer_datatype, nullptr);
  EXPECT_EQ(funcs->tiling, nullptr);
  EXPECT_EQ(funcs->tiling_parse, nullptr);
  EXPECT_EQ(funcs->compile_info_creator, nullptr);
  EXPECT_EQ(funcs->compile_info_deleter, nullptr);
  EXPECT_EQ(funcs->inputs_dependency, 0);
  EXPECT_EQ(funcs->private_attrs.size(), 0U);
}
TEST_F(OpImplRegistryUT, Register_DefaultTilingSize2048_Tiling) {
  IMPL_OP(TestFoo).Tiling(TestTilingFunc1);

  auto funcs = gert::OpImplRegistry::GetInstance().GetOpImpl("TestFoo");
  ASSERT_NE(funcs, nullptr);
  EXPECT_EQ(funcs->max_tiling_data_size, 2048);
}
TEST_F(OpImplRegistryUT, Register_Ok_TilingWithSize) {
  IMPL_OP(TestFoo).Tiling(TestTilingFunc1, 1024);

  auto funcs = gert::OpImplRegistry::GetInstance().GetOpImpl("TestFoo");
  ASSERT_NE(funcs, nullptr);
  EXPECT_EQ(funcs->tiling, &TestTilingFunc1);
  EXPECT_EQ(funcs->max_tiling_data_size, 1024);
}
TEST_F(OpImplRegistryUT, RegisterInferShapeOk) {
  IMPL_OP(TestConv2D).InferShape(TestInferShapeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestConv2D").infer_shape, &TestInferShapeFunc1);

  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestConv2D").tiling, nullptr);
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestConv2D").inputs_dependency);
}

TEST_F(OpImplRegistryUT, RegisterInferShapeAndTilingOk) {
  IMPL_OP(TestAdd).InferShape(TestInferShapeFunc1).Tiling(TestTilingFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestAdd").infer_shape, &TestInferShapeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestAdd").tiling, &TestTilingFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().GetOpImpl("TestAdd")->max_tiling_data_size, 2048);
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestAdd").inputs_dependency);
}

TEST_F(OpImplRegistryUT, RegisterInferShapeRangeOk) {
  IMPL_OP(TestConv2D2).InferShapeRange(TestInferShapeRangeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestConv2D2").infer_shape_range,
            &TestInferShapeRangeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestConv2D2").infer_shape, nullptr);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestConv2D2").tiling, nullptr);
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestConv2D2").inputs_dependency);
}

TEST_F(OpImplRegistryUT, RegisterInferShapeAndInferShapeAndTilingOk) {
  IMPL_OP(TestAdd).InferShape(TestInferShapeFunc1).InferShapeRange(TestInferShapeRangeFunc1).Tiling(TestTilingFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestAdd").infer_shape, &TestInferShapeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestAdd").infer_shape_range,
            &TestInferShapeRangeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestAdd").tiling, &TestTilingFunc1);
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestAdd").inputs_dependency);
}

TEST_F(OpImplRegistryUT, InputsDependencyOk) {
  IMPL_OP(TestReshape).InferShape(TestInferShapeFunc1).InputsDataDependency({1});
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestReshape").infer_shape, &TestInferShapeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestReshape").tiling, nullptr);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestReshape").inputs_dependency, 2);
}

TEST_F(OpImplRegistryUT, Registry_null_GetOpNotRegistered) {
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().GetOpImpl("Test"), nullptr);
}
TEST_F(OpImplRegistryUT, DefaultImpl) {
  // auto tiling
  IMPL_OP_DEFAULT().Tiling(TestTilingFunc2);
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DefaultImpl"), nullptr);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().GetOpImpl("DefaultImpl")->tiling, &TestTilingFunc2);
}

TEST_F(OpImplRegistryUT, RegisterPrivateAttrOk) {
  IMPL_OP(TestPrivateConv2D).InferShape(TestInferShapeFunc1).PrivateAttr("attr1");

  const char *op_type = "TestPrivateConv2D";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 1);
  EXPECT_EQ(private_attrs[0].first.GetString(), std::string("attr1"));
  EXPECT_TRUE(private_attrs[0].second.IsEmpty());

  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type).infer_shape, &TestInferShapeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type).tiling, nullptr);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type).unique_private_attrs.size(), 1);
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type).inputs_dependency);
}

TEST_F(OpImplRegistryUT, RegisterPrivateAttrDuplicatedUsingSameOpType) {
  IMPL_OP(TestPrivateConv2D).InferShape(TestInferShapeFunc1).PrivateAttr("attr1");
  IMPL_OP(TestPrivateConv2D).PrivateAttr("attr2");

  const char *op_type = "TestPrivateConv2D";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 1);
  EXPECT_EQ(private_attrs[0].first.GetString(), std::string("attr2"));
  EXPECT_TRUE(private_attrs[0].second.IsEmpty());

  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type).infer_shape, &TestInferShapeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type).tiling, nullptr);
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type).inputs_dependency);
}

TEST_F(OpImplRegistryUT, UsePrivateAttrAlreadyRegistered) {
  IMPL_OP(TestPrivateConv2D).InferShape(TestInferShapeFunc1).PrivateAttr("attr2");

  const char *op_type = "TestPrivateConv2D";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 1);
  EXPECT_EQ(private_attrs[0].first.GetString(), std::string("attr2"));
  EXPECT_TRUE(private_attrs[0].second.IsEmpty());

  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type).infer_shape, &TestInferShapeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type).tiling, nullptr);
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type).inputs_dependency);
}

TEST_F(OpImplRegistryUT, RegisterMultiPrivateAttrs) {
  IMPL_OP(TestPrivateAdd).PrivateAttr("attr1").PrivateAttr("attr2").PrivateAttr("attr3");

  const char *op_type = "TestPrivateAdd";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  const std::vector<std::string> private_attr_names = {"attr1", "attr2", "attr3"};
  EXPECT_EQ(private_attrs.size(), private_attr_names.size());
  for (size_t index = 0UL; index < private_attr_names.size(); ++index) {
    EXPECT_TRUE(private_attrs[index].second.IsEmpty());
    EXPECT_EQ(private_attrs[index].first.GetString(), private_attr_names[index]);
  }
}

TEST_F(OpImplRegistryUT, RegisterSamePrivateAttrs) {
  IMPL_OP(TestPrivateSub).PrivateAttr("attr1").PrivateAttr("attr1").PrivateAttr("attr2").PrivateAttr("attr3");
  const char *op_type = "TestPrivateSub";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  const std::vector<std::string> private_attr_names = {"attr1", "attr2", "attr3"};
  EXPECT_EQ(private_attrs.size(), private_attr_names.size());
  for (size_t index = 0UL; index < private_attr_names.size(); ++index) {
    EXPECT_TRUE(private_attrs[index].second.IsEmpty());
    EXPECT_EQ(private_attrs[index].first.GetString(), private_attr_names[index]);
  }
}

TEST_F(OpImplRegistryUT, RegisterPrivateAttrsUsingNullptr) {
  IMPL_OP(TestPrivateMul).PrivateAttr(nullptr);

  const char *op_type = "TestPrivateMul";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 0);
}

TEST_F(OpImplRegistryUT, RegisterPrivateAttrsUsingEmptyName) {
  IMPL_OP(TestPrivateDiv).PrivateAttr("");

  const char *op_type = "TestPrivateDiv";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 0);
}

TEST_F(OpImplRegistryUT, GetPrivateAttrFailedWhenTypeMismatchName) {
  IMPL_OP(TestPrivateDiv).PrivateAttr("TestOpType");

  const char *op_type = "TestOptype";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 0);
}

TEST_F(OpImplRegistryUT, RegisterIntPrivateAttrOk) {
  constexpr int64_t private_attr_val = 10;
  IMPL_OP(TestIntOpdesc).PrivateAttr("attr1", private_attr_val);
  const char *op_type = "TestIntOpdesc";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 1);
  EXPECT_EQ(private_attrs[0].first.GetString(), std::string("attr1"));
  EXPECT_TRUE(!private_attrs[0].second.IsEmpty());
  int64_t private_attr_val_ret;
  EXPECT_EQ(private_attrs[0].second.GetValue(private_attr_val_ret), ge::GRAPH_SUCCESS);
  EXPECT_EQ(private_attr_val_ret, private_attr_val);
}

TEST_F(OpImplRegistryUT, RegisterListIntPrivateAttrOk) {
  std::vector<int64_t> private_attr_val = {10, 20, 30};
  IMPL_OP(TestListIntOpdesc).PrivateAttr("attr1", private_attr_val);
  const char *op_type = "TestListIntOpdesc";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 1);
  EXPECT_EQ(private_attrs[0].first.GetString(), std::string("attr1"));
  EXPECT_TRUE(!private_attrs[0].second.IsEmpty());
  std::vector<int64_t> private_attr_val_ret;
  EXPECT_EQ(private_attrs[0].second.GetValue(private_attr_val_ret), ge::GRAPH_SUCCESS);
  EXPECT_EQ(private_attr_val_ret, private_attr_val);
}

TEST_F(OpImplRegistryUT, RegisterStringPrivateAttrOk) {
  const char *private_attr_val = "10";
  IMPL_OP(TestStringOpdesc).PrivateAttr("attr1", private_attr_val);
  const char *op_type = "TestStringOpdesc";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 1);
  EXPECT_EQ(private_attrs[0].first.GetString(), std::string("attr1"));
  EXPECT_TRUE(!private_attrs[0].second.IsEmpty());
  std::string private_attr_val_ret;
  EXPECT_EQ(private_attrs[0].second.GetValue(private_attr_val_ret), ge::GRAPH_SUCCESS);
  EXPECT_EQ(private_attr_val_ret, std::string(private_attr_val));
}

TEST_F(OpImplRegistryUT, RegisterFloatPrivateAttrOk) {
  float private_attr_val = 10.0;
  IMPL_OP(TestFloatOpdesc).PrivateAttr("attr1", private_attr_val);
  const char *op_type = "TestFloatOpdesc";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 1);
  EXPECT_EQ(private_attrs[0].first.GetString(), std::string("attr1"));
  EXPECT_TRUE(!private_attrs[0].second.IsEmpty());
  float private_attr_val_ret;
  EXPECT_EQ(private_attrs[0].second.GetValue(private_attr_val_ret), ge::GRAPH_SUCCESS);
  EXPECT_EQ(private_attr_val_ret, private_attr_val);
}

TEST_F(OpImplRegistryUT, RegisterV1FloatPrivateAttrOk) {
  float private_attr_val = 10.0;
  gert::OpImplRegister register_v1("TestDeprecatedRegister");
  register_v1.PrivateAttr("attr1", private_attr_val);
  const char *op_type = "TestDeprecatedRegister";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 1);
  EXPECT_EQ(private_attrs[0].first.GetString(), std::string("attr1"));
  EXPECT_TRUE(!private_attrs[0].second.IsEmpty());
  float private_attr_val_ret;
  EXPECT_EQ(private_attrs[0].second.GetValue(private_attr_val_ret), ge::GRAPH_SUCCESS);
  EXPECT_EQ(private_attr_val_ret, private_attr_val);
}

TEST_F(OpImplRegistryUT, RegisterListFloatPrivateAttrOk) {
  std::vector<float> private_attr_val = {10.0, 20.0, 30.0};
  IMPL_OP(TestListFloatOpdesc).PrivateAttr("attr1", private_attr_val);
  const char *op_type = "TestListFloatOpdesc";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 1);
  EXPECT_EQ(private_attrs[0].first.GetString(), std::string("attr1"));
  EXPECT_TRUE(!private_attrs[0].second.IsEmpty());
  std::vector<float> private_attr_val_ret;
  EXPECT_EQ(private_attrs[0].second.GetValue(private_attr_val_ret), ge::GRAPH_SUCCESS);
  EXPECT_EQ(private_attr_val_ret, private_attr_val);
}

TEST_F(OpImplRegistryUT, OpImplRegister_RegisterListFloatOK) {
  const char *OpType = "ListFloatOp";
  std::vector<float> private_attr_val = {10.0, 20.0, 30.0};
  static gert::OpImplRegister list_float_op = gert::OpImplRegister(OpType);
  list_float_op.PrivateAttr("attr1", private_attr_val);
  list_float_op.InputsDataDependency({0});

  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(OpType);
  EXPECT_EQ(private_attrs.size(), 1);
  EXPECT_EQ(private_attrs[0].first.GetString(), std::string("attr1"));
  EXPECT_TRUE(!private_attrs[0].second.IsEmpty());
  std::vector<float> private_attr_val_ret;
  EXPECT_EQ(private_attrs[0].second.GetValue(private_attr_val_ret), ge::GRAPH_SUCCESS);
  EXPECT_EQ(private_attr_val_ret, private_attr_val);
}

TEST_F(OpImplRegistryUT, RegisterBoolPrivateAttrOk) {
  bool private_attr_val = false;
  IMPL_OP(TestBoolOpdesc).PrivateAttr("attr1", private_attr_val);
  const char *op_type = "TestBoolOpdesc";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 1);
  EXPECT_EQ(private_attrs[0].first.GetString(), std::string("attr1"));
  EXPECT_TRUE(!private_attrs[0].second.IsEmpty());
  bool private_attr_val_ret;
  EXPECT_EQ(private_attrs[0].second.GetValue(private_attr_val_ret), ge::GRAPH_SUCCESS);
  EXPECT_EQ(private_attr_val_ret, private_attr_val);
}

TEST_F(OpImplRegistryUT, RegisterMixPrivateAttrOk) {
  const char *str_attr_val = "Test";
  std::vector<int64_t> listint_attr_val = {10, 20, 30};
  IMPL_OP(TestMixOpdesc).PrivateAttr("attr1")
      .PrivateAttr("attr2", str_attr_val)
      .PrivateAttr("attr3", listint_attr_val);
  const char *op_type = "TestMixOpdesc";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  constexpr size_t private_attr_size = 3UL;
  EXPECT_EQ(private_attrs.size(), private_attr_size);
  EXPECT_EQ(private_attrs[0].first.GetString(), std::string("attr1"));
  EXPECT_TRUE(private_attrs[0].second.IsEmpty());
  std::string str_attr_val_ret;
  EXPECT_EQ(private_attrs[1].first.GetString(), std::string("attr2"));
  EXPECT_EQ(private_attrs[1].second.GetValue(str_attr_val_ret), ge::GRAPH_SUCCESS);
  EXPECT_EQ(str_attr_val_ret, std::string(str_attr_val));
  std::vector<int64_t> listint_attr_val_ret;
  EXPECT_EQ(private_attrs[2].first.GetString(), std::string("attr3"));
  EXPECT_EQ(private_attrs[2].second.GetValue(listint_attr_val_ret), ge::GRAPH_SUCCESS);
  EXPECT_EQ(listint_attr_val_ret, listint_attr_val);
}

TEST_F(OpImplRegistryUT, RegisterInferDatatypeOk) {
  IMPL_OP(TestConv2D).InferShape(TestInferShapeFunc1).InferDataType(TestInferDataTypeFunc);

  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestConv2D").infer_shape, &TestInferShapeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestConv2D").infer_datatype, &TestInferDataTypeFunc);
}
TEST_F(OpImplRegistryUT, GetOpImplFunctionsOk) {
  IMPL_OP(TestConv2D).InferShape(TestInferShapeFunc1).InferDataType(TestInferDataTypeFunc);

  auto impl_num = GetRegisteredOpNum();
  auto impl_funcs = std::unique_ptr<TypesToImpl[]>(new(std::nothrow) TypesToImpl[impl_num]);
  auto ret = GetOpImplFunctions(reinterpret_cast<TypesToImpl *>(impl_funcs.get()), impl_num);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  bool check = false;
  for (uint32_t i = 0; i < impl_num; i++) {
    std::string op_type = impl_funcs[i].op_type;
    if (op_type == "TestConv2D") {
      check = true;
      EXPECT_EQ(impl_funcs[i].funcs.infer_shape, &TestInferShapeFunc1);
      EXPECT_EQ(impl_funcs[i].funcs.infer_datatype, &TestInferDataTypeFunc);
    }
  }
  EXPECT_EQ(check, true);
}

TEST_F(OpImplRegistryUT, GetOpImplFunctionsERR) {
  IMPL_OP(TestConv2D).InferShape(TestInferShapeFunc1).InferDataType(TestInferDataTypeFunc);

  auto impl_num = GetRegisteredOpNum();
  auto impl_funcs = std::unique_ptr<TypesToImpl[]>(new(std::nothrow) TypesToImpl[impl_num]);
  auto ret = GetOpImplFunctions(reinterpret_cast<TypesToImpl *>(impl_funcs.get()), 10);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(OpImplRegistryUT, Retpeat_register_InputsDataDependency_success) {
  auto funcs = gert::OpImplRegistry::GetInstance().GetOpImpl("TestFoo");
  ASSERT_EQ(funcs, nullptr);

  IMPL_OP(TestFoo)
      .InferShape(TestInferShapeFunc1)
      .InferShapeRange(TestInferShapeRangeFunc1)
      .InferDataType(TestInferDataTypeFunc)
      .InputsDataDependency({0, 1})
      .InputsDataDependency({0, 2, 3});
  funcs = gert::OpImplRegistry::GetInstance().GetOpImpl("TestFoo");
  ASSERT_NE(funcs, nullptr);
  EXPECT_TRUE(funcs->IsInputDataDependency(0));
  EXPECT_TRUE(funcs->IsInputDataDependency(1));
  EXPECT_TRUE(funcs->IsInputDataDependency(2));
  EXPECT_TRUE(funcs->IsInputDataDependency(3));
}
}  // namespace gert_test
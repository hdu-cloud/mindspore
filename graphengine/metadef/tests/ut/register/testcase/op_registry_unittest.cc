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
#include "register/op_impl_registry.h"
#include <gtest/gtest.h>
#include "exe_graph/runtime/kernel_context.h"
#include "graph/any_value.h"
namespace gert_test {
class OpImplRegistryUT : public testing::Test {};
namespace {
ge::graphStatus TestFunc1(gert::KernelContext *context) {
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus TestInferShapeFunc1(gert::InferShapeContext *context) {
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus TestInferShapeRangeFunc1(gert::InferShapeRangeContext *context) {
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus TestTilingFunc1(gert::TilingContext *context) {
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus TestTilingFunc2(gert::TilingContext *context) {
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus TestInferDataTypeFunc(gert::InferDataTypeContext *context) {
  return ge::GRAPH_SUCCESS;
}
}

TEST_F(OpImplRegistryUT, RegisterInferShapeOk) {
  IMPL_OP(TestConv2D);
  op_impl_register_TestConv2D.InferShape(TestInferShapeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestConv2D").infer_shape, TestInferShapeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestConv2D").tiling, nullptr);
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestConv2D").inputs_dependency);
}

TEST_F(OpImplRegistryUT, RegisterInferShapeAndTilingOk) {
  IMPL_OP(TestAdd);
  op_impl_register_TestAdd.InferShape(TestInferShapeFunc1).Tiling(TestTilingFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestAdd").infer_shape, TestInferShapeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestAdd").tiling, TestTilingFunc1);
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestAdd").inputs_dependency);
}

TEST_F(OpImplRegistryUT, RegisterInferShapeRangeOk) {
  IMPL_OP(TestConv2D2);
  op_impl_register_TestConv2D2.InferShapeRange(TestInferShapeRangeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestConv2D2").infer_shape_range,
            TestInferShapeRangeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestConv2D2").infer_shape, nullptr);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestConv2D2").tiling, nullptr);
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestConv2D2").inputs_dependency);
}

TEST_F(OpImplRegistryUT, RegisterInferShapeAndInferShapeAndTilingOk) {
  IMPL_OP(TestAdd);
  op_impl_register_TestAdd.InferShape(TestInferShapeFunc1).InferShapeRange(TestInferShapeRangeFunc1).Tiling(TestTilingFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestAdd").infer_shape, TestInferShapeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestAdd").infer_shape_range,
            TestInferShapeRangeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestAdd").tiling, TestTilingFunc1);
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestAdd").inputs_dependency);
}

TEST_F(OpImplRegistryUT, InputsDependencyOk) {
  IMPL_OP(TestReshape);
  op_impl_register_TestReshape.InferShape(TestInferShapeFunc1).InputsDataDependency({1});
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestReshape").infer_shape, TestInferShapeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestReshape").tiling, nullptr);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestReshape").inputs_dependency, 2);
}

TEST_F(OpImplRegistryUT, NotRegisterOp) {
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().GetOpImpl("Test"), nullptr);
}
TEST_F(OpImplRegistryUT, DefaultImpl) {
  // auto tiling
  IMPL_OP_DEFAULT().Tiling(TestTilingFunc2);
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DefaultImpl"), nullptr);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().GetOpImpl("DefaultImpl")->tiling, TestTilingFunc2);
}

TEST_F(OpImplRegistryUT, RegisterPrivateAttrOk) {
  IMPL_OP(TestPrivateConv2D);
  op_impl_register_TestPrivateConv2D.InferShape(TestInferShapeFunc1).PrivateAttr("attr1");

  const char *op_type = "TestPrivateConv2D";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 1);
  EXPECT_EQ(private_attrs[0].first, string("attr1"));
  EXPECT_TRUE(private_attrs[0].second.IsEmpty());

  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type).infer_shape, TestInferShapeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type).tiling, nullptr);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type).unique_private_attrs.size(), 1);
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type).inputs_dependency);
}

TEST_F(OpImplRegistryUT, RegisterPrivateAttrDuplicatedUsingSameOpType) {
  IMPL_OP(TestPrivateConv2D);
  op_impl_register_TestPrivateConv2D.PrivateAttr("attr2");

  const char *op_type = "TestPrivateConv2D";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 1);
  EXPECT_EQ(private_attrs[0].first, std::string("attr2"));
  EXPECT_TRUE(private_attrs[0].second.IsEmpty());

  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type).infer_shape, TestInferShapeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type).tiling, nullptr);
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type).inputs_dependency);
}

TEST_F(OpImplRegistryUT, UsePrivateAttrAlreadyRegistered) {
  const char *op_type = "TestPrivateConv2D";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 1);
  EXPECT_EQ(private_attrs[0].first, std::string("attr2"));
  EXPECT_TRUE(private_attrs[0].second.IsEmpty());

  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type).infer_shape, TestInferShapeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type).tiling, nullptr);
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type).inputs_dependency);
}

TEST_F(OpImplRegistryUT, RegisterMultiPrivateAttrs) {
  IMPL_OP(TestPrivateAdd);
  op_impl_register_TestPrivateAdd.PrivateAttr("attr1").PrivateAttr("attr2").PrivateAttr("attr3");

  const char *op_type = "TestPrivateAdd";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  const std::vector<std::string> private_attr_names = {"attr1", "attr2", "attr3"};
  EXPECT_EQ(private_attrs.size(), private_attr_names.size());
  for (size_t index = 0UL; index < private_attr_names.size(); ++index) {
    EXPECT_TRUE(private_attrs[index].second.IsEmpty());
    EXPECT_EQ(private_attrs[index].first, private_attr_names[index]);
  }
}

TEST_F(OpImplRegistryUT, RegisterSamePrivateAttrs) {
  IMPL_OP(TestPrivateSub);
  op_impl_register_TestPrivateSub
      .PrivateAttr("attr1")
      .PrivateAttr("attr1")
      .PrivateAttr("attr2")
      .PrivateAttr("attr3");
  const char *op_type = "TestPrivateSub";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  const std::vector<std::string> private_attr_names = {"attr1", "attr2", "attr3"};
  EXPECT_EQ(private_attrs.size(), private_attr_names.size());
  for (size_t index = 0UL; index < private_attr_names.size(); ++index) {
    EXPECT_TRUE(private_attrs[index].second.IsEmpty());
    EXPECT_EQ(private_attrs[index].first, private_attr_names[index]);
  }
}

TEST_F(OpImplRegistryUT, RegisterPrivateAttrsUsingNullptr) {
  IMPL_OP(TestPrivateMul);
  op_impl_register_TestPrivateMul.PrivateAttr(nullptr);

  const char *op_type = "TestPrivateMul";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 0);
}

TEST_F(OpImplRegistryUT, RegisterPrivateAttrsUsingEmptyName) {
  IMPL_OP(TestPrivateDiv);
  op_impl_register_TestPrivateDiv.PrivateAttr("");

  const char *op_type = "TestPrivateDiv";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 0);
}

TEST_F(OpImplRegistryUT, GetPrivateAttrFailedWhenTypeMismatchName) {
  IMPL_OP(TestPrivateDiv);
  op_impl_register_TestPrivateDiv.PrivateAttr("TestOpType");

  const char *op_type = "TestOptype";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 0);
}

TEST_F(OpImplRegistryUT, RegisterIntPrivateAttrOk) {
  constexpr int64_t private_attr_val = 10;
  IMPL_OP(TestIntOpdesc);
  op_impl_register_TestIntOpdesc.PrivateAttr("attr1", private_attr_val);
  const char *op_type = "TestIntOpdesc";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 1);
  EXPECT_EQ(private_attrs[0].first, string("attr1"));
  EXPECT_TRUE(!private_attrs[0].second.IsEmpty());
  int64_t private_attr_val_ret;
  EXPECT_EQ(private_attrs[0].second.GetValue(private_attr_val_ret), ge::GRAPH_SUCCESS);
  EXPECT_EQ(private_attr_val_ret, private_attr_val);
}

TEST_F(OpImplRegistryUT, RegisterListIntPrivateAttrOk) {
  std::vector<int64_t> private_attr_val = {10, 20, 30};
  IMPL_OP(TestListIntOpdesc);
  op_impl_register_TestListIntOpdesc.PrivateAttr("attr1", private_attr_val);
  const char *op_type = "TestListIntOpdesc";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 1);
  EXPECT_EQ(private_attrs[0].first, string("attr1"));
  EXPECT_TRUE(!private_attrs[0].second.IsEmpty());
  std::vector<int64_t> private_attr_val_ret;
  EXPECT_EQ(private_attrs[0].second.GetValue(private_attr_val_ret), ge::GRAPH_SUCCESS);
  EXPECT_EQ(private_attr_val_ret, private_attr_val);
}

TEST_F(OpImplRegistryUT, RegisterStringPrivateAttrOk) {
  const char *private_attr_val = "10";
  IMPL_OP(TestStringOpdesc);
  op_impl_register_TestStringOpdesc.PrivateAttr("attr1", private_attr_val);
  const char *op_type = "TestStringOpdesc";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 1);
  EXPECT_EQ(private_attrs[0].first, string("attr1"));
  EXPECT_TRUE(!private_attrs[0].second.IsEmpty());
  string private_attr_val_ret;
  EXPECT_EQ(private_attrs[0].second.GetValue(private_attr_val_ret), ge::GRAPH_SUCCESS);
  EXPECT_EQ(private_attr_val_ret, string(private_attr_val));
}

TEST_F(OpImplRegistryUT, RegisterFloatPrivateAttrOk) {
  float private_attr_val = 10.0;
  IMPL_OP(TestFloatOpdesc);
  op_impl_register_TestFloatOpdesc.PrivateAttr("attr1", private_attr_val);
  const char *op_type = "TestFloatOpdesc";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 1);
  EXPECT_EQ(private_attrs[0].first, string("attr1"));
  EXPECT_TRUE(!private_attrs[0].second.IsEmpty());
  float private_attr_val_ret;
  EXPECT_EQ(private_attrs[0].second.GetValue(private_attr_val_ret), ge::GRAPH_SUCCESS);
  EXPECT_EQ(private_attr_val_ret, private_attr_val);
}

TEST_F(OpImplRegistryUT, RegisterListFloatPrivateAttrOk) {
  std::vector<float> private_attr_val = {10.0, 20.0, 30.0};
  IMPL_OP(TestListFloatOpdesc);
  op_impl_register_TestListFloatOpdesc.PrivateAttr("attr1", private_attr_val);
  const char *op_type = "TestListFloatOpdesc";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 1);
  EXPECT_EQ(private_attrs[0].first, string("attr1"));
  EXPECT_TRUE(!private_attrs[0].second.IsEmpty());
  std::vector<float> private_attr_val_ret;
  EXPECT_EQ(private_attrs[0].second.GetValue(private_attr_val_ret), ge::GRAPH_SUCCESS);
  EXPECT_EQ(private_attr_val_ret, private_attr_val);
}

TEST_F(OpImplRegistryUT, RegisterBoolPrivateAttrOk) {
  bool private_attr_val = false;
  IMPL_OP(TestBoolOpdesc);
  op_impl_register_TestBoolOpdesc.PrivateAttr("attr1", private_attr_val);
  const char *op_type = "TestBoolOpdesc";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  EXPECT_EQ(private_attrs.size(), 1);
  EXPECT_EQ(private_attrs[0].first, string("attr1"));
  EXPECT_TRUE(!private_attrs[0].second.IsEmpty());
  bool private_attr_val_ret;
  EXPECT_EQ(private_attrs[0].second.GetValue(private_attr_val_ret), ge::GRAPH_SUCCESS);
  EXPECT_EQ(private_attr_val_ret, private_attr_val);
}

TEST_F(OpImplRegistryUT, RegisterMixPrivateAttrOk) {
  const char *str_attr_val= "Test";
  std::vector<int64_t> listint_attr_val = {10, 20, 30};
  IMPL_OP(TestMixOpdesc);
  op_impl_register_TestMixOpdesc.PrivateAttr("attr1").PrivateAttr("attr2", str_attr_val).PrivateAttr("attr3", listint_attr_val);
  const char *op_type = "TestMixOpdesc";
  const auto &private_attrs = gert::OpImplRegistry::GetInstance().GetPrivateAttrs(op_type);
  constexpr size_t private_attr_size = 3UL;
  EXPECT_EQ(private_attrs.size(), private_attr_size);
  EXPECT_EQ(private_attrs[0].first, string("attr1"));
  EXPECT_TRUE(private_attrs[0].second.IsEmpty());
  std::string str_attr_val_ret;
  EXPECT_EQ(private_attrs[1].first, string("attr2"));
  EXPECT_EQ(private_attrs[1].second.GetValue(str_attr_val_ret), ge::GRAPH_SUCCESS);
  EXPECT_EQ(str_attr_val_ret, string(str_attr_val));
  std::vector<int64_t> listint_attr_val_ret;
  EXPECT_EQ(private_attrs[2].first, string("attr3"));
  EXPECT_EQ(private_attrs[2].second.GetValue(listint_attr_val_ret), ge::GRAPH_SUCCESS);
  EXPECT_EQ(listint_attr_val_ret, listint_attr_val);
}

TEST_F(OpImplRegistryUT, RegisterInferDatatypeOk) {
  IMPL_OP(TestConv2D).InferShape(TestInferShapeFunc1).InferDataType(TestInferDataTypeFunc);

  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestConv2D").infer_shape, TestInferShapeFunc1);
  EXPECT_EQ(gert::OpImplRegistry::GetInstance().CreateOrGetOpImpl("TestConv2D").infer_datatype, TestInferDataTypeFunc);
}
}  // namespace gert_test
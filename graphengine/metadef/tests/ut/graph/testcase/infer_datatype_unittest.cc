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
#include <memory>

#define protected public
#define private public
#include "graph/op_desc.h"
#include "graph/op_desc_impl.h"  // to test inner func
#define protected public
#define private public
#include "graph_builder_utils.h"
#include "graph/operator_reg.h"
#include "graph/utils/op_desc_utils.h"

namespace ge {
class UTInferDataType : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};
// 输出由输入推导
// 校验fix_input1、fix_input2数据类型一致
REG_OP(FixIOOpWithT)
    .INPUT(fix_input1, "T")
    .INPUT(fix_input2, "T")
    .OUTPUT(fix_output, "T")
    .OP_END_FACTORY_REG(FixIOOpWithT);

// 输出由输入推导
// 校验fix_input1、fix_input2数据类型一致
// 校验fix_input1数据类型在range内
REG_OP(FixIOOpWithTRange)
    .INPUT(fix_input1, "T")
    .INPUT(fix_input2, "T")
    .OUTPUT(fix_output, "T")
    .DATATYPE(T, TensorType({DT_INT64, DT_INT32}))
    .OP_END_FACTORY_REG(FixIOOpWithTRange);

// 输出由输入推导
// 校验fix_input1、fix_input2数据类型一致
// 校验fix_input1数据类型在range内
REG_OP(FixIOOpWithTwoTRange)
    .INPUT(fix_input1, "T")
    .INPUT(fix_input2, "T2")
    .OUTPUT(fix_output, "T2")
    .DATATYPE(T, TensorType({DT_INT64, DT_INT32}))
    .DATATYPE(T2, TensorType({DT_FLOAT16, DT_BOOL}))
    .OP_END_FACTORY_REG(FixIOOpWithTwoTRange);

// 输出由输入推导
// 有可选输入的情况下，校验opt_input1在T1 rang内
// 校验fix_input1在range内
REG_OP(OptionalInputOpWithTRange2)
    .INPUT(fix_input1, "T")
    .OPTIONAL_INPUT(opt_input1, "T1")
    .OUTPUT(fix_output, "T")
    .DATATYPE(T, TensorType({DT_INT64, DT_INT32}))
    .DATATYPE(T1, TensorType({DT_INT64, DT_INT32, DT_BOOL}))
    .OP_END_FACTORY_REG(OptionalInputOpWithTRange2);

//===========================================================================================
// 固定输入固定输出-无range校验
TEST_F(UTInferDataType, infer_from_fix_input_without_range_success) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOpWithT");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  auto input_fix_1 = op_desc->MutableInputDesc("fix_input1");
  auto input_fix_2 = op_desc->MutableInputDesc("fix_input2");
  ASSERT_NE(input_fix_1, nullptr);
  ASSERT_NE(input_fix_2, nullptr);
  input_fix_1->SetDataType(DT_FLOAT16);
  input_fix_2->SetDataType(DT_FLOAT16);

  auto output_fix = op_desc->GetOutputDesc("fix_output");
  ASSERT_EQ(output_fix.GetDataType(), DT_FLOAT);
  ASSERT_EQ(op_desc->VerifyIR(), GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->DefaultInferDataType(), GRAPH_SUCCESS);
  output_fix = op_desc->GetOutputDesc("fix_output");
  ASSERT_EQ(output_fix.GetDataType(), DT_FLOAT16);
}

// 固定输入固定输出-无range校验-输入一致性校验失败
TEST_F(UTInferDataType, infer_from_fix_input_without_range_input_consistant_check_failed) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOpWithT");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  auto input_fix_1 = op_desc->MutableInputDesc("fix_input1");
  auto input_fix_2 = op_desc->MutableInputDesc("fix_input2");
  ASSERT_NE(input_fix_1, nullptr);
  ASSERT_NE(input_fix_2, nullptr);
  input_fix_1->SetDataType(DT_FLOAT16);
  input_fix_2->SetDataType(DT_INT32);

  auto output_fix = op_desc->GetOutputDesc("fix_output");
  ASSERT_EQ(output_fix.GetDataType(), DT_FLOAT);
  ASSERT_EQ(op_desc->VerifyIR(), GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->impl_->VerifyInputDataType(), GRAPH_PARAM_INVALID);
  ASSERT_EQ(op_desc->DefaultInferDataType(), GRAPH_PARAM_INVALID);
}

// 固定输入固定输出-有range校验
TEST_F(UTInferDataType, infer_from_fix_input_with_range_success) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOpWithTwoTRange");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  auto input_fix_1 = op_desc->MutableInputDesc("fix_input1");
  auto input_fix_2 = op_desc->MutableInputDesc("fix_input2");
  ASSERT_NE(input_fix_1, nullptr);
  ASSERT_NE(input_fix_2, nullptr);
  input_fix_1->SetDataType(DT_INT32);
  input_fix_2->SetDataType(DT_FLOAT16);

  auto output_fix = op_desc->GetOutputDesc("fix_output");
  ASSERT_EQ(output_fix.GetDataType(), DT_FLOAT);
  ASSERT_EQ(op_desc->VerifyIR(), GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->DefaultInferDataType(), GRAPH_SUCCESS);
  output_fix = op_desc->GetOutputDesc("fix_output");
  ASSERT_EQ(output_fix.GetDataType(), DT_FLOAT16);
}
// 固定输入固定输出-有range校验-输入1超range
// fix_input1 out of range
TEST_F(UTInferDataType, infer_from_fix_input_with_T_range_check_failed) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOpWithTwoTRange");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  auto input_fix_1 = op_desc->MutableInputDesc("fix_input1");
  auto input_fix_2 = op_desc->MutableInputDesc("fix_input2");
  ASSERT_NE(input_fix_1, nullptr);
  ASSERT_NE(input_fix_2, nullptr);
  input_fix_1->SetDataType(DT_FLOAT16);
  input_fix_2->SetDataType(DT_FLOAT16);

  auto output_fix = op_desc->GetOutputDesc("fix_output");
  ASSERT_EQ(output_fix.GetDataType(), DT_FLOAT);
  ASSERT_EQ(op_desc->VerifyIR(), GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->DefaultInferDataType(), GRAPH_PARAM_INVALID);
}
// 固定输入固定输出-有range校验-输入2超range
// fix_input2 out of range
TEST_F(UTInferDataType, infer_from_fix_input_with_T2_range_check_failed) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOpWithTwoTRange");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  auto input_fix_1 = op_desc->MutableInputDesc("fix_input1");
  auto input_fix_2 = op_desc->MutableInputDesc("fix_input2");
  ASSERT_NE(input_fix_1, nullptr);
  ASSERT_NE(input_fix_2, nullptr);
  input_fix_1->SetDataType(DT_INT32);
  input_fix_2->SetDataType(DT_INT32);

  auto output_fix = op_desc->GetOutputDesc("fix_output");
  ASSERT_EQ(output_fix.GetDataType(), DT_FLOAT);
  ASSERT_EQ(op_desc->VerifyIR(), GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->DefaultInferDataType(), GRAPH_PARAM_INVALID);
}

// 输出由输入推导
// 校验fix_input1和opt_input1数据类型一致，有可选输入的情况下
// 校验fix_input1在range内
REG_OP(OptionalInputOpWithTRange)
    .INPUT(fix_input1, "T")
    .OPTIONAL_INPUT(opt_input1, "T")
    .OUTPUT(fix_output, "T")
    .DATATYPE(T, TensorType({DT_INT64, DT_INT32}))
    .OP_END_FACTORY_REG(OptionalInputOpWithTRange);
// 固定输入+可选输入-有range校验
// 可选输入没连边
TEST_F(UTInferDataType, infer_from_fix_input_validate_optional_input) {
  auto op = OperatorFactory::CreateOperator("test1", "OptionalInputOpWithTRange");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  auto input_fix_1 = op_desc->MutableInputDesc("fix_input1");
  ASSERT_NE(input_fix_1, nullptr);
  input_fix_1->SetDataType(DT_INT32);

  auto output_fix = op_desc->GetOutputDesc("fix_output");
  ASSERT_EQ(output_fix.GetDataType(), DT_FLOAT);
  ASSERT_EQ(op_desc->VerifyIR(), GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->DefaultInferDataType(), GRAPH_SUCCESS);
  output_fix = op_desc->GetOutputDesc("fix_output");
  ASSERT_EQ(output_fix.GetDataType(), DT_INT32);
}

// 固定输入+可选输入-有range校验
TEST_F(UTInferDataType, infer_from_fix_input_validate_optional_input_consistant_failed) {
  auto builder = ut::GraphBuilder("root");
  const auto &input1 = builder.AddNode("data1", "Data", 1, 1);
  const auto &input2 = builder.AddNode("data2", "Data", 1, 1);
  auto graph = builder.GetGraph();
  auto op = OperatorFactory::CreateOperator("test1", "OptionalInputOpWithTRange");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  auto test_node = graph->AddNode(op_desc);
  builder.AddDataEdge(input1, 0, test_node, 0);
  builder.AddDataEdge(input2, 0, test_node, 1);

  auto input_fix_1 = op_desc->MutableInputDesc("fix_input1");
  auto input_opt_1 = op_desc->MutableInputDesc("opt_input1");
  ASSERT_NE(input_fix_1, nullptr);
  input_fix_1->SetDataType(DT_INT32);
  op_desc->UpdateInputDesc("opt_input1", GeTensorDesc(GeShape(), FORMAT_ND, DT_INT64));

  auto output_fix = op_desc->GetOutputDesc("fix_output");
  ASSERT_EQ(output_fix.GetDataType(), DT_FLOAT);
  ASSERT_EQ(op_desc->VerifyIR(), GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->impl_->VerifyInputDataType(), GRAPH_PARAM_INVALID);
  ASSERT_EQ(op_desc->DefaultInferDataType(), GRAPH_PARAM_INVALID);
}

// 输出由输入推导
// 校验fix_input1在T range内
// 校验dy_input1-n 数据类型一致
// 校验dy_input1 在T1 range内
REG_OP(DynamicInputOpWithT)
    .INPUT(fix_input1, "T")
    .DYNAMIC_INPUT(dy_input, "T1")
    .OUTPUT(fix_output, "T1")
    .DATATYPE(T, TensorType({DT_INT64, DT_INT32}))
    .DATATYPE(T1, TensorType({DT_INT64, DT_INT32, DT_BOOL}))
    .OP_END_FACTORY_REG(DynamicInputOpWithT);

// 固定输入+动态输入，输出由动态输入推导
TEST_F(UTInferDataType, infer_from_dynamic_input) {
  auto op = OperatorFactory::CreateOperator("test1", "DynamicInputOpWithT");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  op_desc->AddDynamicInputDesc("dy_input", 2, true);
  auto input_fix_1 = op_desc->MutableInputDesc("fix_input1");
  auto dy_input0 = op_desc->MutableInputDesc("dy_input0");
  auto dy_input1 = op_desc->MutableInputDesc("dy_input1");
  ASSERT_NE(input_fix_1, nullptr);
  ASSERT_NE(dy_input0, nullptr);
  ASSERT_NE(dy_input1, nullptr);
  input_fix_1->SetDataType(DT_INT32);
  dy_input0->SetDataType(DT_BOOL);
  dy_input1->SetDataType(DT_BOOL);

  auto output_fix = op_desc->GetOutputDesc("fix_output");
  ASSERT_EQ(output_fix.GetDataType(), DT_FLOAT);
  ASSERT_EQ(op_desc->VerifyIR(), GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->DefaultInferDataType(), GRAPH_SUCCESS);
  output_fix = op_desc->GetOutputDesc("fix_output");
  ASSERT_EQ(output_fix.GetDataType(), DT_BOOL);
}

// 固定输入+动态输入，输出由动态输入推导
TEST_F(UTInferDataType, infer_from_dynamic_input_consistant_check_failed) {
  auto op = OperatorFactory::CreateOperator("test1", "DynamicInputOpWithT");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  op_desc->AddDynamicInputDesc("dy_input", 2, true);
  auto input_fix_1 = op_desc->MutableInputDesc("fix_input1");
  auto dy_input0 = op_desc->MutableInputDesc("dy_input0");
  auto dy_input1 = op_desc->MutableInputDesc("dy_input1");
  ASSERT_NE(input_fix_1, nullptr);
  ASSERT_NE(dy_input0, nullptr);
  ASSERT_NE(dy_input1, nullptr);
  input_fix_1->SetDataType(DT_INT32);
  dy_input0->SetDataType(DT_BOOL);
  dy_input1->SetDataType(DT_INT32);

  ASSERT_EQ(op_desc->VerifyIR(), GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->impl_->VerifyInputDataType(), GRAPH_PARAM_INVALID);
  ASSERT_EQ(op_desc->DefaultInferDataType(), GRAPH_PARAM_INVALID);
}

// 输出由输入推导
// 校验fix_input1在T range内
// 校验dy_input1-n 数据类型一致
// 校验dy_input1 在T1 range内
REG_OP(DynamicInputOpWithListT)
    .DYNAMIC_INPUT(dy_input, "T")
    .DYNAMIC_OUTPUT(dy_output, "T")
    .DATATYPE(T, ListTensorType({DT_INT64, DT_INT32}))
    .OP_END_FACTORY_REG(DynamicInputOpWithListT);

TEST_F(UTInferDataType, infer_from_dynamic_input_list_tensor_type) {
  auto op = OperatorFactory::CreateOperator("test1", "DynamicInputOpWithListT");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  op_desc->AddDynamicInputDesc("dy_input", 2, true);
  op_desc->AddDynamicOutputDesc("dy_output", 2, true);
  auto dy_input0 = op_desc->MutableInputDesc("dy_input0");
  auto dy_input1 = op_desc->MutableInputDesc("dy_input1");
  ASSERT_NE(dy_input0, nullptr);
  ASSERT_NE(dy_input1, nullptr);
  dy_input0->SetDataType(DT_INT32);
  dy_input1->SetDataType(DT_INT64);

  auto dyn_output0 = op_desc->GetOutputDesc("dy_output0");
  ASSERT_EQ(dyn_output0.GetDataType(), DT_FLOAT);
  ASSERT_EQ(op_desc->VerifyIR(), GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->DefaultInferDataType(), GRAPH_SUCCESS);
  dyn_output0 = op_desc->GetOutputDesc("dy_output0");
  auto dyn_output1 = op_desc->GetOutputDesc("dy_output1");
  ASSERT_EQ(dyn_output0.GetDataType(), DT_INT32);
  ASSERT_EQ(dyn_output1.GetDataType(), DT_INT64);
}

// 输出由输入推导
// 校验fix_input1在T range内
// 校验dy_input1-n 数据类型一致
// 校验dy_input1 在T1 range内
REG_OP(DynamicInputOutputOpWithT)
    .DYNAMIC_INPUT(dy_input, "T")
    .DYNAMIC_OUTPUT(dy_output, "T")
    .OP_END_FACTORY_REG(DynamicInputOutputOpWithT);

TEST_F(UTInferDataType, infer_from_dynamic_input_without_list_tensor_type) {
  auto op = OperatorFactory::CreateOperator("test1", "DynamicInputOutputOpWithT");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  op_desc->AddDynamicInputDesc("dy_input", 2, true);
  op_desc->AddDynamicOutputDesc("dy_output", 2, true);
  auto dy_input0 = op_desc->MutableInputDesc("dy_input0");
  auto dy_input1 = op_desc->MutableInputDesc("dy_input1");
  ASSERT_NE(dy_input0, nullptr);
  ASSERT_NE(dy_input1, nullptr);
  dy_input0->SetDataType(DT_INT32);
  dy_input1->SetDataType(DT_INT32);

  auto dyn_output0 = op_desc->GetOutputDesc("dy_output0");
  ASSERT_EQ(dyn_output0.GetDataType(), DT_FLOAT);
  ASSERT_EQ(op_desc->VerifyIR(), GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->DefaultInferDataType(), GRAPH_SUCCESS);
  dyn_output0 = op_desc->GetOutputDesc("dy_output0");
  auto dyn_output1 = op_desc->GetOutputDesc("dy_output1");
  ASSERT_EQ(dyn_output0.GetDataType(), DT_INT32);
  ASSERT_EQ(dyn_output1.GetDataType(), DT_INT32);
}

// dynamic输入若无listT定义，则退化为T，需要横向校验一致性
TEST_F(UTInferDataType, infer_from_dynamic_input_without_list_tensor_type_consistant_check_fail) {
  auto op = OperatorFactory::CreateOperator("test1", "DynamicInputOutputOpWithT");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  op_desc->AddDynamicInputDesc("dy_input", 2, true);
  op_desc->AddDynamicOutputDesc("dy_output", 2, true);
  auto dy_input0 = op_desc->MutableInputDesc("dy_input0");
  auto dy_input1 = op_desc->MutableInputDesc("dy_input1");
  ASSERT_NE(dy_input0, nullptr);
  ASSERT_NE(dy_input1, nullptr);
  dy_input0->SetDataType(DT_INT32);
  dy_input1->SetDataType(DT_INT64);

  auto dyn_output0 = op_desc->GetOutputDesc("dy_output0");
  ASSERT_EQ(dyn_output0.GetDataType(), DT_FLOAT);
  ASSERT_EQ(op_desc->VerifyIR(), GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->impl_->VerifyInputDataType(), GRAPH_PARAM_INVALID);
  ASSERT_EQ(op_desc->DefaultInferDataType(), GRAPH_PARAM_INVALID);
}

//================================================================================================
// infer from attr
REG_OP(FixIOOpWithAttr)
    .INPUT(fix_input1, "T")
    .REQUIRED_ATTR(dst_type, Int)
    .OUTPUT(fix_output1, "dst_type")
    .DATATYPE(dst_type, TensorType({DT_BOOL, DT_INT64}))
    .DATATYPE(T, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(FixIOOpWithAttr);

TEST_F(UTInferDataType, infer_from_attr_success) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOpWithAttr");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  auto input_fix_1 = op_desc->MutableInputDesc("fix_input1");
  AttrUtils::SetInt(op_desc, "dst_type", 12);
  ASSERT_NE(input_fix_1, nullptr);
  input_fix_1->SetDataType(DT_INT32);

  auto output_fix = op_desc->GetOutputDesc("fix_output1");
  ASSERT_EQ(output_fix.GetDataType(), DT_FLOAT);
  ASSERT_EQ(op_desc->VerifyIR(), GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->DefaultInferDataType(), GRAPH_SUCCESS);
  output_fix = op_desc->GetOutputDesc("fix_output1");
  ASSERT_EQ(output_fix.GetDataType(), DT_BOOL);
}

TEST_F(UTInferDataType, infer_from_attr_check_attr_out_of_range) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOpWithAttr");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  auto input_fix_1 = op_desc->MutableInputDesc("fix_input1");
  AttrUtils::SetInt(op_desc, "dst_type", 15);
  ASSERT_NE(input_fix_1, nullptr);
  input_fix_1->SetDataType(DT_INT32);

  auto output_fix = op_desc->GetOutputDesc("fix_output1");
  ASSERT_EQ(output_fix.GetDataType(), DT_FLOAT);
  ASSERT_EQ(op_desc->VerifyIR(), GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->impl_->VerifyInputDataType(), GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->DefaultInferDataType(), GRAPH_PARAM_INVALID);
}

TEST_F(UTInferDataType, infer_from_attr_check_input_out_of_range) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOpWithAttr");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  auto input_fix_1 = op_desc->MutableInputDesc("fix_input1");
  AttrUtils::SetInt(op_desc, "dst_type", 12);
  ASSERT_NE(input_fix_1, nullptr);
  input_fix_1->SetDataType(DT_BOOL);

  ASSERT_EQ(op_desc->VerifyIR(), GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->impl_->VerifyInputDataType(), GRAPH_PARAM_INVALID);
  ASSERT_EQ(op_desc->DefaultInferDataType(), GRAPH_PARAM_INVALID);
}
//================================================================================================
// infer from output
REG_OP(FixIOOp_OutputIsFix)
    .INPUT(fix_input1, "T")
    .INPUT(fix_input2, "T")
    .OUTPUT(fix_output, "T2")
    .DATATYPE(T2, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(FixIOOp_OutputIsFix);
TEST_F(UTInferDataType, infer_from_output_success) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOp_OutputIsFix");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  auto fix_input1 = op_desc->MutableInputDesc("fix_input1");
  auto fix_input2 = op_desc->MutableInputDesc("fix_input2");
  ASSERT_NE(fix_input1, nullptr);
  ASSERT_NE(fix_input2, nullptr);
  fix_input1->SetDataType(DT_INT32);
  fix_input2->SetDataType(DT_INT32);

  auto output_fix = op_desc->GetOutputDesc("fix_output");
  ASSERT_EQ(output_fix.GetDataType(), DT_FLOAT);
  ASSERT_EQ(op_desc->VerifyIR(), GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->DefaultInferDataType(), GRAPH_SUCCESS);
  output_fix = op_desc->GetOutputDesc("fix_output");
  ASSERT_EQ(output_fix.GetDataType(), DT_BOOL);
}

//================================================================================================
// validate IR
// 输出由输入推导
REG_OP(FixIOOp_OutputNoMapping)
    .INPUT(fix_input1, "T")
    .INPUT(fix_input2, "T")
    .OUTPUT(fix_output, "T2")
    .OP_END_FACTORY_REG(FixIOOp_OutputNoMapping);

TEST_F(UTInferDataType, validate_ir_output_no_mapping) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOp_OutputNoMapping");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  auto input_fix_1 = op_desc->MutableInputDesc("fix_input1");
  auto input_fix_2 = op_desc->MutableInputDesc("fix_input2");
  ASSERT_NE(input_fix_1, nullptr);
  ASSERT_NE(input_fix_2, nullptr);
  input_fix_1->SetDataType(DT_FLOAT16);
  input_fix_2->SetDataType(DT_FLOAT16);

  auto output_fix = op_desc->GetOutputDesc("fix_output");

  ASSERT_EQ(output_fix.GetDataType(), DT_FLOAT);
  ASSERT_EQ(op_desc->VerifyIR(), GRAPH_INVALID_IR_DEF);
  ASSERT_EQ(op_desc->DefaultInferDataType(), GRAPH_INVALID_IR_DEF);
}

REG_OP(DynamicIOOp_OutputNoMapping)
    .INPUT(fix_input1, "T")
    .DYNAMIC_INPUT(dy_input2, "T")
    .DYNAMIC_OUTPUT(dy_output, "T2")
    .OP_END_FACTORY_REG(DynamicIOOp_OutputNoMapping);
TEST_F(UTInferDataType, validate_ir_dynamic_output_no_mapping) {
  auto op = OperatorFactory::CreateOperator("test1", "DynamicIOOp_OutputNoMapping");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  auto input_fix_1 = op_desc->MutableInputDesc("fix_input1");
  ASSERT_NE(input_fix_1, nullptr);
  input_fix_1->SetDataType(DT_FLOAT16);

  ASSERT_EQ(op_desc->VerifyIR(), GRAPH_INVALID_IR_DEF);
  ASSERT_EQ(op_desc->DefaultInferDataType(), GRAPH_INVALID_IR_DEF);
}

REG_OP(FixDynamicIOOp_T_conflict)
    .INPUT(fix_input1, "T")
    .DYNAMIC_INPUT(dy_input, "T")
    .DYNAMIC_INPUT(dy_output, "T")
    .DATATYPE(T, ListTensorType(TensorType::ALL()))
    .OP_END_FACTORY_REG(FixDynamicIOOp_T_conflict);
TEST_F(UTInferDataType, validate_ir_fix_input_on_list_tensor_type) {
  auto op = OperatorFactory::CreateOperator("test1", "FixDynamicIOOp_T_conflict");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  op_desc->AddDynamicInputDesc("dy_input", 2, true);
  op_desc->AddDynamicOutputDesc("dy_output", 2, true);
  auto input_fix_1 = op_desc->MutableInputDesc("fix_input1");
  auto dy_input0 = op_desc->MutableInputDesc("dy_input0");
  auto dy_input1 = op_desc->MutableInputDesc("dy_input1");
  ASSERT_NE(input_fix_1, nullptr);
  ASSERT_NE(dy_input0, nullptr);
  ASSERT_NE(dy_input1, nullptr);
  input_fix_1->SetDataType(DT_INT32);
  dy_input0->SetDataType(DT_BOOL);
  dy_input1->SetDataType(DT_BOOL);

  auto dy_output0 = op_desc->GetOutputDesc("dy_output0");
  auto dy_output1 = op_desc->GetOutputDesc("dy_output1");
  ASSERT_EQ(dy_output0.GetDataType(), DT_FLOAT);
  ASSERT_EQ(op_desc->VerifyIR(), GRAPH_INVALID_IR_DEF);
  ASSERT_EQ(op_desc->DefaultInferDataType(), GRAPH_INVALID_IR_DEF);
}
}  // namespace ge
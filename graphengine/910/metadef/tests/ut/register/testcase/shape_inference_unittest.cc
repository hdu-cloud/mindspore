/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "graph/operator_factory_impl.h"
#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"
#include "register/shape_inference.h"
#include "utils/op_desc_utils.h"
#include <gtest/gtest.h>
#include "graph/utils/graph_utils.h"
#include "graph/attr_value.h"
#include "external/graph/operator_factory.h"
#include "register/op_impl_space_registry.h"
#include "register/op_impl_registry_holder_manager.h"
#include "common/ge_common/ge_inner_error_codes.h"

namespace ge{
REG_OP(Const)
    .OUTPUT(y,
            TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                        DT_UINT64, DT_BOOL, DT_DOUBLE}))
        .ATTR(value, Tensor, Tensor())
        .OP_END_FACTORY_REG(Const);
}
namespace gert {
using namespace ge;
class ShapeInferenceUT : public testing::Test {};
// infer from output
REG_OP(FixIOOp_OutputIsFix)
    .INPUT(fix_input1, "T")
    .INPUT(fix_input2, "T")
    .OUTPUT(fix_output, "T2")
    .DATATYPE(T2, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(FixIOOp_OutputIsFix);
// 无可选输入，无动态输入，正常流程，infer shape & infer data type
TEST_F(ShapeInferenceUT, CallInferV2Func_success) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOp_OutputIsFix");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1, 1, 1, 1});
  GeTensorDesc tensor_desc(shape, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{0, 10000}};
  tensor_desc.SetOriginShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc);
  op_desc->UpdateInputDesc(1, tensor_desc);
  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    const auto input_shape = context->GetInputShape(0U);
    auto output = context->GetOutputShape(0);
    for (size_t dim = 0UL; dim < input_shape->GetDimNum(); dim++) {
      output->AppendDim(input_shape->GetDim(dim));
    }
    output->SetDimNum(input_shape->GetDimNum());
    return GRAPH_SUCCESS;
  };
  const auto infer_data_type_func = [](gert::InferDataTypeContext *context) -> graphStatus {
    const auto date_type = context->GetInputDataType(0U);
    EXPECT_EQ(context->SetOutputDataType(0, date_type), SUCCESS);
    return GRAPH_SUCCESS;
  };
  const auto infer_shape_range_func = [](gert::InferShapeRangeContext *context) -> graphStatus {
    auto input_shape_range = context->GetInputShapeRange(0U);
    auto output_shape_range = context->GetOutputShapeRange(0U);
    output_shape_range->SetMin(const_cast<gert::Shape *>(input_shape_range->GetMin()));
    output_shape_range->SetMax(const_cast<gert::Shape *>(input_shape_range->GetMax()));
    return GRAPH_SUCCESS;
  };
  IMPL_OP(FixIOOp_OutputIsFix).InferShape(infer_shape_func)
      .InferDataType(infer_data_type_func)
      .InferShapeRange(infer_shape_range_func);

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistry>();
  auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
  gert::OpImplKernelRegistry::OpImplFunctions op_impl_func;
  op_impl_func.infer_shape = infer_shape_func;
  op_impl_func.infer_datatype = infer_data_type_func;
  op_impl_func.infer_shape_range = infer_shape_range_func;
  registry_holder->AddTypesToImpl("FixIOOp_OutputIsFix", op_impl_func);
  space_registry->AddRegistry(registry_holder);
  DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(space_registry);

  const auto call_infer_data_type = OperatorFactoryImpl::GetInferDataTypeFunc();
  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();

  const auto call_infer_shape_range = OperatorFactoryImpl::GetInferShapeRangeFunc();
  ASSERT_NE(call_infer_data_type, nullptr);
  ASSERT_NE(call_infer_shape_v2, nullptr);
  ASSERT_NE(call_infer_shape_range, nullptr);
  auto status = call_infer_data_type(op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  status = call_infer_shape_v2(op, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  status = call_infer_shape_range(op, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetDataType(), DT_FLOAT16);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 4);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDim(0), 1);
}

REG_OP(OptionalInput3Input3Output)
    .INPUT(input1, "T")
    .OPTIONAL_INPUT(input2, "T")
    .INPUT(input3, "T")
    .OUTPUT(output1, "T2")
    .OUTPUT(output2, "T2")
    .OUTPUT(output3, "T2")
    .DATATYPE(T2, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(OptionalInput3Input3Output);
// 未实例化的optional input测试
TEST_F(ShapeInferenceUT, CallInferV2Func_OptionalInputWithOutInstance) {
  auto op = OperatorFactory::CreateOperator("test2", "OptionalInput3Input3Output");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  // input1
  GeShape shape1({1, 2, 3, 4});
  GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  // input3
  GeShape shape2({4, 3, 2});
  GeTensorDesc tensor_desc2(shape2, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);
  tensor_desc2.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(2, tensor_desc2);
  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    const auto option_input_shape = context->GetOptionalInputShape(1U);
    if (option_input_shape != nullptr) {
      return GRAPH_FAILED;
    }
    auto output = context->GetOutputShape(0);
    const auto input_shape = context->GetInputShape(1U);
    for (size_t dim = 0UL; dim < input_shape->GetDimNum(); dim++) {
      output->AppendDim(input_shape->GetDim(dim));
    }
    output->SetDimNum(input_shape->GetDimNum());
    return GRAPH_SUCCESS;
  };
  IMPL_OP(OptionalInput3Input3Output).InferShape(infer_shape_func)
      .InferDataType(nullptr)
      .InferShapeRange(nullptr);

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistry>();
  auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
  gert::OpImplKernelRegistry::OpImplFunctions op_impl_func;
  op_impl_func.infer_shape = infer_shape_func;
  op_impl_func.infer_shape_range = nullptr;
  registry_holder->AddTypesToImpl("OptionalInput3Input3Output", op_impl_func);
  space_registry->AddRegistry(registry_holder);
  DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(space_registry);

  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  ASSERT_NE(call_infer_shape_v2, nullptr);
  auto status = call_infer_shape_v2(op, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 3);
  const auto call_infer_shape_range = OperatorFactoryImpl::GetInferShapeRangeFunc();
  status = call_infer_shape_range(op, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
}

// 实例化的optional input测试
TEST_F(ShapeInferenceUT, CallInferV2Func_OptionalInputWithInstance) {
  auto op = OperatorFactory::CreateOperator("test3", "OptionalInput3Input3Output");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  // input1
  GeShape shape1({1, 2, 3, 4});
  GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  // input2
  GeShape shape2({4, 3, 2});
  GeTensorDesc tensor_desc2(shape2, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);
  tensor_desc2.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(1, tensor_desc2);
  // input3
  GeShape shape3({4, 3});
  GeTensorDesc tensor_desc3(shape3, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc3.SetOriginShape(shape3);
  tensor_desc3.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(2, tensor_desc3);
  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    // update option input to output0
    const auto input_shape = context->GetOptionalInputShape(1U);
    auto output = context->GetOutputShape(0);
    for (size_t dim = 0UL; dim < input_shape->GetDimNum(); dim++) {
      output->AppendDim(input_shape->GetDim(dim));
    }
    output->SetDimNum(input_shape->GetDimNum());
    // update input3 to output2
    const auto input_shape2 = context->GetInputShape(2U);
    auto output2 = context->GetOutputShape(1);
    for (size_t dim = 0UL; dim < input_shape2->GetDimNum(); dim++) {
      output2->AppendDim(input_shape2->GetDim(dim));
    }
    output2->SetDimNum(input_shape2->GetDimNum());
    return GRAPH_SUCCESS;
  };
  IMPL_OP(OptionalInput3Input3Output).InferShape(infer_shape_func)
      .InferDataType(nullptr)
      .InferShapeRange(nullptr);

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistry>();
  auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
  gert::OpImplKernelRegistry::OpImplFunctions op_impl_func;
  op_impl_func.infer_shape = infer_shape_func;
  registry_holder->AddTypesToImpl("OptionalInput3Input3Output", op_impl_func);
  space_registry->AddRegistry(registry_holder);
  DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(space_registry);

  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  ASSERT_NE(call_infer_shape_v2, nullptr);
  const auto status = call_infer_shape_v2(op, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 3);
  ASSERT_EQ(op_desc->GetOutputDesc(1U).GetShape().GetDimNum(), 2);
}


// 动态输入的input测试
REG_OP(DynamicInput3Input3Output3)
    .INPUT(input1, "T")
    .DYNAMIC_INPUT(dyn_input, "D")
        .INPUT(input3, "T")
        .OUTPUT(output1, "T2")
        .OUTPUT(output2, "T2")
        .OUTPUT(output3, "T2")
        .DATATYPE(T2, TensorType({DT_BOOL}))
        .OP_END_FACTORY_REG(DynamicInput3Input3Output3);
const auto INFER_SHAPE_FUNC = [](gert::InferShapeContext *context) -> graphStatus {
  // update input3 input to output0
  const auto input_shape = context->GetInputShape(1U);
  auto output = context->GetOutputShape(0);
  for (size_t dim = 0UL; dim < input_shape->GetDimNum(); dim++) {
    output->AppendDim(input_shape->GetDim(dim));
  }
  output->SetDimNum(input_shape->GetDimNum());
  // update dyn_input_0 to output1, dyn_input_1 to output2
  const auto input_shape2 = context->GetInputShape(2U);
  auto output2 = context->GetOutputShape(1);
  for (size_t dim = 0UL; dim < input_shape2->GetDimNum(); dim++) {
    output2->AppendDim(input_shape2->GetDim(dim));
  }
  output2->SetDimNum(input_shape2->GetDimNum());

  const auto input_shape3 = context->GetInputShape(3U);
  auto output3 = context->GetOutputShape(2);
  for (size_t dim = 0UL; dim < input_shape3->GetDimNum(); dim++) {
    output3->AppendDim(input_shape3->GetDim(dim));
  }
  output3->SetDimNum(input_shape3->GetDimNum());
  return GRAPH_SUCCESS;
};
TEST_F(ShapeInferenceUT, CallInferV2Func_DynamicInput) {
  auto operator_dynamic = op::DynamicInput3Input3Output3("test4");
  operator_dynamic.create_dynamic_input_byindex_dyn_input(2, true);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(operator_dynamic);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllInputsSize(), 4);
  // input1
  GeShape shape1({1, 2, 3, 4});
  GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  // input3
  GeShape shape2({4, 3, 2});
  GeTensorDesc tensor_desc2(shape2, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);
  tensor_desc2.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(1, tensor_desc2);
  // dynamic input
  GeShape shape3({4, 3});
  GeTensorDesc tensor_desc3(shape3, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc3.SetOriginShape(shape3);
  tensor_desc3.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(2, tensor_desc3);
  op_desc->UpdateInputDesc(3, tensor_desc1);
  IMPL_OP(DynamicInput3Input3Output3).InferShape(INFER_SHAPE_FUNC)
      .InferDataType(nullptr)
      .InferShapeRange(nullptr);

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistry>();
  auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
  gert::OpImplKernelRegistry::OpImplFunctions op_impl_func;
  op_impl_func.infer_shape = INFER_SHAPE_FUNC;
  op_impl_func.infer_shape_range = nullptr;
  registry_holder->AddTypesToImpl("DynamicInput3Input3Output3", op_impl_func);
  space_registry->AddRegistry(registry_holder);
  DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(space_registry);

  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  ASSERT_NE(call_infer_shape_v2, nullptr);
  auto status = call_infer_shape_v2(operator_dynamic, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 3);
  ASSERT_EQ(op_desc->GetOutputDesc(1U).GetShape().GetDimNum(), 2);
  ASSERT_EQ(op_desc->GetOutputDesc(2U).GetShape().GetDimNum(), 4);
  const auto call_infer_shape_range = OperatorFactoryImpl::GetInferShapeRangeFunc();
  status = call_infer_shape_range(operator_dynamic, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
}

// 动态输入的input测试 动态轴-2
TEST_F(ShapeInferenceUT, CallInferV2Func_DynamicInput_unknow_2) {
  auto operator_dynamic = op::DynamicInput3Input3Output3("test4");
  operator_dynamic.create_dynamic_input_byindex_dyn_input(2, true);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(operator_dynamic);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllInputsSize(), 4);
  // input1
  GeShape shape1({-2});
  GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  // input3
  GeShape shape2({4, 3, 2});
  GeTensorDesc tensor_desc2(shape2, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);
  tensor_desc2.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(1, tensor_desc2);
  // dynamic input
  GeShape shape3({4, 3});
  GeTensorDesc tensor_desc3(shape3, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc3.SetOriginShape(shape3);
  tensor_desc3.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(2, tensor_desc3);
  op_desc->UpdateInputDesc(3, tensor_desc1);
  IMPL_OP(DynamicInput3Input3Output3).InferShape(INFER_SHAPE_FUNC)
    .InferDataType(nullptr)
    .InferShapeRange(nullptr);

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistry>();
  auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
  gert::OpImplKernelRegistry::OpImplFunctions op_impl_func;
  op_impl_func.infer_shape = INFER_SHAPE_FUNC;
  op_impl_func.infer_shape_range = nullptr;
  registry_holder->AddTypesToImpl("DynamicInput3Input3Output3", op_impl_func);
  space_registry->AddRegistry(registry_holder);
  DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(space_registry);

  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  ASSERT_NE(call_infer_shape_v2, nullptr);
  auto status = call_infer_shape_v2(operator_dynamic, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 3);
  ASSERT_EQ(op_desc->GetOutputDesc(1U).GetShape().GetDimNum(), 2);
  ASSERT_EQ(op_desc->GetOutputDesc(2U).GetShape().GetDimNum(), 0);
  const auto call_infer_shape_range = OperatorFactoryImpl::GetInferShapeRangeFunc();
  status = call_infer_shape_range(operator_dynamic, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
}

// 动态输入的input测试 动态轴-1, shape range 不设值
TEST_F(ShapeInferenceUT, CallInferV2Func_DynamicInput_unknow_no_shaperange) {
  auto operator_dynamic = op::DynamicInput3Input3Output3("test4");
  operator_dynamic.create_dynamic_input_byindex_dyn_input(2, true);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(operator_dynamic);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllInputsSize(), 4);
  // input1
  GeShape shape1({1, 2, 3, -1});
  GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  // input3
  GeShape shape2({4, 3, 2});
  GeTensorDesc tensor_desc2(shape2, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);
  tensor_desc2.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(1, tensor_desc2);
  // dynamic input
  GeShape shape3({4, 3});
  GeTensorDesc tensor_desc3(shape3, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc3.SetOriginShape(shape3);
  tensor_desc3.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(2, tensor_desc3);
  op_desc->UpdateInputDesc(3, tensor_desc1);
  IMPL_OP(DynamicInput3Input3Output3).InferShape(INFER_SHAPE_FUNC)
      .InferDataType(nullptr)
      .InferShapeRange(nullptr);

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistry>();
  auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
  gert::OpImplKernelRegistry::OpImplFunctions op_impl_func;
  op_impl_func.infer_shape = INFER_SHAPE_FUNC;
  op_impl_func.infer_shape_range = nullptr;
  registry_holder->AddTypesToImpl("DynamicInput3Input3Output3", op_impl_func);
  space_registry->AddRegistry(registry_holder);
  DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(space_registry);

  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  ASSERT_NE(call_infer_shape_v2, nullptr);
  auto status = call_infer_shape_v2(operator_dynamic, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 3);
  ASSERT_EQ(op_desc->GetOutputDesc(1U).GetShape().GetDimNum(), 2);
  ASSERT_EQ(op_desc->GetOutputDesc(2U).GetShape().GetDimNum(), 4);
  const auto call_infer_shape_range = OperatorFactoryImpl::GetInferShapeRangeFunc();
  status = call_infer_shape_range(operator_dynamic, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
}

// 动态输入的input测试 动态轴-1, shape range 设值
TEST_F(ShapeInferenceUT, CallInferV2Func_DynamicInput_unknow_shaperange) {
  auto operator_dynamic = op::DynamicInput3Input3Output3("test4");
  operator_dynamic.create_dynamic_input_byindex_dyn_input(2, true);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(operator_dynamic);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllInputsSize(), 4);
  // input1
  GeShape shape1({1, 2, 3, -1});
  GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{1, 1}, {2, 2}, {3, 3}, {22, 999}};
  tensor_desc1.SetShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  // input3
  GeShape shape2({4, 3, 2});
  GeTensorDesc tensor_desc2(shape2, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);
  tensor_desc2.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(1, tensor_desc2);
  // dynamic input
  GeShape shape3({4, 3});
  GeTensorDesc tensor_desc3(shape3, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc3.SetOriginShape(shape3);
  tensor_desc3.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(2, tensor_desc3);
  op_desc->UpdateInputDesc(3, tensor_desc1);
  IMPL_OP(DynamicInput3Input3Output3).InferShape(INFER_SHAPE_FUNC)
      .InferDataType(nullptr)
      .InferShapeRange(nullptr);

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistry>();
  auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
  gert::OpImplKernelRegistry::OpImplFunctions op_impl_func;
  op_impl_func.infer_shape = INFER_SHAPE_FUNC;
  op_impl_func.infer_shape_range = nullptr;
  registry_holder->AddTypesToImpl("DynamicInput3Input3Output3", op_impl_func);
  space_registry->AddRegistry(registry_holder);
  DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(space_registry);

  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  ASSERT_NE(call_infer_shape_v2, nullptr);
  auto status = call_infer_shape_v2(operator_dynamic, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 3);
  ASSERT_EQ(op_desc->GetOutputDesc(1U).GetShape().GetDimNum(), 2);
  ASSERT_EQ(op_desc->GetOutputDesc(2U).GetShape().GetDimNum(), 4);
  const auto call_infer_shape_range = OperatorFactoryImpl::GetInferShapeRangeFunc();
  status = call_infer_shape_range(operator_dynamic, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  (void)op_desc->GetOutputDesc(2U).GetShapeRange(shape_range);
  ASSERT_EQ(shape_range.size(), 4U);
  for (size_t i = 0UL; i < shape_range.size(); ++i) {
    ASSERT_EQ(shape_range[i].first, range[i].first);
    ASSERT_EQ(shape_range[i].second, range[i].second);
  }
}

// 动态输入的input测试 动态轴-1, shape range 设值,min大于max异常场景
TEST_F(ShapeInferenceUT, CallInferV2Func_DynamicInput_unknow_shaperange_min_bigger_max) {
  auto operator_dynamic = op::DynamicInput3Input3Output3("test4");
  operator_dynamic.create_dynamic_input_byindex_dyn_input(2, true);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(operator_dynamic);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllInputsSize(), 4);
  // input1
  GeShape shape1({1, 2, 3, -1});
  GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{1, 1}, {2, 2}, {3, 3}, {999, 22}};
  tensor_desc1.SetShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  // input3
  GeShape shape2({4, 3, 2});
  GeTensorDesc tensor_desc2(shape2, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);
  tensor_desc2.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(1, tensor_desc2);
  // dynamic input
  GeShape shape3({4, 3});
  GeTensorDesc tensor_desc3(shape3, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc3.SetOriginShape(shape3);
  tensor_desc3.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(2, tensor_desc3);
  op_desc->UpdateInputDesc(3, tensor_desc1);
  IMPL_OP(DynamicInput3Input3Output3).InferShape(INFER_SHAPE_FUNC)
    .InferDataType(nullptr)
    .InferShapeRange(nullptr);

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistry>();
  auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
  gert::OpImplKernelRegistry::OpImplFunctions op_impl_func;
  op_impl_func.infer_shape = INFER_SHAPE_FUNC;
  op_impl_func.infer_shape_range = nullptr;
  registry_holder->AddTypesToImpl("DynamicInput3Input3Output3", op_impl_func);
  space_registry->AddRegistry(registry_holder);
  DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(space_registry);

  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  ASSERT_NE(call_infer_shape_v2, nullptr);
  auto status = call_infer_shape_v2(operator_dynamic, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 3);
  ASSERT_EQ(op_desc->GetOutputDesc(1U).GetShape().GetDimNum(), 2);
  ASSERT_EQ(op_desc->GetOutputDesc(2U).GetShape().GetDimNum(), 4);
  const auto call_infer_shape_range = OperatorFactoryImpl::GetInferShapeRangeFunc();
  status = call_infer_shape_range(operator_dynamic, op_desc);
  ASSERT_EQ(status, ge::PARAM_INVALID);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  (void)op_desc->GetOutputDesc(2U).GetShapeRange(shape_range);
  ASSERT_EQ(shape_range.size(), 0U);
}

// 动态输入的input测试 动态轴-1, shape range 设值, min大于max, max为-1的正常场景
TEST_F(ShapeInferenceUT, CallInferV2Func_DynamicInput_unknow_shaperange_min_bigger_max_success) {
  auto operator_dynamic = op::DynamicInput3Input3Output3("test4");
  operator_dynamic.create_dynamic_input_byindex_dyn_input(2, true);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(operator_dynamic);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllInputsSize(), 4);
  // input1
  GeShape shape1({1, 2, 3, -1});
  GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{1, -1}, {2, -1}, {3, -1}, {999, -1}};
  tensor_desc1.SetShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  // input3
  GeShape shape2({4, 3, 2});
  GeTensorDesc tensor_desc2(shape2, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);
  tensor_desc2.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(1, tensor_desc2);
  // dynamic input
  GeShape shape3({4, 3});
  GeTensorDesc tensor_desc3(shape3, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc3.SetOriginShape(shape3);
  tensor_desc3.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(2, tensor_desc3);
  op_desc->UpdateInputDesc(3, tensor_desc1);
  IMPL_OP(DynamicInput3Input3Output3).InferShape(INFER_SHAPE_FUNC)
    .InferDataType(nullptr)
    .InferShapeRange(nullptr);

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistry>();
  auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
  gert::OpImplKernelRegistry::OpImplFunctions op_impl_func;
  op_impl_func.infer_shape = INFER_SHAPE_FUNC;
  op_impl_func.infer_shape_range = nullptr;
  registry_holder->AddTypesToImpl("DynamicInput3Input3Output3", op_impl_func);
  space_registry->AddRegistry(registry_holder);
  DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(space_registry);

  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  ASSERT_NE(call_infer_shape_v2, nullptr);
  auto status = call_infer_shape_v2(operator_dynamic, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 3);
  ASSERT_EQ(op_desc->GetOutputDesc(1U).GetShape().GetDimNum(), 2);
  ASSERT_EQ(op_desc->GetOutputDesc(2U).GetShape().GetDimNum(), 4);
  const auto call_infer_shape_range = OperatorFactoryImpl::GetInferShapeRangeFunc();
  status = call_infer_shape_range(operator_dynamic, op_desc);
  ASSERT_EQ(status, ge::GRAPH_SUCCESS);
}

// 二类算子值依赖测试
REG_OP(Type2_1Input_1Output)
    .INPUT(input1, "T")
        .OPTIONAL_INPUT(input2, "T")
        .INPUT(input3, "T")
        .OUTPUT(output1, "T2")
        .DATATYPE(T2, TensorType({DT_BOOL}))
        .OP_END_FACTORY_REG(Type2_1Input_1Output);
TEST_F(ShapeInferenceUT, CallInferV2Func_Type2ValueDepend) {
  // construct const input
  auto const_input = ge::op::Const("const_input");
  ge::TensorDesc td{ge::Shape(std::vector<int64_t>({1, 2, 3, 4})), FORMAT_NCHW, DT_UINT8};
  ge::Tensor tensor(td);
  std::vector<uint8_t> val{0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58};
  tensor.SetData(val);
  const_input.set_attr_value(tensor);
  // const input link to op
  auto op = op::Type2_1Input_1Output("test5");
  op.set_input_input1(const_input);
  op.set_input_input3(const_input);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllInputsSize(), 3);
  // input1
  ge::GeShape shape1({1, 2, 3, 5});
  ge::GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  op_desc->UpdateInputDesc(2, tensor_desc1);
  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    // update input3(因为option输入未实例化，所以是第二个) value to output0
    const auto data = context->GetInputTensor(1U)->GetData<uint8_t>();
    std::vector<int64_t> dims = {data[0], data[1], data[2], data[3]};
    ge::Shape input_shape(dims);
    auto output = context->GetOutputShape(0);
    for (size_t dim = 0UL; dim < input_shape.GetDimNum(); dim++) {
      output->AppendDim(input_shape.GetDim(dim));
    }
    output->SetDimNum(input_shape.GetDimNum());
    return GRAPH_SUCCESS;
  };
  IMPL_OP(Type2_1Input_1Output).InferShape(infer_shape_func).InputsDataDependency({2})
      .InferDataType(nullptr)
      .InferShapeRange(nullptr);

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistry>();
  auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
  gert::OpImplKernelRegistry::OpImplFunctions op_impl_func;
  op_impl_func.infer_shape = infer_shape_func;
  op_impl_func.SetInputDataDependency(2);
  registry_holder->AddTypesToImpl("Type2_1Input_1Output", op_impl_func);
  space_registry->AddRegistry(registry_holder);
  DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(space_registry);

  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  ASSERT_NE(call_infer_shape_v2, nullptr);
  const auto status = call_infer_shape_v2(op, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  const auto &shape = op_desc->GetOutputDesc(0U).GetShape();
  ASSERT_EQ(shape.GetDimNum(), 4);
  ASSERT_EQ(shape.GetDim(0U), 85);
  ASSERT_EQ(shape.GetDim(1U), 86);
  ASSERT_EQ(shape.GetDim(2U), 87);
  ASSERT_EQ(shape.GetDim(3U), 88);
}

// 二类算子值依赖测试,带shape range
REG_OP(Type2_3Input_2Output)
  .INPUT(input1, "T")
  .OPTIONAL_INPUT(input2, "T")
  .INPUT(input3, "T")
  .OUTPUT(output1, "T2")
  .OUTPUT(output2, "T2")
  .DATATYPE(T2, TensorType({DT_BOOL}))
  .OP_END_FACTORY_REG(Type2_3Input_2Output);
TEST_F(ShapeInferenceUT, CallInferV2Func_Type2ValueDepend_unknow_shaperange) {
  // construct const input
  auto const_input = ge::op::Const("const_input");
  ge::TensorDesc td{ge::Shape(std::vector<int64_t>({1, 2, 3, 4})), FORMAT_NCHW, DT_UINT8};
  ge::Tensor tensor(td);
  std::vector<uint8_t> val{0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58,
                           0x55, 0x56, 0x57, 0x58, 0x58, 0x58};
  tensor.SetData(val);
  const_input.set_attr_value(tensor);
  // const input link to op
  auto op = op::Type2_3Input_2Output("test5");
  op.set_input_input1(const_input);
  op.set_input_input3(const_input);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllInputsSize(), 3);
  // input1
  GeShape shape1({1, 2, 3, -1});
  GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{1, 1}, {2, 2}, {3, 3}, {22, 999}};
  tensor_desc1.SetShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  // input3
  ge::GeShape shape3({1, 2, 3, 5});
  ge::GeTensorDesc tensor_desc3(shape3, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc3.SetOriginShape(shape3);
  tensor_desc3.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(2, tensor_desc3);
  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    // update input3(因为option输入未实例化，所以是第二个) value to output0
    const auto data = context->GetInputTensor(1U)->GetData<uint8_t>();
    std::vector<int64_t> dims = {data[0], data[1], data[2], data[3]};
    ge::Shape input_shape(dims);
    auto output = context->GetOutputShape(0);
    for (size_t dim = 0UL; dim < input_shape.GetDimNum(); dim++) {
      output->AppendDim(input_shape.GetDim(dim));
    }
    output->SetDimNum(input_shape.GetDimNum());

    const auto input_shape1 = context->GetInputShape(0U);
    auto output1 = context->GetOutputShape(1);
    for (size_t dim = 0UL; dim < input_shape1->GetDimNum(); dim++) {
      output1->AppendDim(input_shape1->GetDim(dim));
    }
    output1->SetDimNum(input_shape1->GetDimNum());
    return GRAPH_SUCCESS;
  };
  IMPL_OP(Type2_3Input_2Output).InferShape(infer_shape_func).InputsDataDependency({2})
    .InferDataType(nullptr)
    .InferShapeRange(nullptr);

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistry>();
  auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
  gert::OpImplKernelRegistry::OpImplFunctions op_impl_func;
  op_impl_func.infer_shape = infer_shape_func;
  op_impl_func.SetInputDataDependency(2);
  op_impl_func.infer_shape_range = nullptr;
  registry_holder->AddTypesToImpl("Type2_3Input_2Output", op_impl_func);
  space_registry->AddRegistry(registry_holder);
  DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(space_registry);

  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  ASSERT_NE(call_infer_shape_v2, nullptr);
  auto status = call_infer_shape_v2(op, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  const auto &shape = op_desc->GetOutputDesc(0U).GetShape();
  ASSERT_EQ(shape.GetDimNum(), 4);
  ASSERT_EQ(shape.GetDim(0U), 85);
  ASSERT_EQ(shape.GetDim(1U), 86);
  ASSERT_EQ(shape.GetDim(2U), 87);
  ASSERT_EQ(shape.GetDim(3U), 88);
  const auto call_infer_shape_range = OperatorFactoryImpl::GetInferShapeRangeFunc();
  status = call_infer_shape_range(op, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  (void)op_desc->GetOutputDesc(1U).GetShapeRange(shape_range);
  ASSERT_EQ(shape_range.size(), 4U);
  for (size_t i = 0UL; i < shape_range.size(); ++i) {
    ASSERT_EQ(shape_range[i].first, range[i].first);
    ASSERT_EQ(shape_range[i].second, range[i].second);
  }
}

// 资源类算子测试
REG_OP(RegisterAndGetReiledOnResource)
    .INPUT(input1, "T")
        .OUTPUT(output1, "T2")
        .DATATYPE(T2, TensorType({DT_BOOL}))
        .OP_END_FACTORY_REG(RegisterAndGetReiledOnResource);
TEST_F(ShapeInferenceUT, CallInferV2Func_RegisterAndGetReiledOnResource) {
  auto op = OperatorFactory::CreateOperator("test6", "RegisterAndGetReiledOnResource");
  const char_t *resource_key = "224";
  auto read_inference_context = std::shared_ptr<InferenceContext>(InferenceContext::Create());
  read_inference_context->RegisterReliedOnResourceKey(AscendString(resource_key));
  op.SetInferenceContext(read_inference_context);

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);    // simulate read_op register relied resource
  // input1
  ge::GeShape shape1({1, 2, 3, 5});
  ge::GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_FLOAT16);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    const auto &read_inference_context = context->GetInferenceContextPtr();
    const auto &reiled_keys = read_inference_context->GetReliedOnResourceKeys();
    const char_t *resource_key_ = "224";
    // check result
    EXPECT_EQ(reiled_keys.empty(), false);
    EXPECT_EQ(*reiled_keys.begin(), resource_key_);
    if (reiled_keys.empty() ||
        (*reiled_keys.begin() != resource_key_)) {
      return GRAPH_FAILED;
    }
    auto out_shape = context->GetOutputShape(0UL);
    out_shape->SetDimNum(1UL);
    out_shape->SetDim(0UL, std::strtol(resource_key_, nullptr, 10));
    return GRAPH_SUCCESS;
  };
  IMPL_OP(RegisterAndGetReiledOnResource)
      .InferShape(infer_shape_func)
      .InferDataType(nullptr)
      .InferShapeRange(nullptr);

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistry>();
  auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
  gert::OpImplKernelRegistry::OpImplFunctions op_impl_func;
  op_impl_func.infer_shape = infer_shape_func;
  registry_holder->AddTypesToImpl("RegisterAndGetReiledOnResource", op_impl_func);
  space_registry->AddRegistry(registry_holder);
  DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(space_registry);

  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  ASSERT_NE(call_infer_shape_v2, nullptr);
  const auto status = call_infer_shape_v2(op, op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  const auto &shape = op_desc->GetOutputDesc(0U).GetShape();
  ASSERT_EQ(shape.GetDim(0), std::strtol(resource_key, nullptr, 10));
}

// 默认infer datatype测试
REG_OP(TestDefaultInferDataType)
    .INPUT(input1, "T")
        .OUTPUT(output1, "T")
        .DATATYPE(T, TensorType({DT_BOOL}))
        .OP_END_FACTORY_REG(TestDefaultInferDataType);
TEST_F(ShapeInferenceUT, CallInferV2Func_TestDefaultInferShape) {
  auto op = OperatorFactory::CreateOperator("test7", "TestDefaultInferDataType");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);    // simulate read_op register relied resource
  // input1
  ge::GeShape shape1({1, 2, 3, 5});
  ge::GeTensorDesc tensor_desc1(shape1, Format::FORMAT_NCHW, DT_BOOL);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginDataType(DT_BOOL);
  op_desc->UpdateInputDesc(0, tensor_desc1);
  const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
    return GRAPH_SUCCESS;
  };
  IMPL_OP(TestDefaultInferDataType)
      .InferShape(infer_shape_func)
      .InferDataType(nullptr)
      .InferShapeRange(nullptr);
  const auto call_infer_data_type = OperatorFactoryImpl::GetInferDataTypeFunc();
  const auto status = call_infer_data_type(op_desc);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  const auto &data_type = op_desc->GetOutputDesc(0U).GetDataType();
  ASSERT_EQ(data_type, DT_BOOL);
}
TEST_F(ShapeInferenceUT, AdaptFuncRegisterOk) {
  ASSERT_NE(OperatorFactoryImpl::GetInferShapeV2Func(), nullptr);
  ASSERT_NE(OperatorFactoryImpl::GetInferShapeRangeFunc(), nullptr);
  ASSERT_NE(OperatorFactoryImpl::GetInferDataTypeFunc(), nullptr);
}
}  // namespace gert
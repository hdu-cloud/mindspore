/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#define protected public
#define private public
#include "graph/op_desc.h"
#include "graph/op_desc_impl.h"
#include "graph/ge_tensor.h"
#include "graph/utils/ge_ir_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#undef private
#undef protected
#include "graph/utils/transformer_utils.h"
#include "graph/common_error_codes.h"
#include "graph/operator_factory_impl.h"
#include "register/op_tiling_registry.h"
#include "external/graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"
#include "external/graph/operator_reg.h"
#include "external/register/op_impl_registry.h"

namespace ge {
class UtestOpDescUtilsEx : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestOpDescUtilsEx, OpVerify_success) {
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(OpDescUtilsEx::OpVerify(op_desc), GRAPH_SUCCESS);
  const auto verify_func = [](Operator &op) {
    return GRAPH_SUCCESS;
  };
  op_desc->AddVerifierFunc(verify_func);
  EXPECT_EQ(OpDescUtilsEx::OpVerify(op_desc), GRAPH_SUCCESS);
}

TEST_F(UtestOpDescUtilsEx, InferShapeAndType_success) {
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(OpDescUtilsEx::InferShapeAndType(op_desc), GRAPH_SUCCESS);
  const auto add_func = [](Operator &op) {
    return GRAPH_SUCCESS;
  };
  op_desc->AddInferFunc(add_func);
  EXPECT_EQ(OpDescUtilsEx::InferShapeAndType(op_desc), GRAPH_SUCCESS);
}

TEST_F(UtestOpDescUtilsEx, InferDataSlice_success) {
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(OpDescUtilsEx::InferDataSlice(op_desc), NO_DEPENDENCE_FUNC);
  const auto infer_data_slice_func = [](Operator &op) {
    return GRAPH_SUCCESS;
  };
  auto op = std::make_shared<Operator>();
  op_desc->SetType("test");
  OperatorFactoryImpl::RegisterInferDataSliceFunc("test",infer_data_slice_func);
  EXPECT_EQ(OpDescUtilsEx::InferDataSlice(op_desc), GRAPH_SUCCESS);
}

REG_OP(FixInfer_OutputIsFix)
  .INPUT(fix_input1, "T")
  .INPUT(fix_input2, "T")
  .OUTPUT(fix_output, "T2")
  .DATATYPE(T2, TensorType({DT_BOOL}))
  .OP_END_FACTORY_REG(FixInfer_OutputIsFix);
TEST_F(UtestOpDescUtilsEx, CallInferFormatFunc_success) {
  auto op = OperatorFactory::CreateOperator("test", "FixInfer_OutputIsFix");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetType("test");
  const auto infer_format_func = [](Operator &op) {
    return GRAPH_SUCCESS;
  };
  OperatorFactoryImpl::RegisterInferFormatFunc("test", infer_format_func);
  EXPECT_EQ(OpDescUtilsEx::CallInferFormatFunc(op_desc, op), GRAPH_SUCCESS);
}

TEST_F(UtestOpDescUtilsEx, SetType_success) {
  auto op_desc = std::make_shared<OpDesc>();
  string type = "tmp";
  OpDescUtilsEx::SetType(op_desc, type);
  EXPECT_EQ(op_desc->GetType(), type);
}
}

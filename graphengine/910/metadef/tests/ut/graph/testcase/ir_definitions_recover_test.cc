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
#include <gtest/gtest.h>

#define private public
#define protected public
#include "graph/op_desc.h"
#include "graph/op_desc_impl.h"
#undef private
#undef protected

#include "graph/ir_definitions_recover.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "external/graph/operator_reg.h"

using namespace ge;

namespace gert {
class IrDefinitionsRecoverUT : public testing::Test {};

REG_OP(MatMulUt)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .REQUIRED_ATTR(loss_attr, Bool)
    .OP_END_FACTORY_REG(MatMulUt)

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_ir_inputs_not_match_failed) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);

  op_desc->impl_->meta_data_.ir_meta_.ir_attr_names_ = op_desc_origin->GetIrAttrNames();
  op_desc->impl_->meta_data_.ir_meta_.ir_inputs_.ir_inputs = op_desc_origin->GetIrInputs();
  ASSERT_FALSE(op_desc->impl_->meta_data_.ir_meta_.ir_inputs_.ir_inputs.empty());
  ASSERT_FALSE(op_desc->impl_->meta_data_.ir_meta_.ir_attr_names_.empty());
  op_desc->impl_->meta_data_.ir_meta_.ir_inputs_.ir_inputs[0].first = "fake";
  auto ret = RecoverIrDefinitions(computeGraph);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->impl_->meta_data_.ir_meta_.ir_inputs_.ir_inputs[0].first,  "fake");
}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_ir_inputs_num_check_failed) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);
  op_desc->impl_->meta_data_.ir_meta_.ir_inputs_.ir_inputs.emplace_back(std::pair<std::string, IrInputType>("fake", kIrInputRequired));
  auto ret = RecoverIrDefinitions(computeGraph);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->impl_->meta_data_.ir_meta_.ir_inputs_.ir_inputs[0].first,  "fake");
}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_ir_attr_name_not_match_failed) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);

  op_desc->impl_->meta_data_.ir_meta_.ir_attr_names_.emplace_back("fake");
  auto ret = RecoverIrDefinitions(computeGraph);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->impl_->meta_data_.ir_meta_.ir_attr_names_[0],  "fake");
}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_ir_attr_name_num_check_failed) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);

  op_desc->impl_->meta_data_.ir_meta_.ir_attr_names_.emplace_back("fake");
  auto ret = RecoverIrDefinitions(computeGraph);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->impl_->meta_data_.ir_meta_.ir_attr_names_.back(),  "fake");
}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_empty_success) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);

  // recover success
  auto ret = RecoverIrDefinitions(computeGraph);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->GetIrAttrNames().size(), op_desc_origin->GetIrAttrNames().size());
  EXPECT_EQ(op_desc->GetIrInputs().size(), op_desc_origin->GetIrInputs().size());
  EXPECT_EQ(op_desc->GetIrOutputs().size(), op_desc_origin->GetIrOutputs().size());
}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_partial_success) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);

  op_desc->AppendIrAttrName(op_desc_origin->GetIrAttrNames().at(0));
  auto &pair = op_desc_origin->GetIrInputs().at(0);
  op_desc->AppendIrInput(pair.first, pair.second);

  // recover success
  auto ret = RecoverIrDefinitions(computeGraph);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->GetIrAttrNames().size(), op_desc_origin->GetIrAttrNames().size());
  EXPECT_EQ(op_desc->GetIrInputs().size(), op_desc_origin->GetIrInputs().size());
  EXPECT_EQ(op_desc->GetIrOutputs().size(), op_desc_origin->GetIrOutputs().size());
}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_same_success) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);

  for (const auto &attr : op_desc_origin->GetIrAttrNames()) {
    op_desc->AppendIrAttrName(attr);
  }
  for (const auto &pair : op_desc_origin->GetIrInputs()) {
    op_desc->AppendIrInput(pair.first, pair.second);
  }
  // recover success
  auto ret = RecoverIrDefinitions(computeGraph);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->GetIrAttrNames().size(), op_desc_origin->GetIrAttrNames().size());
  EXPECT_EQ(op_desc->GetIrInputs().size(), op_desc_origin->GetIrInputs().size());
  EXPECT_EQ(op_desc->GetIrOutputs().size(), op_desc_origin->GetIrOutputs().size());
}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_frameworkop_success) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "FrameworkOp");
  AttrUtils::SetStr(op_desc, "original_type", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMul", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);

  // recover success
  auto ret = RecoverIrDefinitions(computeGraph);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->GetIrAttrNames().size(), op_desc_origin->GetIrAttrNames().size());
  EXPECT_EQ(op_desc->GetIrInputs().size(), op_desc_origin->GetIrInputs().size());
  EXPECT_EQ(op_desc->GetIrOutputs().size(), op_desc_origin->GetIrOutputs().size());

}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_op_loss_not_has_default_value) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);

  // recover success
  auto ret = RecoverIrDefinitions(computeGraph);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_FALSE(ge::AttrUtils::HasAttr(op_desc, "loss_attr"));
  EXPECT_TRUE(ge::AttrUtils::HasAttr(op_desc, "transpose_x1"));
}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_ir_outputs_not_match_failed) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);

  op_desc->impl_->meta_data_.ir_meta_.ir_attr_names_ = op_desc_origin->GetIrAttrNames();
  op_desc->impl_->meta_data_.ir_meta_.ir_outputs_.ir_outputs = op_desc_origin->GetIrOutputs();
  ASSERT_FALSE(op_desc->impl_->meta_data_.ir_meta_.ir_outputs_.ir_outputs.empty());
  ASSERT_FALSE(op_desc->impl_->meta_data_.ir_meta_.ir_attr_names_.empty());
  op_desc->impl_->meta_data_.ir_meta_.ir_outputs_.ir_outputs[0].first = "fake";
  auto ret = RecoverIrDefinitions(computeGraph);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->impl_->meta_data_.ir_meta_.ir_outputs_.ir_outputs[0].first,  "fake");
}

TEST_F(IrDefinitionsRecoverUT, RecoverIrDefinitions_ir_outputs_num_check_failed) {
  auto op_desc = std::make_shared<ge::OpDesc>("matmul", "MatMulUt");
  ASSERT_NE(op_desc, nullptr);
  auto computeGraph = std::make_shared<ge::ComputeGraph>("graph_name");
  ASSERT_NE(computeGraph, nullptr);
  ASSERT_NE(computeGraph->AddNode(op_desc), nullptr);

  auto op = ge::OperatorFactory::CreateOperator("MatMulUt", "MatMulUt");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);
  op_desc->impl_->meta_data_.ir_meta_.ir_outputs_.ir_outputs.emplace_back(std::pair<std::string, IrOutputType>("fake", kIrOutputRequired));
  auto ret = RecoverIrDefinitions(computeGraph);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->impl_->meta_data_.ir_meta_.ir_outputs_.ir_outputs[0].first,  "fake");
}

} // namespace gert

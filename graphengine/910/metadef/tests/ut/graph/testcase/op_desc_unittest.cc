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
#undef private
#undef protected
#include "graph/utils/transformer_utils.h"
#include "graph/common_error_codes.h"
#include "graph/operator_factory_impl.h"
#include "register/op_tiling_registry.h"
#include "external/graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "external/graph/operator_reg.h"
#include "external/register/op_impl_registry.h"

namespace ge {
class UtestOpDesc : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestOpDesc, TestCommonVerifyOnDummyShape) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({-3}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>("test", "Identity");
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddOutputDesc(tensor_desc->Clone());

  EXPECT_EQ(GRAPH_SUCCESS, op_desc->CommonVerify());
}

TEST_F(UtestOpDesc, TestOpDescGetSetTensorDesc) {
  GeTensorDesc desc(GeShape(), FORMAT_NCHW, DT_INT32);
  OpDesc op_desc("foo", "Foo");
  EXPECT_EQ(GRAPH_SUCCESS, op_desc.AddInputDesc("x", desc));
  EXPECT_EQ(GRAPH_SUCCESS, op_desc.AddOutputDesc("y", desc));

  EXPECT_EQ(op_desc.GetInputDesc("x"), desc);
  EXPECT_EQ(op_desc.GetOutputDesc("y"), desc);
}

TEST_F(UtestOpDesc, TestNodeShapeTransUtils) {

  NodeShapeTransUtils transformer1(nullptr);
  EXPECT_NE(transformer1.Init(), true);

  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1, 1, 16, 16}));
  tensor_desc->SetFormat(FORMAT_FRACTAL_NZ);
  tensor_desc->SetDataType(DT_FLOAT);
  tensor_desc->SetOriginFormat(FORMAT_ND);

  auto op_desc = std::make_shared<OpDesc>("test", "Identity");
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddOutputDesc(tensor_desc->Clone());
  NodeShapeTransUtils transformer2(op_desc);
  EXPECT_EQ(transformer2.Init(), true);
  EXPECT_EQ(transformer2.CatchFormatAndShape(), true);
  EXPECT_EQ(transformer2.UpdateFormatAndShape(), true);


  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddOutputDesc(tensor_desc->Clone());

  NodeShapeTransUtils transformer3(op_desc);
  EXPECT_EQ(transformer3.Init(), true);
  EXPECT_EQ(transformer3.CatchFormatAndShape(), true);
  EXPECT_EQ(transformer3.UpdateFormatAndShape(), true);


  EXPECT_EQ(GRAPH_SUCCESS, op_desc->CommonVerify());
}

TEST_F(UtestOpDesc, IndexOutOfRange) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>("test", "Identity");
  op_desc->AddInputDesc(tensor_desc->Clone());

  EXPECT_NE(nullptr, op_desc->MutableInputDesc(0));
  EXPECT_EQ(nullptr, op_desc->MutableInputDesc(1));
  EXPECT_EQ(nullptr, op_desc->MutableInputDesc(999));
}

TEST_F(UtestOpDesc, SerializeMetadata) {
  OpDescImpl impl;
  impl.meta_data_.inputs_.emplace_back("input");
  impl.meta_data_.input_names_.emplace_back("names");
  impl.meta_data_.src_names_.push_back("src");
  impl.meta_data_.dst_names_.push_back("dst");
  impl.meta_data_.dst_indexes_.push_back(2);
  impl.meta_data_.src_indexes_.push_back(2);
  impl.meta_data_.input_offsets_.push_back(987654321);
  impl.meta_data_.output_offsets_.push_back(987654321);
  impl.meta_data_.workspaces.push_back(222);
  impl.meta_data_.workspace_bytes_list_.push_back(111);
  impl.meta_data_.is_input_consts_.push_back(false);

  proto::OpDef def;
  impl.SerializeMetaDataToOpDef(&def);
  EXPECT_EQ(def.input(0), "input");
  EXPECT_EQ(def.input_name(0), "names");
  EXPECT_EQ(def.src_name(0), "src");
  EXPECT_EQ(def.dst_name(0), "dst");
  EXPECT_EQ(def.dst_index(0), 2);
  EXPECT_EQ(def.src_index(0), 2);
  EXPECT_EQ(def.input_i(0), 987654321);
  EXPECT_EQ(def.output_i(0), 987654321);
  EXPECT_EQ(def.workspace(0), 222);
  EXPECT_EQ(def.workspace_bytes(0), 111);
  EXPECT_EQ(def.is_input_const(0), false);
}

TEST_F(UtestOpDesc, DeSerializeMetadata) {
  proto::OpDef def;
  def.add_input("input");
  def.add_input_name("names");
  def.add_src_name("src");
  def.add_dst_name("dst");
  def.add_dst_index(2);
  def.add_src_index(2);
  def.add_input_i(987654321);
  def.add_output_i(987654321);
  def.add_workspace(222);
  def.add_workspace_bytes(222);
  def.add_is_input_const(false);
  OpDescImpl impl;
  impl.DeSerializeOpDefToMetaData(def);
  EXPECT_EQ(impl.meta_data_.inputs_.size(), 1);
  EXPECT_EQ(impl.meta_data_.inputs_[0], "input");
  EXPECT_EQ(impl.meta_data_.input_names_.size(), 1);
  EXPECT_EQ(impl.meta_data_.input_names_[0], "names");
  EXPECT_EQ(impl.meta_data_.src_names_.size(), 1);
  EXPECT_EQ(impl.meta_data_.src_names_[0], "src");
  EXPECT_EQ(impl.meta_data_.dst_names_.size(), 1);
  EXPECT_EQ(impl.meta_data_.dst_names_[0], "dst");
  EXPECT_EQ(impl.meta_data_.dst_indexes_.size(), 1);
  EXPECT_EQ(impl.meta_data_.dst_indexes_[0], 2);
  EXPECT_EQ(impl.meta_data_.src_indexes_.size(), 1);
  EXPECT_EQ(impl.meta_data_.src_indexes_[0], 2);
  EXPECT_EQ(impl.meta_data_.input_offsets_.size(), 1);
  EXPECT_EQ(impl.meta_data_.input_offsets_[0], 987654321);
  EXPECT_EQ(impl.meta_data_.output_offsets_.size(), 1);
  EXPECT_EQ(impl.meta_data_.output_offsets_[0], 987654321);
  EXPECT_EQ(impl.meta_data_.workspaces.size(), 1);
  EXPECT_EQ(impl.meta_data_.workspaces[0], 222);
  EXPECT_EQ(impl.meta_data_.workspace_bytes_list_.size(), 1);
  EXPECT_EQ(impl.meta_data_.workspace_bytes_list_[0], 222);
  EXPECT_EQ(impl.meta_data_.is_input_consts_.size(), 1);
  EXPECT_EQ(impl.meta_data_.is_input_consts_[0], false);

  OpDescImpl impl1;
  impl1.DeSerializeOpDefToMetaData(def);
  EXPECT_TRUE(impl1.OpDescAttrsAreEqual(impl));
}

TEST_F(UtestOpDesc, AddDescForward) {
  GeTensorDesc desc(GeShape(), FORMAT_NCHW, DT_INT32);
  OpDesc op_desc("foo", "Foo");
  EXPECT_EQ(GRAPH_SUCCESS, op_desc.AddOutputDesc("x", desc));
  EXPECT_EQ(GRAPH_SUCCESS, op_desc.AddOutputDesc("y", desc));
  EXPECT_EQ(GRAPH_SUCCESS, op_desc.AddOutputDesc("z", desc));
  EXPECT_EQ(GRAPH_SUCCESS, op_desc.AddOutputDescForward("t", 2));

  EXPECT_EQ(5, op_desc.GetOutputsSize());
}

TEST_F(UtestOpDesc, AddInputDesc1_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  EXPECT_EQ(op_desc->AddInputDesc(0, tensor_desc->Clone()), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->AddInputDesc(0, tensor_desc->Clone()), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, AddInputDesc2_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  EXPECT_EQ(op_desc->AddInputDesc("input_desc1", tensor_desc->Clone()), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->AddInputDesc("input_desc1", tensor_desc->Clone()), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, AddInputDescMiddle_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  op_desc->AddInputDesc("input_desc1", tensor_desc->Clone());
  op_desc->AddInputDesc("input_desc2", tensor_desc->Clone());

  EXPECT_EQ(op_desc->AddInputDescMiddle("x", 2, 1), GRAPH_SUCCESS);
  auto name_idx = op_desc->GetAllInputName();
  ASSERT_EQ(name_idx.size(), 4U);
  EXPECT_EQ(name_idx["x0"], 1);
  EXPECT_EQ(name_idx["x1"], 2);
  EXPECT_EQ(name_idx["input_desc2"], 3);
}

TEST_F(UtestOpDesc, AddOutputDescMiddle_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  op_desc->AddOutputDesc("output_desc1", tensor_desc->Clone());
  op_desc->AddOutputDesc("output_desc2", tensor_desc->Clone());

  EXPECT_EQ(op_desc->AddOutputDescMiddle("y", 2, 1), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->AddOutputDescMiddle("output_desc4", 1, 5), GRAPH_FAILED);
  auto name_idx = op_desc->GetAllOutputName();
  ASSERT_EQ(name_idx.size(), 4U);
  EXPECT_EQ(name_idx["y0"], 1);
  EXPECT_EQ(name_idx["y1"], 2);
  EXPECT_EQ(name_idx["output_desc2"], 3);
}

TEST_F(UtestOpDesc, UpdateInputDesc_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  op_desc->AddInputDesc("input_desc1", tensor_desc->Clone());
  op_desc->AddInputDesc("input_desc2", tensor_desc->Clone());

  EXPECT_EQ(op_desc->UpdateInputDesc(1, tensor_desc->Clone()), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->UpdateInputDesc(4, tensor_desc->Clone()), GRAPH_FAILED);
}

TEST_F(UtestOpDesc, UpdateInputDescForward_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  op_desc->AddInputDesc("input1", tensor_desc->Clone());
  EXPECT_EQ(op_desc->AddDynamicInputDesc("x", 2, false), GRAPH_SUCCESS);
  auto input_name_idx = op_desc->GetAllInputName();
  ASSERT_EQ(input_name_idx.size(), 3U);
  EXPECT_EQ(input_name_idx["x0"], 0);
  EXPECT_EQ(input_name_idx["x1"], 1);
  EXPECT_EQ(input_name_idx["input1"], 2);
}

TEST_F(UtestOpDesc, AddOutputDescForward_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddOutputDesc(tensor_desc->Clone());
  EXPECT_EQ(op_desc->AddOutputDescForward("y", 2), GRAPH_SUCCESS);

  auto output_name_idx = op_desc->GetAllOutputName();
  ASSERT_EQ(output_name_idx.size(), 3U);
  EXPECT_EQ(output_name_idx["y0"], 0);
  EXPECT_EQ(output_name_idx["y1"], 1);
  EXPECT_EQ(output_name_idx["__output0"], 2);
}

TEST_F(UtestOpDesc, AddOptionalInputDesc_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(op_desc->AddOptionalInputDesc("test", tensor_desc->Clone()), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, OpDescMembersAreEqual_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  auto op_desc1 = std::make_shared<OpDesc>();
  op_desc1->AddInputDesc("input_desc", tensor_desc->Clone());
  op_desc1->AddOutputDesc("output_desc", tensor_desc->Clone());
  op_desc1->AddOptionalInputDesc("optional_input", tensor_desc->Clone());
  op_desc1->SetOpEngineName("DNN_VM_HOST_CPU");
  op_desc1->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");

  auto op_desc2 = std::make_shared<OpDesc>();
  op_desc1->AddInputDesc("input_desc_diff", tensor_desc->Clone());
  op_desc1->AddOutputDesc("output_desc", tensor_desc->Clone());
  op_desc1->AddOptionalInputDesc("optional_input", tensor_desc->Clone());
  op_desc1->SetOpEngineName("DNN_VM_HOST_CPU");
  op_desc1->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");

  auto op_desc3 = op_desc1;

  EXPECT_EQ(op_desc1->OpDescMembersAreEqual(*(op_desc3)), true);
  EXPECT_EQ(op_desc1->OpDescMembersAreEqual(*(op_desc2)), false);
}

TEST_F(UtestOpDesc, OpDescGenTensorDescsAreEqual_success) {
  auto tensor_desc1 = std::make_shared<GeTensorDesc>();
  tensor_desc1->SetShape(GeShape({1}));
  tensor_desc1->SetFormat(FORMAT_NCHW);
  tensor_desc1->SetDataType(DT_FLOAT);

  auto tensor_desc2 = std::make_shared<GeTensorDesc>();
  tensor_desc2->SetShape(GeShape({-1}));
  tensor_desc2->SetFormat(FORMAT_NHWC);
  tensor_desc2->SetDataType(DT_INT32);

  auto op_desc1 = std::make_shared<OpDesc>();
  op_desc1->AddInputDesc(tensor_desc1->Clone());
  auto op_desc2 = std::make_shared<OpDesc>();
  EXPECT_EQ(op_desc1->OpDescGenTensorDescsAreEqual(*(op_desc2)), false);
  op_desc2->AddInputDesc(tensor_desc2->Clone());
  op_desc1->AddOutputDesc(tensor_desc1->Clone());
  EXPECT_EQ(op_desc1->OpDescGenTensorDescsAreEqual(*(op_desc2)), false);
  op_desc2->AddOutputDesc(tensor_desc2->Clone());
  auto op_desc3 = std::make_shared<OpDesc>();
  EXPECT_EQ(op_desc1->OpDescGenTensorDescsAreEqual(*(op_desc2)), false);
  op_desc3->AddInputDesc(tensor_desc1->Clone());
  op_desc3->AddOutputDesc(tensor_desc2->Clone());
  EXPECT_EQ(op_desc1->OpDescGenTensorDescsAreEqual(*(op_desc3)), false);
  EXPECT_EQ(op_desc1->OpDescGenTensorDescsAreEqual(*(op_desc1)), true);
}

TEST_F(UtestOpDesc, InputIsSet_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(op_desc->InputIsSet("input_test"), false);
  op_desc->AddInputDesc("input_test",tensor_desc->Clone());
  EXPECT_EQ(op_desc->InputIsSet("input_test"), true);
}

TEST_F(UtestOpDesc, MutableInputDesc_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddInputDesc("input_test1",tensor_desc->Clone());
  EXPECT_EQ(op_desc->MutableInputDesc("input_test"), nullptr);
  EXPECT_NE(op_desc->MutableInputDesc("input_test1"), nullptr);
}

TEST_F(UtestOpDesc, Get_SetOpKernelLibName_success) {
  auto op_desc = std::make_shared<OpDesc>();
  op_desc->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");
  EXPECT_EQ(op_desc->GetOpKernelLibName(), "DNN_VM_RTS_OP_STORE");
}

TEST_F(UtestOpDesc, Get_SetOpEngineName_success) {
  auto op_desc = std::make_shared<OpDesc>();
  op_desc->SetOpEngineName("DNN_VM_HOST_CPU");
  EXPECT_EQ(op_desc->GetOpEngineName(), "DNN_VM_HOST_CPU");
}

TEST_F(UtestOpDesc, GetAllOutputsDescSize_sucess) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddOutputDesc(tensor_desc->Clone());
  op_desc->AddOutputDesc(tensor_desc->Clone());
  EXPECT_EQ(op_desc->GetAllOutputsDescSize(), 2);
}

TEST_F(UtestOpDesc, AddDynamicInputDescByIndex_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddInputDesc("input_test1",tensor_desc->Clone());
  op_desc->AddInputDesc("input_test2",tensor_desc->Clone());
  EXPECT_EQ(op_desc->AddDynamicInputDescByIndex("input_test2", 1, 1), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, IsOptionalInput_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddOptionalInputDesc("optional_test", tensor_desc->Clone());
  op_desc->AddInputDesc("input_test", tensor_desc->Clone());
  EXPECT_EQ(op_desc->IsOptionalInput("input_test"), false);
  EXPECT_EQ(op_desc->IsOptionalInput("optional_test"), true);
}

TEST_F(UtestOpDesc, GetAllOutputName_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  op_desc->AddOutputDesc("output1", tensor_desc->Clone());
  op_desc->AddOutputDesc("output2", tensor_desc->Clone());
  std::map<std::string, uint32_t> all_output;
  all_output = op_desc->GetAllOutputName();
  EXPECT_EQ(all_output.size(), 2);
  EXPECT_EQ(all_output["output1"], 0);
  EXPECT_EQ(all_output["output2"], 1);
}

TEST_F(UtestOpDesc, UpdateInputName_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>();

  op_desc->AddInputDesc("name1", tensor_desc->Clone());
  op_desc->AddInputDesc("name2", tensor_desc->Clone());

  std::map<std::string, uint32_t> input_name_idx;
  input_name_idx.insert(pair<std::string, uint32_t>("update_name1", 0));
  EXPECT_EQ(op_desc->UpdateInputName(input_name_idx), false);
  input_name_idx.insert(pair<std::string, uint32_t>("update_name2", 1));
  EXPECT_EQ(op_desc->UpdateInputName(input_name_idx), true);
  auto all_input_name = op_desc->GetAllInputName();
  EXPECT_EQ(input_name_idx, all_input_name);
  input_name_idx.insert(pair<std::string, uint32_t>("update_name3", 2));
  EXPECT_EQ(op_desc->UpdateInputName(input_name_idx), true);
}

TEST_F(UtestOpDesc, UpdateOutputName_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>();

  op_desc->AddOutputDesc("name1", tensor_desc->Clone());
  op_desc->AddOutputDesc("name2", tensor_desc->Clone());

  std::map<std::string, uint32_t> output_name_idx;
  output_name_idx.insert(pair<std::string, uint32_t>("update_name1", 0));
  EXPECT_EQ(op_desc->UpdateOutputName(output_name_idx), false);
  output_name_idx.insert(pair<std::string, uint32_t>("update_name2", 1));
  EXPECT_EQ(op_desc->UpdateOutputName(output_name_idx), true);
  auto all_output_name = op_desc->GetAllOutputName();
  EXPECT_EQ(output_name_idx, all_output_name);
  output_name_idx.insert(pair<std::string, uint32_t>("update_name3", 2));
  EXPECT_EQ(op_desc->UpdateOutputName(output_name_idx), true);
}

TEST_F(UtestOpDesc, GetInferFunc_success) {
  auto op_desc = std::make_shared<OpDesc>();
  const auto add_func = [](Operator &op) {
    return GRAPH_SUCCESS;
  };
  op_desc->AddInferFunc(add_func);

  Operator op;
  auto func = op_desc->GetInferFunc();
  EXPECT_EQ(func == nullptr, false);
  EXPECT_EQ(func(op), GRAPH_SUCCESS);
}

// infer from output
REG_OP(FixIOOp_OutputIsFix)
    .INPUT(fix_input1, "T")
        .INPUT(fix_input2, "T")
        .OUTPUT(fix_output, "T2")
        .DATATYPE(T2, TensorType({DT_BOOL}))
        .OP_END_FACTORY_REG(FixIOOp_OutputIsFix);
TEST_F(UtestOpDesc, CallInferV2Func_success) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOp_OutputIsFix");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1,1,1,1});
  GeTensorDesc tensor_desc(shape, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{0, 10000}};
  tensor_desc.SetOriginShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc);
  op_desc->UpdateInputDesc(1, tensor_desc);
  op_desc->impl_->infer_func_ = nullptr;
  auto infer_shape_func = [](const ge::Operator &op, const OpDescPtr &op_desc) -> uint32_t {
    const ge::GeTensorDesc &input_desc = op_desc->GetInputDesc(0UL);
    return op_desc->UpdateOutputDesc(0UL, input_desc);
  };
  auto infer_shape_range_func = [](const ge::Operator &op, const OpDescPtr &op_desc) -> uint32_t {
    return GRAPH_SUCCESS;
  };
  auto infer_data_type_func = [](const OpDescPtr &op) -> uint32_t {
    return GRAPH_SUCCESS;
  };
  (void) ge::OperatorFactoryImpl::RegisterInferShapeV2Func(infer_shape_func);
  (void) ge::OperatorFactoryImpl::RegisterInferShapeRangeFunc(infer_shape_range_func);
  (void) ge::OperatorFactoryImpl::RegisterInferDataTypeFunc(infer_data_type_func);
  auto status = OpDescUtilsEx::CallInferFunc(op_desc, op);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetDataType(), DT_FLOAT16);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 4);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDim(0), 1);
  ge::OperatorFactoryImpl::operator_infer_shape_v2_func_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_datatype_func_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_shape_range_func_ = nullptr;
}

// 测试输入format和原始format不一致的情况下，infershape结果是否正确
TEST_F(UtestOpDesc, CallInferV2Func_UpdateShape_success) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOp_OutputIsFix");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1,64,1,1});
  GeShape origin_shape({1,4,1,1,16});
  GeTensorDesc tensor_desc(shape, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginShape(origin_shape);
  tensor_desc.SetOriginFormat(Format::FORMAT_NC1HWC0);
  tensor_desc.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{0, 10000}};
  tensor_desc.SetOriginShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc);
  op_desc->UpdateInputDesc(1, tensor_desc);
  op_desc->impl_->infer_func_ = nullptr;
  auto infer_shape_func = [](const ge::Operator &op, const OpDescPtr &op_desc) -> uint32_t {
    const ge::GeTensorDesc &input_desc = op_desc->GetInputDesc(0UL);
    return op_desc->UpdateOutputDesc(0UL, input_desc);
  };
  auto infer_shape_range_func = [](const ge::Operator &op, const OpDescPtr &op_desc) -> uint32_t {
    return GRAPH_SUCCESS;
  };
  auto infer_data_type_func = [](const OpDescPtr &op) -> uint32_t {
    return GRAPH_SUCCESS;
  };
  (void) ge::OperatorFactoryImpl::RegisterInferShapeV2Func(infer_shape_func);
  (void) ge::OperatorFactoryImpl::RegisterInferShapeRangeFunc(infer_shape_range_func);
  (void) ge::OperatorFactoryImpl::RegisterInferDataTypeFunc(infer_data_type_func);
  auto status = OpDescUtilsEx::CallInferFunc(op_desc, op);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetDataType(), DT_FLOAT16);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 5);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDim(0), 1);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDim(1), 4);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDim(2), 1);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDim(3), 1);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDim(4), 16);
  ge::OperatorFactoryImpl::operator_infer_shape_v2_func_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_datatype_func_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_shape_range_func_ = nullptr;
}

TEST_F(UtestOpDesc, CallInferV2Func_failed) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOp_OutputIsFix");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1,1,1,1});
  GeTensorDesc tensor_desc(shape, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{0, 10000}};
  tensor_desc.SetOriginShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc);
  op_desc->UpdateInputDesc(1, tensor_desc);
  op_desc->impl_->infer_func_ = nullptr;
  auto status = OpDescUtilsEx::CallInferFunc(op_desc, op);
  ASSERT_EQ(status, GRAPH_PARAM_INVALID);
}

TEST_F(UtestOpDesc, CallInferFunc_success) {
  OpDescImpl op_desc_impl;
  Operator op;
  OpDescPtr op_desc;
  auto status = OpDescUtilsEx::CallInferFunc(op_desc, op);
  const auto func = [](Operator &op) { return GRAPH_SUCCESS; };
  op_desc_impl.infer_func_ = func;
  status = OpDescUtilsEx::CallInferFunc(op_desc, op);
  const auto infer_data_slice_func = [](Operator &op) {
    return GRAPH_SUCCESS;
  };

  OpDescPtr odp = std::make_shared<OpDesc>("name", "type");
  op_desc_impl.infer_func_ = infer_data_slice_func;
  status = OpDescUtilsEx::CallInferFunc(odp, op);
}

TEST_F(UtestOpDesc, InferDataSlice_success) {
  auto op_desc = std::make_shared<OpDesc>();
  const auto func = [](Operator &op) { return GRAPH_SUCCESS; };
  EXPECT_EQ(OpDescUtilsEx::InferDataSlice(op_desc), NO_DEPENDENCE_FUNC);
  const auto infer_data_slice_func = [](Operator &op) {
    return GRAPH_SUCCESS;
  };
  auto op = std::make_shared<Operator>();
  op_desc->SetType("test");
  OperatorFactoryImpl::RegisterInferDataSliceFunc("test",infer_data_slice_func);
  EXPECT_EQ(OpDescUtilsEx::InferDataSlice(op_desc), GRAPH_SUCCESS);
}

REG_OP(MatMulUt)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .OP_END_FACTORY_REG(MatMulUt)

REG_OP(AddUt)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(AddUt)

TEST_F(UtestOpDesc, SetTypeModifyIrAttrName_type_change) {
  auto op = ge::OperatorFactory::CreateOperator("MatMul", "MatMulUt");
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  EXPECT_NE(op_desc, nullptr);
  EXPECT_FALSE(op_desc->GetIrAttrNames().empty());
  EXPECT_FALSE(op_desc->GetIrInputs().empty());
  op_desc->SetType("AddUt");

  auto add_op = ge::OperatorFactory::CreateOperator("add", "AddUt");
  auto add_op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  EXPECT_TRUE(op_desc->GetIrAttrNames() == add_op_desc->GetIrAttrNames());
  EXPECT_TRUE(op_desc->GetIrInputs() == add_op_desc->GetIrInputs());
}

TEST_F(UtestOpDesc, SetTypeModifyIrAttrName_type_not_exist_clear) {
  auto op = ge::OperatorFactory::CreateOperator("MatMul", "MatMul");
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  EXPECT_NE(op_desc, nullptr);
  EXPECT_FALSE(op_desc->GetIrAttrNames().empty());
  EXPECT_FALSE(op_desc->GetIrInputs().empty());

  OpDescUtilsEx::SetType(op_desc, "NotExist");
  EXPECT_TRUE(op_desc->GetIrAttrNames().empty());
  EXPECT_TRUE(op_desc->GetIrInputs().empty());
}

TEST_F(UtestOpDesc, SetTypeModifyIrAttrName_type_not_change) {
  auto op = ge::OperatorFactory::CreateOperator("MatMul", "MatMulUt");
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  EXPECT_NE(op_desc, nullptr);
  auto &check_ir_attr = op_desc->GetIrAttrNames();
  auto &check_ir_inputs = op_desc->GetIrInputs();
  EXPECT_FALSE(op_desc->GetIrAttrNames().empty());
  EXPECT_FALSE(op_desc->GetIrInputs().empty());

  op_desc->SetType("MatMulUt");
  EXPECT_TRUE(op_desc->GetIrAttrNames() == check_ir_attr);
  EXPECT_TRUE(op_desc->GetIrInputs() == check_ir_inputs);
}

TEST_F(UtestOpDesc, InferShapeAndType_success) {
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(OpDescUtilsEx::InferShapeAndType(op_desc), GRAPH_SUCCESS);
  const auto add_func = [](Operator &op) {
    return GRAPH_SUCCESS;
  };
  op_desc->AddInferFunc(add_func);
  EXPECT_EQ(OpDescUtilsEx::InferShapeAndType(op_desc), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, OpVerify_success) {
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(OpDescUtilsEx::OpVerify(op_desc), GRAPH_SUCCESS);
  const auto verify_func = [](Operator &op) {
    return GRAPH_SUCCESS;
  };
  op_desc->AddVerifierFunc(verify_func);
  EXPECT_EQ(OpDescUtilsEx::OpVerify(op_desc), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, GetValidInputNameByIndex_success) {
  auto op_desc = std::make_shared<OpDesc>("verify", "Rule");
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  op_desc->AddInputDesc("name1", tensor_desc->Clone());
  op_desc->AddInputDesc("name2", tensor_desc->Clone());

  EXPECT_EQ(op_desc->GetValidInputNameByIndex(0), "name1");
  EXPECT_EQ(op_desc->GetValidInputNameByIndex(1), "name2");
}

TEST_F(UtestOpDesc, GetStreamId_success) {
  auto op_desc = std::make_shared<OpDesc>();
  op_desc->SetStreamId(1);
  EXPECT_EQ(op_desc->GetStreamId(), 1);
}

TEST_F(UtestOpDesc, Set_GetInputName_success) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<std::string> input_name {"name1", "name2"};
  op_desc->SetInputName(input_name);
  auto get_input_name = op_desc->GetInputName();
  EXPECT_EQ(get_input_name.size(), 2);
  EXPECT_EQ(get_input_name[0], "name1");
  EXPECT_EQ(get_input_name[1], "name2");
}

TEST_F(UtestOpDesc, GetSrcName_success) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<std::string> src_name {"src"};
  op_desc->SetSrcName(src_name);
  auto get_src_name = op_desc->GetSrcName();
  EXPECT_EQ(get_src_name.size(), 1);
  EXPECT_EQ(get_src_name[0], "src");
}

TEST_F(UtestOpDesc, GetSrcIndex_success) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<int64_t> src_index{2};
  op_desc->SetSrcIndex(src_index);
  auto get_src_index = op_desc->GetSrcIndex();
  EXPECT_EQ(get_src_index.size(), 1);
  EXPECT_EQ(get_src_index[0], 2);
}

TEST_F(UtestOpDesc, GetInputOffset_success) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<int64_t> input_offset{987654321};
  op_desc->SetInputOffset(input_offset);
  auto get_input_offset = op_desc->GetInputOffset();
  EXPECT_EQ(get_input_offset.size(), 1);
  EXPECT_EQ(get_input_offset[0], 987654321);
}

TEST_F(UtestOpDesc, GetOutputOffset_success) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<int64_t> output_offset{987654321};
  op_desc->SetOutputOffset(output_offset);
  auto get_output_offset = op_desc->GetOutputOffset();
  EXPECT_EQ(get_output_offset.size(), 1);
  EXPECT_EQ(get_output_offset[0], 987654321);
}

TEST_F(UtestOpDesc, GetDstName_success) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<std::string> dst_name{"dst"};
  op_desc->SetDstName(dst_name);
  auto get_dst_name = op_desc->GetDstName();
  EXPECT_EQ(get_dst_name.size(), 1);
  EXPECT_EQ(get_dst_name[0], "dst");
}

TEST_F(UtestOpDesc, Set_GetOpInferDepends_success) {
  auto op_desc = std::make_shared<OpDesc>("verify", "Rule");
  std::vector<std::string> depend_names {"depend_name1", "depend_name2"};
  op_desc->SetOpInferDepends(depend_names);
  auto get_depend_names = op_desc->GetOpInferDepends();
  EXPECT_EQ(get_depend_names.size(), 2);
  EXPECT_EQ(get_depend_names[0], "depend_name1");
  EXPECT_EQ(get_depend_names[1], "depend_name2");
}

TEST_F(UtestOpDesc, GetWorkspace_success) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<int64_t> workspace{222};
  op_desc->SetWorkspace(workspace);
  auto get_workspace = op_desc->GetWorkspace();
  EXPECT_EQ(get_workspace.size(), 1);
  EXPECT_EQ(get_workspace[0], 222);
}

TEST_F(UtestOpDesc, GetSubgraphNameByInstanceName_success) {
  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddSubgraphName("subgraph");
  op_desc->SetSubgraphInstanceName(0, "subgraph");
  std::string subname("");
  EXPECT_EQ(op_desc->GetSubgraphNameByInstanceName("subgraph", subname), GRAPH_SUCCESS);
  EXPECT_EQ(subname, "subgraph");

  auto op_desc1 = std::make_shared<OpDesc>();
  op_desc1->AddSubgraphName("subgraph1");
  op_desc1->SetSubgraphInstanceName(0, "sub");
  EXPECT_EQ(op_desc1->GetSubgraphNameByInstanceName("sub", subname), GRAPH_SUCCESS);
  EXPECT_EQ(subname, "subgraph1");
}

TEST_F(UtestOpDesc, GetTilingInfo) {
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_NE(op_desc, nullptr);
  EXPECT_EQ(op_desc->GetTilingFuncInfo(), nullptr);
  EXPECT_EQ(op_desc->GetAtomicTilingFuncInfo(), nullptr);

  ::optiling::OpTilingFuncInfo tiling_info, atomic_tiling_info;
  op_desc->SetTilingFuncInfo(&tiling_info);
  op_desc->SetAtomicTilingFuncInfo(&atomic_tiling_info);
  EXPECT_EQ(op_desc->GetTilingFuncInfo(), &tiling_info);
  EXPECT_EQ(op_desc->GetAtomicTilingFuncInfo(), &atomic_tiling_info);
}

TEST_F(UtestOpDesc, CopyAssignTest) {
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_NE(op_desc, nullptr);
  op_desc->SetType("Test");
  OpDescImpl op_desc_impl;
  op_desc_impl = *(op_desc->impl_);
  EXPECT_EQ(op_desc_impl.GetType(), op_desc->GetType());
  // same object
  auto fake = &op_desc_impl;
  op_desc_impl = *fake;
  EXPECT_EQ(op_desc_impl.GetType(), op_desc->GetType());
}
}

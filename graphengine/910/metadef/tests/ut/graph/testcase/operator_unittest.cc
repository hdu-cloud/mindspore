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

#include "external/graph/operator.h"
#include "external/graph/operator_factory.h"
#include "external/graph/attr_value.h"
#include "graph/ge_attr_value.h"
#include "graph/operator_impl.h"
#include "external/graph/tensor.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/op_desc_impl.h"
#include "graph/tensor_type_impl.h"
#include "graph_builder_utils.h"
#include <string.h>
#include "graph/utils/tensor_utils.h"
#include "graph/compute_graph_impl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "inc/external/graph/graph.h"
#include "external/graph/operator_reg.h"
#include "checker/summary_checker.h"
#include "checker/topo_checker.h"

#undef private
#undef protected

namespace ge {
namespace {
REG_OP(Foo01).OUTPUT(y, TensorType::NumberType()).OP_END_FACTORY_REG(Foo01);
REG_OP(Foo11).INPUT(x, TensorType::NumberType()).OUTPUT(y, TensorType::NumberType()).OP_END_FACTORY_REG(Foo11);
REG_OP(Foo02).OUTPUT(x, TensorType::NumberType()).OUTPUT(y, TensorType::NumberType()).OP_END_FACTORY_REG(Foo02);
REG_OP(Foo22)
    .INPUT(m, TensorType::NumberType())
    .INPUT(n, TensorType::NumberType())
    .OUTPUT(x, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .OP_END_FACTORY_REG(Foo22);
REG_OP(DFoo22)
    .INPUT(m, TensorType::NumberType())
    .DYNAMIC_INPUT(n, TensorType::NumberType())
    .OUTPUT(x, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .OP_END_FACTORY_REG(DFoo22);

/*
 *                  ┌──────────────────────┐
 *                  │ cond                 │
 *  data   const    │      const           │
 *    │     │       │        |             │
 *    └──┬──┘   ┌───┤ data──add──netoutput │
 *       │      │   │                      │
 *      while  ─┤   ├──────────────────────┤
 *       │      │   ├──────────────────────┴─┐
 *       │      └───┤ body                   │
 *    netoutput     │ data──reshape──netoutpu│
 *                  │        |               │
 *                  │       const            │
 *                  └────────────────────────┘
 */
ComputeGraphPtr BuildWhileGraphWithConstInput() {
  ut::GraphBuilder builder("main_graph");
  auto data_1 = builder.AddNode("data_1", "Data", 1, 1);
  auto const_1 = builder.AddNode("const_1", "Const", 1, 1);
  auto while_1 = builder.AddNode("while_1", "While", 2, 2);
  auto netoutput_1 = builder.AddNode("netoutput_1", "NetOutput", 1, 1);
  builder.AddDataEdge(data_1, 0, while_1, 0);
  builder.AddDataEdge(const_1, 0, while_1, 1);
  builder.AddDataEdge(const_1, 0, netoutput_1, 0);
  auto main_graph = builder.GetGraph();

  ut::GraphBuilder cond_builder("cond_graph");
  auto cond_data_1 = cond_builder.AddNode("cond_data_1", "Data", 1, 1);
  auto cond_const_1 = cond_builder.AddNode("cond_const_1", "Const", 1, 1);
  auto cond_add_1 = cond_builder.AddNode("cond_add_1", "Add", 2, 1);
  auto cond_netoutput_1 = cond_builder.AddNode("cond_netoutput_1", "NetOutput", 1, 1);
  cond_builder.AddDataEdge(cond_data_1, 0, cond_add_1, 0);
  cond_builder.AddDataEdge(cond_const_1, 0, cond_add_1, 1);
  cond_builder.AddDataEdge(cond_add_1, 0, cond_netoutput_1, 0);
  auto cond_graph = cond_builder.GetGraph();
  AttrUtils::SetInt(cond_data_1->GetOpDesc(), "_parent_node_index", static_cast<int>(0));
  cond_graph->SetParentGraph(main_graph);
  cond_graph->SetParentNode(main_graph->FindNode("while_1"));
  main_graph->FindNode("while_1")->GetOpDesc()->AddSubgraphName("cond_graph");
  main_graph->FindNode("while_1")->GetOpDesc()->SetSubgraphInstanceName(0, "cond_graph");
  main_graph->AddSubgraph("cond_graph", cond_graph);

  ut::GraphBuilder body_builder("body_graph");
  auto body_data_1 = body_builder.AddNode("body_data_1", "Data", 1, 1);
  auto body_const_1 = body_builder.AddNode("body_const_1", "Const", 1, 1);
  auto body_reshape_1 = body_builder.AddNode("body_reshape_1", "Reshape", 2, 1);
  auto body_netoutput_1 = body_builder.AddNode("body_netoutput_1", "NetOutput", 1, 1);
  body_builder.AddDataEdge(body_data_1, 0, body_reshape_1, 0);
  body_builder.AddDataEdge(body_const_1, 0, body_reshape_1, 1);
  body_builder.AddDataEdge(body_reshape_1, 0, body_netoutput_1, 0);
  auto body_graph = body_builder.GetGraph();
  AttrUtils::SetInt(cond_data_1->GetOpDesc(), "_parent_node_index", static_cast<int>(1));
  body_graph->SetParentGraph(main_graph);
  body_graph->SetParentNode(main_graph->FindNode("while_1"));
  main_graph->FindNode("while_1")->GetOpDesc()->AddSubgraphName("body_graph");
  main_graph->FindNode("while_1")->GetOpDesc()->SetSubgraphInstanceName(1, "body_graph");
  main_graph->AddSubgraph("body_graph", body_graph);
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value{1, 2, 3};
  std::vector<int64_t> shape{3};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  AttrUtils::SetTensor(body_const_1->GetOpDesc(), "value", tensor);
  auto op_desc = body_reshape_1->GetOpDesc();
  op_desc->impl_->input_name_idx_["x"] = 0;
  op_desc->impl_->input_name_idx_["shape"] = 1;

  return main_graph;
}
}  // namespace
class UtestOperater : public testing::Test {
 public:
  /*
   * Foo11
   *   |
   * Foo01
   */
  void CheckTopoGraph1(const Graph &graph) {
    auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
    ASSERT_NE(compute_graph, nullptr);
    ASSERT_EQ(gert::SummaryChecker(compute_graph).StrictAllNodeTypes({{"Foo01", 1}, {"Foo11", 1}}), "success");
    auto foo11_node = compute_graph->FindNode("foo11");
    ASSERT_NE(foo11_node, nullptr);
    ASSERT_EQ(gert::NodeTopoChecker(foo11_node).StrictConnectFrom({{"Foo01"}}), "success");
    auto foo01_node = compute_graph->FindNode("foo01");
    ASSERT_NE(foo01_node, nullptr);
    ASSERT_EQ(gert::NodeTopoChecker(foo01_node).StrictConnectTo(0, {{"Foo11"}}), "success");
  }

  /*
   *     Foo22
   *     /  |
   * Foo11  |
   *     \  |
   *     Foo02
   */
  void CheckTopoGraph2(const Graph &graph) {
    auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
    ASSERT_NE(compute_graph, nullptr);
    ASSERT_EQ(gert::SummaryChecker(compute_graph).StrictAllNodeTypes({{"Foo02", 1}, {"Foo11", 1}, {"Foo22", 1}}),
              "success");
    auto foo01_node = compute_graph->FindNode("foo02");
    ASSERT_NE(foo01_node, nullptr);
    ASSERT_EQ(gert::NodeTopoChecker(foo01_node).StrictConnectTo(0, {{"Foo11"}}), "success");
    ASSERT_EQ(gert::NodeTopoChecker(foo01_node).StrictConnectTo(1, {{"Foo22"}}), "success");
    auto foo11_node = compute_graph->FindNode("foo11");
    ASSERT_NE(foo11_node, nullptr);
    ASSERT_EQ(gert::NodeTopoChecker(foo11_node).StrictConnectFrom({{"Foo02"}}), "success");
    ASSERT_EQ(gert::NodeTopoChecker(foo11_node).StrictConnectTo(0, {{"Foo22"}}), "success");
    auto foo22_node = compute_graph->FindNode("foo22");
    ASSERT_NE(foo22_node, nullptr);
    ASSERT_EQ(gert::NodeTopoChecker(foo22_node).StrictConnectFrom({{"Foo11"}, {"Foo02"}}), "success");
  }
  /*
   *       Foo22
   *     /  |d0 \d1
   * Foo11  |   |
   *     \0 |0 /1
   *      Foo02
   */
  void CheckTopoGraph3(const Graph &graph) {
    auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
    ASSERT_NE(compute_graph, nullptr);
    ASSERT_EQ(gert::SummaryChecker(compute_graph).StrictAllNodeTypes({{"Foo02", 1}, {"Foo11", 1}, {"DFoo22", 1}}),
              "success");
    auto foo02_node = compute_graph->FindNode("foo02");
    ASSERT_NE(foo02_node, nullptr);
    ASSERT_EQ(gert::NodeTopoChecker(foo02_node).StrictConnectTo(0, {{"Foo11"}, {"DFoo22"}}), "success");
    ASSERT_EQ(gert::NodeTopoChecker(foo02_node).StrictConnectTo(1, {{"DFoo22"}}), "success");
    auto foo11_node = compute_graph->FindNode("foo11");
    ASSERT_NE(foo11_node, nullptr);
    ASSERT_EQ(gert::NodeTopoChecker(foo11_node).StrictConnectFrom({{"Foo02"}}), "success");
    ASSERT_EQ(gert::NodeTopoChecker(foo11_node).StrictConnectTo(0, {{"DFoo22"}}), "success");
    auto foo22_node = compute_graph->FindNode("foo22");
    ASSERT_NE(foo22_node, nullptr);
    ASSERT_EQ(gert::NodeTopoChecker(foo22_node).StrictConnectFrom({{"Foo11"}, {"Foo02"}, {"Foo02"}}), "success");
  }
};

TEST_F(UtestOperater, GetInputConstData) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 0, 1);
  auto data2 = builder.AddNode("Data2", "Data", 0, 1);
  auto enter = builder.AddNode("Enter", "Enter", 1, 1);
  auto transdata = builder.AddNode("Transdata", "Transdata", 2, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data2, 0, enter, 0);
  builder.AddDataEdge(data, 0, transdata, 0);
  builder.AddDataEdge(enter, 0, transdata, 1);
  builder.AddDataEdge(transdata, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  auto ge_tensor = std::make_shared<GeTensor>();
  auto op_desc = transdata->GetOpDesc();
  op_desc->impl_->input_name_idx_["Data"] = 0;
  op_desc->impl_->input_name_idx_["Enter"] = 1;
  auto tensor_desc = op_desc->MutableInputDesc(0);
  AttrUtils::SetTensor(tensor_desc, "_value", ge_tensor);

  Tensor tensor;
  auto op = OpDescUtils::CreateOperatorFromNode(transdata);
  ASSERT_EQ(op.GetInputConstData("Data", tensor), GRAPH_SUCCESS);
  ASSERT_EQ(op.GetInputConstData("Enter", tensor), GRAPH_FAILED);
}
/**                                   --------------------------
 *         const                     |   sub_data    sub_const  |
 *          |                        |         \    /           |
 *        case-----------------------|          Add             |
 *         |                         |          |               |
 *      netoutput                    |     sub_netoutput        |
 *                                   ---------------------------
 */
TEST_F(UtestOperater, GetInputConstData_subgraph) {
  auto ge_tensor = std::make_shared<GeTensor>();
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto const_node = builder.AddNode("Const", "Const", 0, 1);
  AttrUtils::SetTensor(const_node->GetOpDesc(), "value", ge_tensor);
  auto case_node = builder.AddNode("Case", "Case", 1, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(const_node, 0, case_node, 0);
  builder.AddDataEdge(case_node, 0, netoutput, 0);
  auto parent_graph = builder.GetGraph();

  ut::GraphBuilder sub_builder = ut::GraphBuilder("subgraph_graph");
  auto sub_data = sub_builder.AddNode("sub_data", "Data", 0, 1);
  auto sub_const = sub_builder.AddNode("sub_const", "Const", 0, 1);
  AttrUtils::SetTensor(sub_const->GetOpDesc(), "value", ge_tensor);
  auto add = sub_builder.AddNode("Add", "Add", 2, 1);
  auto sub_netoutput = sub_builder.AddNode("sub_netoutput", "NetOutput", 1, 0);
  sub_builder.AddDataEdge(sub_data, 0, add, 0);
  sub_builder.AddDataEdge(sub_const, 0, add, 1);
  sub_builder.AddDataEdge(add, 0, sub_netoutput, 0);

  auto subgraph = sub_builder.GetGraph();
  subgraph->SetParentNode(case_node);
  subgraph->SetParentGraph(parent_graph);
  parent_graph->AddSubgraph(subgraph->GetName(), subgraph);
  AttrUtils::SetInt(sub_data->GetOpDesc(), "_parent_node_index", 0);

  auto op_desc = add->GetOpDesc();
  op_desc->impl_->input_name_idx_["sub_data"] = 0;
  op_desc->impl_->input_name_idx_["sub_const"] = 1;

  Tensor tensor;
  auto op = OpDescUtils::CreateOperatorFromNode(add);
  ASSERT_EQ(op.GetInputConstData("sub_const", tensor), GRAPH_SUCCESS);
  ASSERT_EQ(op.GetInputConstData("sub_data", tensor), GRAPH_SUCCESS);
}


/*                                  -------------------------
*                                  |  partitioncall_0_const1* |
*     partitioncall_0--------------|             |           |
*           |                      |          netoutput      |
*           |                      --------------------------
*           |                       ------------------          -------------
*           |                      |        data      |        |     Pld     |
*           |                      |          |       |        |      |      |
*     partitioncall_1--------------|        FftsSub   |------->|   squeeze*  |
*                                  |          |       |        |      |      |
*                                  |      netoutput   |        |  netoutput  |
*                                   ------------------          -------------
*/
TEST_F(UtestOperater, GetInputConstData_cross_subgraph) {
    auto root_builder = ut::GraphBuilder("root");
    const auto &partitioncall_0 = root_builder.AddNode("partitioncall_0", "PartitionedCall", 0, 1);
    const auto &partitioncall_1 = root_builder.AddNode("partitioncall_1", "PartitionedCall", 1, 1);
    root_builder.AddDataEdge(partitioncall_0, 0, partitioncall_1, 0);
    const auto &root_graph = root_builder.GetGraph();

    // 1.build partitioncall_0 sub graph
    auto p1_sub_builder = ut::GraphBuilder("partitioncall_0_sub");
    const auto &partitioncall_0_const1 = p1_sub_builder.AddNode("partitioncall_0_const1", "Const", 0, 1);
    auto ge_tensor = std::make_shared<GeTensor>();
    ASSERT_TRUE(AttrUtils::SetTensor(partitioncall_0_const1->GetOpDesc(), "value", ge_tensor));

    const auto &partitioncall_0_netoutput = p1_sub_builder.AddNode("partitioncall_0_netoutput", "NetOutput", 1, 1);
    AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);
    p1_sub_builder.AddDataEdge(partitioncall_0_const1, 0, partitioncall_0_netoutput, 0);
    const auto &sub_graph = p1_sub_builder.GetGraph();
    sub_graph->SetParentNode(partitioncall_0);
    sub_graph->SetParentGraph(root_graph);
    partitioncall_0->GetOpDesc()->AddSubgraphName("f");
    partitioncall_0->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall_0_sub");

    // 2.build partitioncall_1 sub graph
    auto p2_sub_builder = ut::GraphBuilder("partitioncall_1_sub");
    const auto &partitioncall_1_data = p2_sub_builder.AddNode("partitioncall_1_data", "Data", 0, 1);
    AttrUtils::SetInt(partitioncall_1_data->GetOpDesc(), "_parent_node_index", 0);
    const auto &partitioncall_1_ffts_sub = p2_sub_builder.AddNode("FftsSub", "PartitionedCall", 1, 1);
    const auto &partitioncall_1_netoutput = p2_sub_builder.AddNode("partitioncall_1_netoutput", "NetOutput", 1, 1);
    p2_sub_builder.AddDataEdge(partitioncall_1_data, 0, partitioncall_1_ffts_sub, 0);
    p2_sub_builder.AddDataEdge(partitioncall_1_ffts_sub, 0, partitioncall_1_netoutput, 0);
    const auto &sub_graph2 = p2_sub_builder.GetGraph();
    sub_graph2->SetParentNode(partitioncall_1);
    sub_graph2->SetParentGraph(root_graph);
    partitioncall_1->GetOpDesc()->AddSubgraphName("f");
    partitioncall_1->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall_1_sub");


    // 2.1 build sgt sub graph
    auto sgt_sub_builder = ut::GraphBuilder("sgt_sub");
    const auto &sgt_pld = sgt_sub_builder.AddNode("sgt_plt", "PlaceHolder", 0, 1);
    const auto &sgt_squeeze = sgt_sub_builder.AddNode("sgt_squeeze", "Squeeze", 1, 1);
    sgt_squeeze->GetOpDesc()->impl_->input_name_idx_["sub_data"] = 0;
    const auto &sgt_netoutput = sgt_sub_builder.AddNode("sgt_netoutput", "NetOutput", 1, 1);
    sgt_sub_builder.AddDataEdge(sgt_pld, 0, sgt_squeeze, 0);
    sgt_sub_builder.AddDataEdge(sgt_squeeze, 0, sgt_netoutput, 0);
    const auto &sgt_sub_graph = sgt_sub_builder.GetGraph();
    sgt_sub_graph->SetParentNode(partitioncall_1_ffts_sub);
    sgt_sub_graph->SetParentGraph(sub_graph2);
    partitioncall_1_ffts_sub->GetOpDesc()->AddSubgraphName("sgt_sub");
    partitioncall_1_ffts_sub->GetOpDesc()->SetSubgraphInstanceName(0, "sgt_sub");


    sgt_pld->GetOpDesc()->SetExtAttr<NodePtr>("parentNode", partitioncall_1_data);


    root_graph->AddSubgraph(sgt_sub_graph->GetName(), sgt_sub_graph);
    root_graph->AddSubgraph(sub_graph->GetName(), sub_graph);
    root_graph->AddSubgraph(sub_graph2->GetName(), sub_graph2);

    auto op = OpDescUtils::CreateOperatorFromNode(sgt_squeeze);
    Tensor res;
    ASSERT_EQ(op.GetInputConstData("sub_data", res), GRAPH_SUCCESS);
}

TEST_F(UtestOperater, TestOperatorSetInputs) {
  ge::Operator dst_op = ge::Operator("Mul");
  ge::Operator src_op = ge::Operator("Add");
  dst_op.InputRegister("x1");
  dst_op.InputRegister("x2");
  dst_op.OutputRegister("y");

  src_op.InputRegister("x1");
  src_op.InputRegister("x2");
  src_op.OutputRegister("y");

  ASSERT_EQ(src_op.GetInputsSize(), 2U);
  ASSERT_EQ(dst_op.GetInputsSize(), 2U);
  // src_index is illegal
  (void) dst_op.SetInput(0U, src_op, 3U);
  ASSERT_EQ(src_op.GetInputsSize(), 2U);
  // dst_index is illegal
  (void) dst_op.SetInput(3U, src_op, 0U);
  ASSERT_EQ(src_op.GetInputsSize(), 2U);

  (void) dst_op.SetInput(1U, src_op, 0U);
  ASSERT_EQ(src_op.GetInputsSize(), 2U);

  ge::Operator null_op;
  (void) null_op.SetInput(1U, src_op, 0U);
  ASSERT_EQ(null_op.GetInputsSize(), 0U);

  std::string dst_name = "x1";
  (void) dst_op.SetInput(dst_name, src_op, 0U);
  ASSERT_EQ(dst_op.GetInputsSize(), 2U);
}

TEST_F(UtestOperater, AttrRegister_Float) {
  auto op = Operator("Data");
  std::string attr = "attr";
  float value = 1.0;
  op.AttrRegister(attr, value);
  float ret = 0;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_FLOAT_EQ(value, ret);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_ListFloat) {
  auto op = Operator("Data");
  std::string attr = "attr";
  std::vector<float> value = {1.0, 2.0};
  op.AttrRegister(attr, value);
  std::vector<float> ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_FLOAT_EQ(value[0], ret[0]);
  ASSERT_FLOAT_EQ(value[1], ret[1]);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_Int) {
  auto op = Operator("Data");
  std::string attr = "attr";
  int64_t value = 1;
  op.AttrRegister(attr, value);
  int64_t ret = 0;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(value, ret);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_ListInt) {
  auto op = Operator("Data");
  std::string attr = "attr";
  std::vector<int64_t> value = {1, 2};
  op.AttrRegister(attr, value);
  std::vector<int64_t> ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(value[0], ret[0]);
  ASSERT_EQ(value[1], ret[1]);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_String) {
  auto op = Operator("Data");
  std::string attr = "attr";
  std::string value = "on";
  op.AttrRegister(attr.c_str(), value.c_str());
  std::string ret;
  op.GetAttr(attr, ret);
  ASSERT_EQ(value, ret);
  op.AttrRegister(nullptr, value.c_str());
}

TEST_F(UtestOperater, AttrRegister_Bool) {
  auto op = Operator("Data");
  std::string attr = "attr";
  bool value = true;
  op.AttrRegister(attr, value);
  bool ret = false;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(value, ret);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_ListBool) {
  auto op = Operator("Data");
  std::string attr = "attr";
  std::vector<bool> value = {false, true};
  op.AttrRegister(attr, value);
  std::vector<bool> ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(value[0], ret[0]);
  ASSERT_EQ(value[1], ret[1]);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_Tensor) {
  auto op = Operator("Data");
  auto value = Tensor();
  std::string attr = "attr";
  op.AttrRegister(attr, value);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_ListTensor) {
  auto op = Operator("Data");
  std::vector<Tensor> value = {Tensor()};
  op.AttrRegister("attr", value);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_OpBytes) {
  auto op = Operator("Data");
  std::string attr = "attr";
  auto value = OpBytes{1, 2, 3};
  op.AttrRegister(attr, value);
  OpBytes ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(value[0], ret[0]);
  ASSERT_EQ(value[1], ret[1]);
  ASSERT_EQ(value[2], ret[2]);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_ListListInt) {
  auto op = Operator("Data");
  std::string attr = "attr";
  std::vector<std::vector<int64_t>> value = {{1, 2}, {3}};
  op.AttrRegister(attr, value);
  std::vector<std::vector<int64_t>> ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(value[0][0], ret[0][0]);
  ASSERT_EQ(value[0][1], ret[0][1]);
  ASSERT_EQ(value[1][0], ret[1][0]);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_ListDataType) {
  auto op = Operator("Data");
  std::string attr = "attr";
  std::vector<DataType> value = {DataType::DT_FLOAT, DataType::DT_INT64};
  op.AttrRegister(attr, value);
  std::vector<DataType> ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(value[0], ret[0]);
  ASSERT_EQ(value[1], ret[1]);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_DataType) {
  auto op = Operator("Data");
  std::string attr = "attr";
  auto value = DataType::DT_FLOAT;
  op.AttrRegister(attr, value);
  DataType ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(value, ret);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_NamedAttrs) {
  auto op = Operator("Data");
  std::string attr = "attr";
  auto value = NamedAttrs();
  value.SetName("name");
  op.AttrRegister(attr, value);
  NamedAttrs ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(value.GetName(), ret.GetName());
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_ListNamedAttrs) {
  auto op = Operator("Data");
  std::string attr = "attr";
  std::vector<NamedAttrs> value = {NamedAttrs()};
  value[0].SetName("name");
  op.AttrRegister(attr, value);
  std::vector<NamedAttrs> ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(value[0].GetName(), ret[0].GetName());
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_AscendString) {
  auto op = Operator("Data");
  std::string attr = "attr";
  auto value = AscendString("1");
  op.AttrRegister(attr, value);
  AscendString ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(std::string(value.GetString()), std::string(ret.GetString()));
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_AscendString2) {
  auto op = Operator("Data");
  std::string attr = "attr";
  auto value = AscendString("1");
  op.AttrRegister(attr, value);
  AscendString ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(std::string(value.GetString()), std::string(ret.GetString()));
  op.AttrRegister(attr, AscendString(""));
}

TEST_F(UtestOperater, AttrRegister_ListAscendString) {
  auto op = Operator("Data");
  std::string attr = "attr";
  std::vector<AscendString> value = {AscendString("1")};
  op.AttrRegister(attr, value);
  std::vector<AscendString> ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(std::string(value[0].GetString()), std::string(ret[0].GetString()));
  op.AttrRegister(nullptr, value);
  op.operator_impl_ = nullptr;
  op.AttrRegister(attr, value);
  value[0].name_ = nullptr;
  op.AttrRegister(attr, value);
}

TEST_F(UtestOperater, AttrRegister_ListString) {
  auto op = Operator("Data");
  std::string attr = "attr";
  std::vector<std::string> value;
  op.AttrRegister(attr, value);
  std::vector<std::string> ret;
  op.GetAttr(attr, ret);
  ASSERT_EQ(ret.size(), 0);
}

TEST_F(UtestOperater, RequiredAttrRegister_Success) {
  auto op = Operator("Data");
  op.RequiredAttrRegister("x");
  op.RequiredAttrRegister(nullptr);
  op.RequiredAttrRegister(std::string("y"));

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetIrAttrNames(), std::vector<std::string>({"x", "y"}));
}

TEST_F(UtestOperater, RequiredAttrWithTypeRegister_Success) {
  auto op = Operator("Cast");
  op.RequiredAttrWithTypeRegister("dst_type", "Int");
  op.AttrRegister("fake_ir_attr", true);
  op.RequiredAttrWithTypeRegister(nullptr, nullptr); // invalid case

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  AttrUtils::SetBool(op_desc, "fake_custom_attr", false);
  ASSERT_EQ(op_desc->GetIrAttrNames(), std::vector<std::string>({"dst_type", "fake_ir_attr"}));
  std::map<AscendString, AscendString> ir_attr_name_types;
  ASSERT_EQ(op.GetAllIrAttrNamesAndTypes(ir_attr_name_types), GRAPH_SUCCESS);
  std::map<AscendString, AscendString> ir_attr_name_types_expected{{"dst_type", "VT_INT"}, {"fake_ir_attr", "VT_BOOL"}};
  ASSERT_EQ(ir_attr_name_types, ir_attr_name_types_expected);
  ASSERT_TRUE(op_desc->HasAttr("dst_type"));
  ASSERT_TRUE(op_desc->HasAttr("fake_ir_attr"));
  ASSERT_TRUE(op_desc->HasAttr("fake_custom_attr"));
  ASSERT_TRUE(op_desc->HasRequiredAttr("dst_type"));
  ASSERT_FALSE(op_desc->HasRequiredAttr("fake_ir_attr"));
  ASSERT_FALSE(op_desc->HasRequiredAttr("fake_custom_attr"));
  std::map<AscendString, AscendString> setted_attr_name_types;
  ASSERT_EQ(op.GetAllAttrNamesAndTypes(setted_attr_name_types), GRAPH_SUCCESS);
  for (auto pair : setted_attr_name_types) {
    std::cout << pair.first.GetString() << "|" << pair.second.GetString() << std::endl;
  }
  std::map<AscendString, AscendString> setted_attr_name_types_expected{{"fake_ir_attr", "VT_BOOL"},
                                                                       {"fake_custom_attr", "VT_BOOL"}};
  ASSERT_EQ(setted_attr_name_types, setted_attr_name_types_expected);
}

TEST_F(UtestOperater, SubgraphRegister) {
  std::string name = "add";
  auto op = Operator("Add");
  bool dynamic = true;
  op.SubgraphRegister(name, dynamic);
  op.SubgraphRegister(nullptr, dynamic);
}

TEST_F(UtestOperater, SubgraphCountRegister) {
  std::string name = "add";
  auto op = Operator("Add");
  uint32_t count = 1;
  op.SubgraphCountRegister(name, count);
  op.SubgraphCountRegister(nullptr, count);
}

TEST_F(UtestOperater, SetSubgraphBuilder) {
  std::string name = "add";
  auto op = Operator("Add");
  uint32_t index = 1;
  SubgraphBuilder builder = []() { return Graph(); };
  op.SetSubgraphBuilder(name, index, builder);
  op.SetSubgraphBuilder(nullptr, index, builder);

  SubgraphBuilder builder2;
  builder2 = op.GetSubgraphBuilder(name);

  SubgraphBuilder builder3;
  builder3 = op.GetDynamicSubgraphBuilder(nullptr, 0);
  builder3 = op.GetDynamicSubgraphBuilder("add", 0);

  std::vector<std::string> vec_name;
  vec_name = op.GetSubgraphNames();
  EXPECT_EQ(vec_name.size(), 0);

  op.GetSubgraph(nullptr);
  Graph graph = op.GetSubgraph(name);
  EXPECT_EQ(graph.GetName(), "");

  graph = op.GetSubgraph("add");
  EXPECT_EQ(graph.GetName(), "");

  op.GetDynamicSubgraph(nullptr, 0);
  graph = op.GetDynamicSubgraph(name, 0);
  EXPECT_EQ(graph.GetName(), "");

  graph = op.GetDynamicSubgraph("add", 0);
  EXPECT_EQ(graph.GetName(), "");
}

TEST_F(UtestOperater, GetSubgraphImpl) {
  std::string name = "add";
  auto op = Operator("Add");
  op.GetSubgraphImpl(name);
  op.GetSubgraphImpl(nullptr);
}

TEST_F(UtestOperater, SetInput_Handler) {
  std::string name = "add";
  std::string type = "Add";
  int index = 1;
  auto op = Operator(type);
  auto handler = OutHandler(nullptr);
  op.SetInput(name.c_str(), handler);
  op.SetInput(nullptr, handler);
}

TEST_F(UtestOperater, GetOutput) {
  std::string name = "add";
  auto op = Operator("Add");
  op.GetOutput(name.c_str());
  op.GetOutput(nullptr);
}

TEST_F(UtestOperater, GetInputConstDataOut) {
  std::string name = "add";
  auto op = Operator("Add");
  Tensor a = Tensor();
  ASSERT_EQ(op.GetInputConstDataOut(name.c_str(), a), GRAPH_FAILED);
  ASSERT_EQ(op.GetInputConstDataOut(nullptr, a), GRAPH_FAILED);
}

TEST_F(UtestOperater, testTensorType) {
  DataType dt(DT_INT16);
  TensorType tt1(dt);
  EXPECT_EQ(*(tt1.tensor_type_impl_->dt_set_.cbegin()), DT_INT16);

  const std::initializer_list<DataType> types = {DT_INT8, DT_UINT8, DT_INT16};
  TensorType tt2(types);
  EXPECT_EQ(tt2.tensor_type_impl_->dt_set_.size(), 3);
}

TEST_F(UtestOperater, CreateOperator) {
  Operator op;
  OpDescPtr op_desc;

  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  EXPECT_FALSE(op.IsEmpty());
}

TEST_F(UtestOperater, testGetName) {
  AscendString name;
  Operator op("one_op", "add");
  op.GetName(name);

  const char *str = name.GetString();
  EXPECT_EQ(strcmp(str, "one_op"), 0);
}

TEST_F(UtestOperater, GetInputConstData2) {
  Operator op;
  OpDescPtr op_desc;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  std::string dst_name("dst_name");
  Tensor td;

  EXPECT_NE(op.GetInputConstData(dst_name, td), GRAPH_SUCCESS);
}

TEST_F(UtestOperater, GetNode) {
  Operator op;
  OpDescPtr op_desc;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  EXPECT_EQ(op.GetNode(), nullptr);
}

TEST_F(UtestOperater, GetInputDesc) {
  Operator op;
  OpDescPtr op_desc;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  std::string str_name = "input_desc_name";
  TensorDesc td = op.GetInputDesc(str_name);

  EXPECT_EQ(td.GetName().length(), 0);
}

TEST_F(UtestOperater, TryGetInputDesc) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  TensorDesc td;
  auto ret = op.TryGetInputDesc("input_name_1", td);
  EXPECT_EQ(ret, GRAPH_FAILED);

std:
  string str = "input_name_2";
  ret = op.TryGetInputDesc(str, td);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestOperater, UpdateInputDesc) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  TensorDesc td;
std:
  string str = "input_name";
  auto ret = op.UpdateInputDesc(str, td);
  EXPECT_EQ(ret, GRAPH_FAILED);

  ret = op.UpdateInputDesc("input_name", td);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestOperater, GetOutputDesc) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

std:
  string str = "output_name";
  TensorDesc td = op.GetOutputDesc(str);
  EXPECT_EQ(td.GetName().length(), 0);
}

TEST_F(UtestOperater, UpdateOutputDesc) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  TensorDesc td;
std:
  string str = "output_name";
  auto ret = op.UpdateOutputDesc(str, td);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestOperater, GetDynamicInputDesc) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

std:
  string str = "input_name";
  TensorDesc td_1 = op.GetDynamicInputDesc(str, 0);
  TensorDesc td_2 = op.GetDynamicInputDesc("input_name", 0);
  EXPECT_EQ(td_1.GetName().length(), 0);
  EXPECT_EQ(td_2.GetName().length(), 0);
}

TEST_F(UtestOperater, UpdateDynamicInputDesc) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  TensorDesc td_1;
std:
  string str = "input_name";
  auto ret = op.UpdateDynamicInputDesc(str, 0, td_1);
  EXPECT_EQ(ret, GRAPH_FAILED);
  ret = op.UpdateDynamicInputDesc("input_name", 0, td_1);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestOperater, GetDynamicOutputDesc) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  string str = "output_name";
  TensorDesc td_1 = op.GetDynamicOutputDesc(str, 0);
  TensorDesc td_2 = op.GetDynamicOutputDesc("output_name", 0);
  EXPECT_EQ(td_1.GetName().length(), 0);
  EXPECT_EQ(td_2.GetName().length(), 0);
}

TEST_F(UtestOperater, UpdateDynamicOutputDesc) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  TensorDesc td_1;
  string str = "output_name";
  auto ret = op.UpdateDynamicOutputDesc(str, 0, td_1);
  EXPECT_EQ(ret, GRAPH_FAILED);
  ret = op.UpdateDynamicOutputDesc("output_name", 0, td_1);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestOperater, InferShapeAndType) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestOperater, InferShapeAndType_param_invalid) {
  Operator op;
  op.operator_impl_ = std::make_shared<OperatorImpl>("name", "type");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST_F(UtestOperater, VerifyAllAttr) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, GRAPH_FAILED);

  ret = op.VerifyAllAttr(false);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestOperater, VerifyAllAttr_success) {
  Operator op;
  op.operator_impl_ = std::make_shared<OperatorImpl>("name", "type");

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestOperater, GetAllAttrNamesAndTypes) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  auto ret = op.GetAllAttrNamesAndTypes();
  EXPECT_EQ(ret.size(), 0);

  std::map<AscendString, AscendString> attr_name_types;
  auto ret_2 = op.GetAllAttrNamesAndTypes(attr_name_types);
  EXPECT_EQ(ret_2, GRAPH_FAILED);
}

TEST_F(UtestOperater, GetAllAttrs) {
  Operator op("name", "type");
  const std::string name = "name";
  std::string value("value");
  op.SetAttr(name, value);
  std::map<AscendString, AscendString> attr_name_types;
  auto ret = op.GetAllAttrNamesAndTypes(attr_name_types);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  auto attr_types = op.GetAllAttrNamesAndTypes();
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestOperater, FuncRegister) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  std::function<graphStatus(Operator &)> func;

  op.InferFuncRegister(func);

  if (op.operator_impl_->GetOpDescImpl() != nullptr) {
    printf("FuncRegister GetOpDescImpl is not null!\n");
    //auto ret1 = op.operator_impl_->GetOpDescImpl()->GetInferFunc();
    //EXPECT_EQ(ret1, nullptr);
  } else {
    printf("FuncRegister GetOpDescImpl is null!\n");
  }

  ASSERT_NE(op.operator_impl_, nullptr);
}

TEST_F(UtestOperater, FuncRegister2) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);
  std::function<graphStatus(Operator &)> func;

  op.InferFormatFuncRegister(func);
  op.VerifierFuncRegister(func);

  ASSERT_NE(op.operator_impl_, nullptr);
}

TEST_F(UtestOperater, GetDynamicInputNum) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  int num1 = op.GetDynamicInputNum("input_name");
  EXPECT_EQ(num1, 0);

  int num2 = op.GetDynamicInputNum(std::string("input_name"));
  EXPECT_EQ(num2, 0);
}

TEST_F(UtestOperater, GetDynamicOutputNum) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  int num1 = op.GetDynamicOutputNum("output_name");
  EXPECT_EQ(num1, 0);

  int num2 = op.GetDynamicOutputNum(std::string("output_name"));
  EXPECT_EQ(num2, 0);
}

TEST_F(UtestOperater, VerifyAll) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  auto ret = op.VerifyAll();
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestOperater, GetOperatorImplPtr) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  auto ret = op.GetOperatorImplPtr();
  EXPECT_NE(ret, nullptr);
}

TEST_F(UtestOperater, AddControlInput_Exception) {
  Operator op1;
  Operator op2;
  OpDescPtr op_desc_1;
  op2 = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  auto ret = op1.AddControlInput(op2);
  EXPECT_EQ(op1.IsEmpty(), ret.IsEmpty());
}

TEST_F(UtestOperater, SetAttr_char_array) {
  Operator op1;
  Operator op2;

  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("", optype_str);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  char_t *name = "data name";
  char_t *attr_value = "abc";

  op2 = op1.SetAttr(name, attr_value);
  std::string value1;

  op1.GetAttr(name, value1);
  printf("c_str1 = %s\n", value1.c_str());

  std::string value2;
  op2.GetAttr(name, value2);
  printf("c_str2 = %s\n", value2.c_str());
  EXPECT_EQ(value2, std::string("abc"));

  op1.SetAttr(nullptr, nullptr);
}

TEST_F(UtestOperater, SetAttr_AscendString) {
  Operator op1;
  Operator op2;

  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("", optype_str);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  char_t *name = "data name";
  AscendString attr_value = "abc";

  op1.SetAttr(nullptr, attr_value);
  op2 = op1.SetAttr(name, attr_value);

  std::string value2;
  op2.GetAttr(name, value2);
  EXPECT_EQ(value2, std::string("abc"));

  AscendString value3;
  EXPECT_EQ(op2.GetAttr(nullptr, value3), GRAPH_FAILED);
}

TEST_F(UtestOperater, SetAttr_vector_AscendString) {
  Operator op1;
  Operator op2;

  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("", optype_str);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  char_t *name = "data name";
  std::vector<AscendString> attr_value = {AscendString("abc"), AscendString("def")};

  op2 = op1.SetAttr(name, attr_value);

  std::vector<AscendString> value2;
  op2.GetAttr(name, value2);

  EXPECT_TRUE(value2.size() > 1);
  EXPECT_EQ(value2[1].GetString(), std::string("def"));

  op1.SetAttr(nullptr, attr_value);
  EXPECT_EQ(op2.GetAttr(nullptr, value2), GRAPH_FAILED);
}

TEST_F(UtestOperater, SetAttr_vector_AscendString2) {
  Operator op1;
  Operator op2;

  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("", optype_str);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  char_t *name = "data name";
  std::vector<AscendString> attr_value = {AscendString("abc"), AscendString("def")};

  op2 = op1.SetAttr(name, attr_value);

  std::vector<AscendString> value2;
  op2.GetAttr(name, value2);

  EXPECT_EQ(value2[1].GetString(), std::string("def"));

  op2 = op1.SetAttr(nullptr, attr_value);
  EXPECT_EQ(op2.GetAttr(nullptr, value2), GRAPH_FAILED);
}

TEST_F(UtestOperater, SetInputAttrByNameOfChar_tWithNull) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  op1.SetInputAttr("x", nullptr, "FIFO");
  AscendString enqueue_policy;
  const auto ret = op1.GetInputAttr("x", nullptr, enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetInputAttrByNameOfChar_tWithNullOp) {
  Operator op1;
  op1.SetInputAttr("x", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), "FIFO");
  AscendString enqueue_policy;
  const auto ret = op1.GetInputAttr("x", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetInputAttrByNameOfChar_tWithNullTensor) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  op1.SetInputAttr("x", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), "FIFO");
  AscendString enqueue_policy;
  const auto ret = op1.GetInputAttr("x", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetInputAttrByIndexOfChar_tWithNull) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  op1.SetInputAttr(0, nullptr, "FIFO");
  AscendString enqueue_policy;
  const auto ret = op1.GetInputAttr(0, nullptr, enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetInputAttrByIndexOfChar_tWithNullOp) {
  Operator op1;
  op1.SetInputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), "FIFO");
  AscendString enqueue_policy;
  const auto ret = op1.GetInputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetInputAttrByIndexOfChar_tWithNullTensor) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  op1.SetInputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), "FIFO");
  AscendString enqueue_policy;
  const auto ret = op1.GetInputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetInputAttrByIndexOfAscendString_tWithNull) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  AscendString policy("FIFO");
  op1.SetInputAttr(0, nullptr, policy);
  AscendString enqueue_policy;
  const auto ret = op1.GetInputAttr(0, nullptr, enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetInputAttrByIndexOfAscendString_tWithNullOp) {
  Operator op1;
  AscendString policy("FIFO");
  op1.SetInputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), policy);
  AscendString enqueue_policy;
  const auto ret = op1.GetInputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetInputAttrByIndexOfAscendString_tWithNullTensor) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  AscendString policy("FIFO");
  op1.SetInputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), policy);
  AscendString enqueue_policy;
  const auto ret = op1.GetInputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetInputAttrByNameOfAscendString_tWithNull) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  AscendString policy("FIFO");
  op1.SetInputAttr("x", nullptr, policy);
  AscendString enqueue_policy;
  const auto ret = op1.GetInputAttr("x", nullptr, enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetInputAttrByNameOfAscendString_tWithNullOp) {
  Operator op1;
  AscendString policy("FIFO");
  op1.SetInputAttr("x", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), policy);
  AscendString enqueue_policy;
  const auto ret = op1.GetInputAttr("x", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetInputAttrByNameOfAscendString_tWithNullTensor) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  AscendString policy("FIFO");
  op1.SetInputAttr("x", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), policy);
  AscendString enqueue_policy;
  const auto ret = op1.GetInputAttr("x", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetOutputAttrByNameOfChar_tWithNull) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  op1.SetOutputAttr("y", nullptr, "FIFO");
  AscendString enqueue_policy;
  const auto ret = op1.GetOutputAttr("y", nullptr, enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetOutputAttrByNameOfChar_tWithNullOp) {
  Operator op1;
  op1.SetOutputAttr("x", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), "FIFO");
  AscendString enqueue_policy;
  const auto ret = op1.GetOutputAttr("x", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetOutputAttrByNameOfChar_tWithNullTensor) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  op1.SetOutputAttr("x", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), "FIFO");
  AscendString enqueue_policy;
  const auto ret = op1.GetOutputAttr("x", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetOutputAttrByIndexOfChar_tWithNull) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  op1.SetOutputAttr(0, nullptr, "FIFO");
  AscendString enqueue_policy;
  const auto ret = op1.GetOutputAttr(0, nullptr, enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetOutputAttrByIndexOfChar_tWithNullOp) {
  Operator op1;
  op1.SetOutputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), "FIFO");
  AscendString enqueue_policy;
  const auto ret = op1.GetOutputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetOutputAttrByIndexOfChar_tWithNullTensor) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  op1.SetOutputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), "FIFO");
  AscendString enqueue_policy;
  const auto ret = op1.GetOutputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetOutputAttrByIndexOfAscendString_tWithNull) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  AscendString policy("FIFO");
  op1.SetOutputAttr(0, nullptr, policy);
  AscendString enqueue_policy;
  const auto ret = op1.GetOutputAttr(0, nullptr, enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetOutputAttrByIndexOfAscendStringWithNullOp) {
  Operator op1;
  AscendString policy("FIFO");
  op1.SetOutputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), "FIFO");
  AscendString enqueue_policy;
  const auto ret = op1.GetOutputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetOutputAttrByIndexOfAscendStringWithNullTensor) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  AscendString policy("FIFO");
  op1.SetOutputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), "FIFO");
  AscendString enqueue_policy;
  const auto ret = op1.GetOutputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetOutputAttrByNameOfAscendString_tWithNull) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  AscendString policy("FIFO");
  op1.SetOutputAttr("y", nullptr, policy);
  AscendString enqueue_policy;
  const auto ret = op1.GetOutputAttr("y", nullptr, enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetOutputAttrByNameOfAscendStringWithNullOp) {
  Operator op1;
  AscendString policy("FIFO");
  op1.SetOutputAttr("x", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), "FIFO");
  AscendString enqueue_policy;
  const auto ret = op1.GetOutputAttr("x", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetOutputAttrByNameOfAscendStringWithNullTensor) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  AscendString policy("FIFO");
  op1.SetOutputAttr("x", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), "FIFO");
  AscendString enqueue_policy;
  const auto ret = op1.GetOutputAttr("x", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_FAILED);
  EXPECT_EQ(enqueue_policy != "FIFO", true);
}

TEST_F(UtestOperater, SetOutputAttrByIndexOfAscendString) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  op1.SetOutputAttr(0, ATTR_NAME_FLOW_ATTR.c_str(), true);
  op1.SetOutputAttr(0, ATTR_NAME_FLOW_ATTR_DEPTH.c_str(), static_cast<int32_t>(8));
  AscendString policy("FIFO");
  op1.SetOutputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), policy);

  bool has_flow_attr = false;
  op1.GetOutputAttr(0, ATTR_NAME_FLOW_ATTR.c_str(), has_flow_attr);
  int32_t depth = 0;
  AscendString enqueue_policy;
  auto ret = op1.GetOutputAttr(0, ATTR_NAME_FLOW_ATTR_DEPTH.c_str(), depth);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = op1.GetOutputAttr("y", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(has_flow_attr, true);
  EXPECT_EQ(depth, 8);
  EXPECT_EQ(enqueue_policy, policy);
}

TEST_F(UtestOperater, SetOutputAttrByNameOfAscendString) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  op1.SetOutputAttr("y", ATTR_NAME_FLOW_ATTR.c_str(), true);
  op1.SetOutputAttr("y", ATTR_NAME_FLOW_ATTR_DEPTH.c_str(), static_cast<int32_t>(8));
  AscendString policy("FIFO");
  op1.SetOutputAttr("y", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), policy);

  bool has_flow_attr = false;
  op1.GetOutputAttr("y", ATTR_NAME_FLOW_ATTR.c_str(), has_flow_attr);
  int32_t depth = 0;
  AscendString enqueue_policy;
  auto ret = op1.GetOutputAttr("y", ATTR_NAME_FLOW_ATTR_DEPTH.c_str(), depth);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = op1.GetOutputAttr("y", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(has_flow_attr, true);
  EXPECT_EQ(depth, 8);
  EXPECT_EQ(enqueue_policy, policy);
}

TEST_F(UtestOperater, SetInputAttrByIndexOfAscendString) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  op1.SetInputAttr(0, ATTR_NAME_FLOW_ATTR.c_str(), true);
  op1.SetInputAttr(0, ATTR_NAME_FLOW_ATTR_DEPTH.c_str(), static_cast<int32_t>(8));
  AscendString policy("FIFO");
  op1.SetInputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), policy);

  bool has_flow_attr = false;
  op1.GetInputAttr(0, ATTR_NAME_FLOW_ATTR.c_str(), has_flow_attr);
  int32_t depth = 0;
  AscendString enqueue_policy;
  auto ret = op1.GetInputAttr(0, ATTR_NAME_FLOW_ATTR_DEPTH.c_str(), depth);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = op1.GetInputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(enqueue_policy, policy);
}

TEST_F(UtestOperater, SetInputAttrByNameOfAscendString) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  op1.SetInputAttr("x", ATTR_NAME_FLOW_ATTR.c_str(), true);
  op1.SetInputAttr("x", ATTR_NAME_FLOW_ATTR_DEPTH.c_str(), static_cast<int32_t>(8));
  AscendString policy("FIFO");
  op1.SetInputAttr("x", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), policy);

  bool has_flow_attr = false;
  auto ret = op1.GetInputAttr("x", ATTR_NAME_FLOW_ATTR.c_str(), has_flow_attr);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  int32_t depth = 0;
  AscendString enqueue_policy;
  ret = op1.GetInputAttr("x", ATTR_NAME_FLOW_ATTR_DEPTH.c_str(), depth);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = op1.GetInputAttr("x", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(enqueue_policy, policy);
}

TEST_F(UtestOperater, SetInputAttrByNameOfChar_t) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  op1.SetInputAttr("x", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), "FIFO");
  AscendString enqueue_policy;
  auto ret = op1.GetInputAttr("x", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(enqueue_policy, "FIFO");
}

TEST_F(UtestOperater, SetInputAttrByIndexOfChar_t) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  op1.SetInputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), "FIFO");
  AscendString enqueue_policy;
  auto ret = op1.GetInputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(enqueue_policy, "FIFO");
}

TEST_F(UtestOperater, SetOutputAttrByNameOfChar_t) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  op1.SetOutputAttr("y", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), "FIFO");
  AscendString enqueue_policy;
  auto ret = op1.GetOutputAttr("y", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(enqueue_policy == "FIFO", true);
}

TEST_F(UtestOperater, SetOutputAttrByIndexOfChar_t) {
  Operator op1;
  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op1", optype_str);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  op1.SetOutputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), "FIFO");
  AscendString enqueue_policy;
  auto ret = op1.GetOutputAttr(0, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), enqueue_policy);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(enqueue_policy, "FIFO");
}

TEST_F(UtestOperater, SetAttr_Tensor) {
  Operator op1;
  Operator op2;

  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("", optype_str);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  char_t *name = "data name";
  TensorDesc tensor_desc;
  std::vector<uint8_t> data = {1, 2, 3};
  Tensor attr_value(tensor_desc, data);

  op2 = op1.SetAttr(name, attr_value);

  Tensor value2;
  op2.GetAttr(name, value2);

  EXPECT_EQ(value2.GetSize(), attr_value.GetSize());
}

TEST_F(UtestOperater, SetAttr_Tensor2) {
  Operator op1;
  Operator op2;

  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("", optype_str);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  std::string name = "data name";
  TensorDesc tensor_desc;
  std::vector<uint8_t> data = {1, 2, 3};
  Tensor attr_value(tensor_desc, data);

  op2 = op1.SetAttr(name, attr_value);

  Tensor value2;
  op2.GetAttr(name, value2);

  EXPECT_EQ(value2.GetSize(), attr_value.GetSize());
}

TEST_F(UtestOperater, SetAttr_vector_Tensor) {
  Operator op1;
  Operator op2;

  op1 = Operator("Data");
  std::vector<Tensor> attr_value = {Tensor()};

  std::string name = "data name";
  op2 = op1.SetAttr(name, attr_value);

  std::vector<Tensor> value2;
  op2.GetAttr(name, value2);

  EXPECT_EQ(value2.size(), attr_value.size());
}

TEST_F(UtestOperater, SetAttr_vector_Tensor2) {
  Operator op1;
  Operator op2;

  op1 = Operator("Data");
  std::vector<Tensor> attr_value = {Tensor()};

  op1.SetAttr(nullptr, attr_value);

  char_t *name = "data name";
  op2 = op1.SetAttr(name, attr_value);

  std::vector<Tensor> value2;
  op2.GetAttr(nullptr, value2);
  op2.GetAttr(name, value2);

  EXPECT_EQ(value2.size(), attr_value.size());
}

TEST_F(UtestOperater, SetAttr_OpBytes) {
  Operator op1;
  Operator op2;

  op1 = Operator("Data");
  auto attr_value = OpBytes{1, 2, 3};

  op1.SetAttr(nullptr, attr_value);

  char_t *name = "data name";
  op2 = op1.SetAttr(name, attr_value);

  OpBytes value2;
  op2.GetAttr(nullptr, value2);
  op2.GetAttr(name, value2);

  EXPECT_EQ(value2.size(), attr_value.size());
}

TEST_F(UtestOperater, SetAttr_OpBytes2) {
  Operator op1;
  Operator op2;

  op1 = Operator("Data");
  auto attr_value = OpBytes{1, 2, 3};

  std::string name = "data name";
  op2 = op1.SetAttr(name, attr_value);

  OpBytes value2;
  op2.GetAttr(name, value2);
  EXPECT_EQ(value2.size(), attr_value.size());
}

TEST_F(UtestOperater, SetAttr_AttrValue) {
  Operator op;
  op = Operator("Data");
  AttrValue attr_value;
  op.SetAttr(nullptr, std::move(attr_value));

  char_t *name = "data name";
  op.SetAttr(name, 10);
  AttrValue attr_value2;

  EXPECT_EQ(op.GetAttr(name, attr_value2), GRAPH_SUCCESS);
  int64_t value = 0;
  attr_value2.GetValue<int64_t>(value);
  EXPECT_EQ(value, 10);

  char_t *name2 = "foo";
  op.SetAttr(name2, std::move(attr_value2));
  AttrValue attr_value3;
  op.GetAttr(name2, attr_value3);
  attr_value3.GetValue<int64_t>(value);
  EXPECT_EQ(value, 10);

  AttrValue attr_value4;
  op.GetAttr(std::string(name2), attr_value4);
  attr_value4.GetValue<int64_t>(value);
  EXPECT_EQ(value, 10);
}

TEST_F(UtestOperater, SetAttr_vector_DataType) {
  Operator op;
  op = Operator("Data");

  char_t *name = "data name";
  std::vector<ge::DataType> attr_value = {DT_INT8, DT_INT16, DT_INT32};

  op.SetAttr(nullptr, attr_value);
  op.SetAttr(name, attr_value);

  std::vector<ge::DataType> attr_value_out;

  op.GetAttr(nullptr, attr_value_out);
  op.GetAttr(name, attr_value_out);

  EXPECT_TRUE(attr_value_out.size() > 2);
  EXPECT_EQ(attr_value_out[2], DT_INT32);
}

TEST_F(UtestOperater, SetAttr_vector_DataType2) {
  Operator op;
  op = Operator("Data");

  std::string name = "data name";
  std::vector<ge::DataType> attr_value = {DT_INT8, DT_INT16, DT_INT32};

  op.SetAttr(name, attr_value);

  std::vector<ge::DataType> attr_value_out;

  op.GetAttr(name, attr_value_out);

  EXPECT_EQ(attr_value_out[1], DT_INT16);
}

TEST_F(UtestOperater, SetAttr_DataType) {
  Operator op;
  op = Operator("Data");

  char_t *name = "data name";
  ge::DataType attr_value = DT_INT16;

  op.SetAttr(nullptr, attr_value);
  op.SetAttr(name, attr_value);

  ge::DataType attr_value_out;

  op.GetAttr(nullptr, attr_value_out);
  op.GetAttr(name, attr_value_out);

  EXPECT_EQ(attr_value_out, DT_INT16);
}

TEST_F(UtestOperater, SetAttr_DataType2) {
  Operator op;
  op = Operator("Data");

  std::string name = "data name";
  ge::DataType attr_value = DT_INT16;

  op.SetAttr(name, attr_value);

  ge::DataType attr_value_out;

  op.GetAttr(name, attr_value_out);

  EXPECT_EQ(attr_value_out, DT_INT16);
}

TEST_F(UtestOperater, CopyOperators1) {

  ge::OpDescPtr add_op(new ge::OpDesc("add_0", "add"));
  std::shared_ptr<ge::ComputeGraph> compute_graph(new ge::ComputeGraph("test_graph"));
  auto add_node = compute_graph->AddNode(add_op);
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  ge::OpDescPtr add_op_2(new ge::OpDesc("add_2", "add"));
  std::shared_ptr<ge::ComputeGraph> compute_graph_2(new ge::ComputeGraph("test_graph_2"));
  auto add_node_2 = compute_graph->AddNode(add_op_2);
  Graph graph2 = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph_2);

  Operator op1("op1");
  Operator op2("op2");
  Operator op3("op3");
  graph.AddOp(op1);
  graph.AddOp(op2);
  graph.AddOp(op3);

  auto ret = GraphUtilsEx::CopyGraph(graph, graph2);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestOperater, CopyOperators2) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto transdata = builder.AddNode("Transdata", "Transdata", 2, 1);
  auto op_desc = transdata->GetOpDesc();
  op_desc->impl_->input_name_idx_["Data"] = 0;
  op_desc->impl_->input_name_idx_["Enter"] = 1;
  auto data = builder.AddNode("Data", "Data", 0, 1);
  auto data2 = builder.AddNode("Data2", "Data", 0, 1);

  Operator op1 = OpDescUtils::CreateOperatorFromNode(transdata);
  Operator op2 = OpDescUtils::CreateOperatorFromNode(data);
  Operator op3 = OpDescUtils::CreateOperatorFromNode(data2);

  ComputeGraphPtr compt_graph = builder.GetGraph();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compt_graph);
  graph.AddOp(op1);
  graph.AddOp(op2);
  graph.AddOp(op3);

  ut::GraphBuilder builder2 = ut::GraphBuilder("graph2");
  auto data3 = builder2.AddNode("Data3", "Data", 0, 1);
  ComputeGraphPtr compt_graph2 = builder2.GetGraph();
  Graph graph2 = GraphUtilsEx::CreateGraphFromComputeGraph(compt_graph2);

  auto ret = GraphUtilsEx::CopyGraph(graph, graph2);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestOperater, CopyOperators3) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto transdata = builder.AddNode("Transdata", "Transdata", 2, 1);
  auto op_desc = transdata->GetOpDesc();
  op_desc->impl_->input_name_idx_["Data"] = 0;
  op_desc->impl_->input_name_idx_["Enter"] = 1;
  auto data = builder.AddNode("Data", "Data", 0, 1);
  auto data2 = builder.AddNode("Data2", "Data", 0, 1);

  Operator op1 = OpDescUtils::CreateOperatorFromNode(transdata);
  Operator op2 = OpDescUtils::CreateOperatorFromNode(data);
  Operator op3 = OpDescUtils::CreateOperatorFromNode(data2);

  ComputeGraphPtr compt_graph = builder.GetGraph();
  Graph src_graph = GraphUtilsEx::CreateGraphFromComputeGraph(compt_graph);
  src_graph.AddOp(op1);
  src_graph.AddOp(op2);
  src_graph.AddOp(op3);

  ut::GraphBuilder builder2 = ut::GraphBuilder("graph2");
  auto data3 = builder2.AddNode("Data3", "Data", 0, 1);
  ComputeGraphPtr dst_compute_graph = builder2.GetGraph();
  Graph dst_graph = GraphUtilsEx::CreateGraphFromComputeGraph(dst_compute_graph);

  std::map<std::string, ge::Operator> src_op_list = {{string("op1"), op1}, {string("op2"), op2}, {string("op3"), op3}};
  std::map<std::string, ge::Operator> dst_op_list;

  std::map<ConstNodePtr, NodePtr> node_old_2_new;
  std::map<ConstOpDescPtr, OpDescPtr> op_desc_old_2_new;

  auto ret = OpDescUtils::CopyOperators(dst_compute_graph, node_old_2_new, op_desc_old_2_new, src_op_list, dst_op_list);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestOperater, TestCallbackToGetConstInputWithRuntimeInferenceContext) {
  // new a tensor
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value{1, 2, 3};
  std::vector<int64_t> shape{3};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);

  // define callback
  RuntimeInferenceContext runtime_ctx;
  OperatorImpl::GetConstInputOnRuntimeFun func_get_input_const =
      [&runtime_ctx](const ConstNodePtr &node, const size_t index, ge::GeTensorPtr &dst_tensor) {
        // from runtime context
        const auto in_data_anchor = node->GetInDataAnchor(static_cast<int32_t>(index));
        const auto out_data_anchor = in_data_anchor->GetPeerOutAnchor();
        auto peer_node = out_data_anchor->GetOwnerNode();
        GeTensorPtr tensor_value = nullptr;
        if (runtime_ctx.GetTensor(peer_node->GetOpDesc()->GetId(), out_data_anchor->GetIdx(), tensor_value) ==
            GRAPH_SUCCESS) {
          dst_tensor = tensor_value;
          return GRAPH_SUCCESS;
        }
        return ge::GRAPH_SUCCESS;
      };

  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto transdata = builder.AddNode("Transdata", "Transdata", 2, 1);
  auto op_desc = transdata->GetOpDesc();
  op_desc->impl_->input_name_idx_["Data"] = 0;
  op_desc->impl_->input_name_idx_["Enter"] = 1;
  auto data = builder.AddNode("Data", "Data", 0, 1);
  auto data2 = builder.AddNode("Data2", "Data", 0, 1);
  GraphUtils::AddEdge(data->GetOutDataAnchor(0), transdata->GetInDataAnchor(0));
  GraphUtils::AddEdge(data2->GetOutDataAnchor(0), transdata->GetInDataAnchor(1));
  Operator op1 = OpDescUtils::CreateOperatorFromNode(transdata);

  OpDescUtils::SetCallbackGetConstInputFuncToOperator(op1, func_get_input_const);

  int output_id = 0;
  runtime_ctx.SetTensor(data->GetOpDesc()->GetId(), output_id, std::move(tensor));

  Tensor test_tensor;
  std::string input_name = "Data";
  EXPECT_EQ(op1.GetInputConstData(input_name.c_str(), test_tensor), GRAPH_SUCCESS);
  EXPECT_EQ(test_tensor.GetSize(), value.size());  // 3 item in tensor
  auto const_data = reinterpret_cast<const uint8_t *>(test_tensor.GetData());
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(const_data[i], value[i]);
  }
}
/*
 * Foo11
 *   |
 * Foo01
 */
TEST_F(UtestOperater, SetInput_Success_SingleIOByStrName) {
  auto foo01 = op::Foo01("foo01");
  auto foo11 = op::Foo11("foo11");
  foo11.SetInput(std::string("x"), foo01);
  Graph graph("graph");
  graph.SetInputs({foo01});
  CheckTopoGraph1(graph);
}
/*
 * Foo11
 *   |
 * Foo01
 */
TEST_F(UtestOperater, SetInput_Success_SingleIOByCharName) {
  auto foo01 = op::Foo01("foo01");
  auto foo11 = op::Foo11("foo11");
  foo11.SetInput("x", foo01);
  Graph graph("graph");
  graph.SetInputs({foo01});
  CheckTopoGraph1(graph);
}

TEST_F(UtestOperater, SetInput_Failed_NullName) {
  auto foo01 = op::Foo01("foo01");
  auto foo11 = op::Foo11("foo11");
  foo11.SetInput(nullptr, foo01);
  Graph graph("graph");
  graph.SetInputs({foo01});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  ASSERT_NE(compute_graph, nullptr);
  ASSERT_EQ(gert::SummaryChecker(compute_graph).StrictAllNodeTypes({{"Foo01", 1}}), "success");
}

TEST_F(UtestOperater, GetOutput_Failed_NullName) {
  auto foo01 = op::Foo01("foo01");
  auto foo11 = op::Foo11("foo11");
  foo11.SetInput("", foo01.GetOutput(nullptr));
  Graph graph("graph");
  graph.SetInputs({foo01});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  ASSERT_NE(compute_graph, nullptr);
  ASSERT_EQ(gert::SummaryChecker(compute_graph).StrictAllNodeTypes({{"Foo01", 1}}), "success");
}
/*
 *     Foo22
 *     /  |
 * Foo11  |
 *     \  |
 *     Foo02
 */
TEST_F(UtestOperater, SetInput_Success_TwoByStrName) {
  auto foo02 = op::Foo02("foo02");
  auto foo11 = op::Foo11("foo11").SetInput(std::string("x"), foo02, std::string("x"));
  auto foo22 = op::Foo22("foo22")
                   .SetInput(std::string("m"), foo11, std::string("y"))
                   .SetInput(std::string("n"), foo02, std::string("y"));

  Graph graph("graph");
  graph.SetInputs({foo02});

  CheckTopoGraph2(graph);
}

TEST_F(UtestOperater, SetInput_Success_TwoByCharName) {
  auto foo02 = op::Foo02("foo02");
  auto foo11 = op::Foo11("foo11").SetInput("x", foo02, "x");
  auto foo22 = op::Foo22("foo22").SetInput("m", foo11, "y").SetInput("n", foo02, "y");

  Graph graph("graph");
  graph.SetInputs({foo02});

  CheckTopoGraph2(graph);
}

TEST_F(UtestOperater, SetInput_Success_TwoByIndex) {
  auto foo02 = op::Foo02("foo02");
  auto foo11 = op::Foo11("foo11").SetInput("x", foo02, 0U);
  auto foo22 = op::Foo22("foo22").SetInput("m", foo11, 0U).SetInput("n", foo02, 1U);

  Graph graph("graph");
  graph.SetInputs({foo02});

  CheckTopoGraph2(graph);
}

TEST_F(UtestOperater, SetInput_Success_TwoByStrNameIndexedHandler) {
  auto foo02 = op::Foo02("foo02");
  auto foo11 = op::Foo11("foo11").SetInput(std::string("x"), foo02.GetOutput(std::string("x")));
  auto foo22 = op::Foo22("foo22")
                   .SetInput(std::string("m"), foo11.GetOutput(std::string("y")))
                   .SetInput(std::string("n"), foo02.GetOutput(std::string("y")));

  Graph graph("graph");
  graph.SetInputs({foo02});

  CheckTopoGraph2(graph);
}

TEST_F(UtestOperater, SetInput_Success_TwoByCharNameIndexedHandler) {
  auto foo02 = op::Foo02("foo02");
  auto foo11 = op::Foo11("foo11").SetInput("x", foo02.GetOutput("x"));
  auto foo22 = op::Foo22("foo22").SetInput("m", foo11.GetOutput("y")).SetInput("n", foo02.GetOutput("y"));

  Graph graph("graph");
  graph.SetInputs({foo02});

  CheckTopoGraph2(graph);
}

/*
 *       Foo22
 *     /  |d0 \d1
 * Foo11  |   |
 *     \0 |0 /1
 *      Foo02
 */
TEST_F(UtestOperater, SetDynamicInput_Success_TwoByStrName) {
  auto foo02 = op::Foo02("foo02");
  auto foo11 = op::Foo11("foo11").SetInput(std::string("x"), foo02, std::string("x"));
  auto foo22 = op::DFoo22("foo22")
                   .create_dynamic_input_n(2)
                   .SetInput(std::string("m"), foo11, std::string("y"))
                   .SetInput(std::string("n"), 0, foo02, std::string("x"))
                   .SetInput(std::string("n"), 1, foo02, std::string("y"));

  Graph graph("graph");
  graph.SetInputs({foo02});

  CheckTopoGraph3(graph);
}
TEST_F(UtestOperater, SetDynamicInput_Success_TwoByCharName) {
  auto foo02 = op::Foo02("foo02");
  auto foo11 = op::Foo11("foo11").SetInput("x", foo02, "x");
  auto foo22 = op::DFoo22("foo22")
                   .create_dynamic_input_n(2)
                   .SetInput("m", foo11, "y")
                   .SetInput("n", 0, foo02, "x")
                   .SetInput("n", 1, foo02, "y");

  Graph graph("graph");
  graph.SetInputs({foo02});

  CheckTopoGraph3(graph);
}
/*
 *       DFoo22
 *    /    |     \
 * Foo11  Foo11  Foo11
 */
TEST_F(UtestOperater, SetDynamicInput_Success_SingleOutput) {
  auto foo01_0 = op::Foo01("foo01_0");
  auto foo01_1 = op::Foo01("foo01_1");
  auto foo01_2 = op::Foo01("foo01_2");
  auto foo22 = op::DFoo22("foo22")
                   .create_dynamic_input_n(2)
                   .SetInput("m", foo01_0)
                   .SetInput(std::string("n"), 0, foo01_1)
                   .SetInput("n", 1, foo01_2);

  Graph graph("graph");
  graph.SetInputs({foo01_0, foo01_1, foo01_2});

  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  ASSERT_NE(compute_graph, nullptr);
  ASSERT_EQ(gert::SummaryChecker(compute_graph).StrictAllNodeTypes({{"Foo01", 3}, {"DFoo22", 1}}), "success");
  auto foo22_node = compute_graph->FindNode("foo22");
  ASSERT_NE(foo22_node, nullptr);
  ASSERT_EQ(gert::NodeTopoChecker(foo22_node).StrictConnectFrom({{"Foo01"}, {"Foo01"}, {"Foo01"}}), "success");
}
TEST_F(UtestOperater, InputRegister_Success_ByString) {
  Operator op("Op", "Op");
  op.InputRegister(std::string("x"));
  op.InputRegister(std::string("y"));
  op.OptionalInputRegister(std::string("o"));
  op.DynamicInputRegister(std::string("d"), 0, true);

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  std::vector<std::pair<std::string, IrInputType>> expected{{"x", kIrInputRequired},
                                                            {"y", kIrInputRequired},
                                                            {"o", kIrInputOptional},
                                                            {"d", kIrInputDynamic}};
  ASSERT_EQ(op_desc->GetIrInputs(), expected);
}
TEST_F(UtestOperater, InputRegister_Success_ByChar) {
  Operator op("Op", "Op");
  op.InputRegister("x");
  op.InputRegister("y");
  op.OptionalInputRegister("o");
  op.DynamicInputRegister("d", 0, true);

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  std::vector<std::pair<std::string, IrInputType>> expected{{"x", kIrInputRequired},
                                                            {"y", kIrInputRequired},
                                                            {"o", kIrInputOptional},
                                                            {"d", kIrInputDynamic}};
  ASSERT_EQ(op_desc->GetIrInputs(), expected);
}
TEST_F(UtestOperater, InputRegister_Failed_NullptrChar) {
  Operator op("Op", "Op");
  op.InputRegister(nullptr);
  op.InputRegister(nullptr);
  op.OptionalInputRegister(nullptr);
  op.DynamicInputRegister(nullptr, 0, true);

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_TRUE(op_desc->GetIrInputs().empty());
}
TEST_F(UtestOperater, OutputRegister_Success) {
  Operator op("Op", "Op");
  op.OutputRegister(std::string("x"));
  op.OutputRegister("y");
  op.OutputRegister(std::string("m"));
  op.OutputRegister("n");

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllOutputsDescSize(), 4);
  EXPECT_EQ(op_desc->GetOutputIndexByName("x"), 0);
  EXPECT_EQ(op_desc->GetOutputIndexByName("y"), 1);
  EXPECT_EQ(op_desc->GetOutputIndexByName("m"), 2);
  EXPECT_EQ(op_desc->GetOutputIndexByName("n"), 3);
}
TEST_F(UtestOperater, DynamicInputRegister_Success_InsertCharDynamicInput) {
  Operator op("Op", "Op");
  op.InputRegister("x");
  op.InputRegister("y");
  op.InputRegister("z");
  op.DynamicInputRegisterByIndex("d", 2, 1);

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllInputsDesc().size(), 5);
  EXPECT_EQ(op_desc->GetInputIndexByName("x"), 0);
  EXPECT_EQ(op_desc->GetInputIndexByName("d0"), 1);
  EXPECT_EQ(op_desc->GetInputIndexByName("d1"), 2);
  EXPECT_EQ(op_desc->GetInputIndexByName("y"), 3);
  EXPECT_EQ(op_desc->GetInputIndexByName("z"), 4);
}
TEST_F(UtestOperater, DynamicInputRegister_Failed_Nullptr) {
  Operator op("Op", "Op");
  op.InputRegister("x");
  op.InputRegister("y");
  op.InputRegister("z");
  op.DynamicInputRegisterByIndex(nullptr, 2, 1);

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllInputsDesc().size(), 3);
  EXPECT_EQ(op_desc->GetInputIndexByName("x"), 0);
  EXPECT_EQ(op_desc->GetInputIndexByName("y"), 1);
  EXPECT_EQ(op_desc->GetInputIndexByName("z"), 2);
}
TEST_F(UtestOperater, DynamicInputRegister_Success_InsertStrDynamicInput) {
  Operator op("Op", "Op");
  op.InputRegister(std::string("x"));
  op.InputRegister(std::string("y"));
  op.InputRegister(std::string("z"));
  op.DynamicInputRegisterByIndex(std::string("d"), 2, 1);

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllInputsDesc().size(), 5);
  EXPECT_EQ(op_desc->GetInputIndexByName("x"), 0);
  EXPECT_EQ(op_desc->GetInputIndexByName("d0"), 1);
  EXPECT_EQ(op_desc->GetInputIndexByName("d1"), 2);
  EXPECT_EQ(op_desc->GetInputIndexByName("y"), 3);
  EXPECT_EQ(op_desc->GetInputIndexByName("z"), 4);
}
TEST_F(UtestOperater, DynamicInputRegister_Success_DuplicateIrInput) {
  Operator op("Op", "Op");
  op.InputRegister(std::string("x"));
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetIrInputsSize(), 1);

  op.DynamicInputRegister(std::string("x"), 0, true);
  op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_EQ(op_desc->GetIrInputsSize(), 1);
}
TEST_F(UtestOperater, GetDynamicInputNum_Success) {
  Operator op("Op", "Op");
  op.DynamicInputRegister("x", 5);
  op.DynamicInputRegister("y", 4);
  EXPECT_EQ(op.GetDynamicInputNum("x"), 5);
  EXPECT_EQ(op.GetDynamicInputNum("y"), 4);
  EXPECT_EQ(op.GetDynamicInputNum("z"), 0);
  EXPECT_EQ(op.GetDynamicInputNum(std::string("x")), 5);
  EXPECT_EQ(op.GetDynamicInputNum(std::string("y")), 4);
  EXPECT_EQ(op.GetDynamicInputNum(std::string("z")), 0);
  EXPECT_EQ(op.GetDynamicInputNum(nullptr), 0);
}
TEST_F(UtestOperater, DynamicOutputRegister_Success) {
  Operator op("Op", "Op");
  op.DynamicOutputRegister(std::string("x"), 2, true);
  op.DynamicOutputRegister("y", 3, true);

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetAllOutputsDescSize(), 5);
  EXPECT_EQ(op_desc->GetOutputIndexByName("x0"), 0);
  EXPECT_EQ(op_desc->GetOutputIndexByName("x1"), 1);
  EXPECT_EQ(op_desc->GetOutputIndexByName("y0"), 2);
  EXPECT_EQ(op_desc->GetOutputIndexByName("y1"), 3);
  EXPECT_EQ(op_desc->GetOutputIndexByName("y2"), 4);
}
TEST_F(UtestOperater, DynamicOutputRegister_duplicate_ir_output_Success) {
  Operator op("Op", "Op");
  op.DynamicOutputRegister("y", 0, true);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetIrOutputs().size(), 1);

  // register duplicated
  op.DynamicOutputRegister("y", 0, true);
  op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(op_desc->GetIrOutputs().size(), 1);
}
TEST_F(UtestOperater, GetDynamicOutputNum_Success) {
  Operator op("Op", "Op");
  op.DynamicOutputRegister("x", 5);
  op.DynamicOutputRegister("y", 4);
  EXPECT_EQ(op.GetDynamicOutputNum("x"), 5);
  EXPECT_EQ(op.GetDynamicOutputNum("y"), 4);
  EXPECT_EQ(op.GetDynamicOutputNum("z"), 0);
  EXPECT_EQ(op.GetDynamicOutputNum(std::string("x")), 5);
  EXPECT_EQ(op.GetDynamicOutputNum(std::string("y")), 4);
  EXPECT_EQ(op.GetDynamicOutputNum(std::string("z")), 0);
  EXPECT_EQ(op.GetDynamicOutputNum(nullptr), 0);
}

/*
 * Foo11
 *   |
 * Foo01
 */
TEST_F(UtestOperater, SetInput_Success_DoNotPassTensorAttrs) {
  auto foo01 = op::Foo01("foo01");
  auto foo11 = op::Foo11("foo11");
  foo01.SetOutputAttr(0, "foo01_output_attr", 1);
  foo11.SetInputAttr(0, "foo11_input_attr", 1);
  foo11.SetInput("x", foo01);
  Graph graph("graph");
  graph.SetInputs({foo01});
  CheckTopoGraph1(graph);
  int64_t value = 0;
  EXPECT_EQ(foo11.GetInputAttr(0, "foo01_output_attr", value), GRAPH_SUCCESS);
  EXPECT_EQ(value, 0);
  EXPECT_EQ(foo11.GetInputAttr(0, "foo11_input_attr", value), GRAPH_SUCCESS);
  EXPECT_EQ(value, 1);
}

TEST_F(UtestOperater, GetInputConstData_While_fail) {
  auto graph = BuildWhileGraphWithConstInput();
  auto nodes = graph->GetAllNodes();
  NodePtr reshape = nullptr;
  for (const auto &n : nodes) {
    if (n->GetType() == "Reshape") {
      reshape = n;
      break;
    }
  }
  ASSERT_NE(reshape, nullptr);
  Tensor tensor;
  auto op = OpDescUtils::CreateOperatorFromNode(reshape);
  ASSERT_EQ(op.GetInputConstData("x", tensor), GRAPH_FAILED);
  ASSERT_EQ(op.GetInputConstData("shape", tensor), GRAPH_SUCCESS);
  ASSERT_EQ(tensor.GetSize(), 3);
}
}  // namespace ge

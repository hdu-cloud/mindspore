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

#include "plugin/device/ascend/optimizer/enhancer/insert_transpose_for_sort.h"
#include <algorithm>
#include <string>

namespace mindspore {
namespace opt {
const size_t kSortOutputNum = 2;
const BaseRef InsertTransposeForSort::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto prim = std::make_shared<Primitive>(kSortOpName);
  return VectorRef({prim, Xs});
}

abstract::BaseShapePtr InferTransposeOutputShape(const abstract::BaseShapePtr &shape,
                                                 const std::vector<int64_t> &perm) {
  auto shapeptr = shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shapeptr);
  if (shapeptr->shape().size() != perm.size()) {
    MS_LOG(EXCEPTION) << "Length of input shape and perm must be same, bug got input shape: " << shape
                      << ", perm: " << perm;
  }
  ShapeVector in_shape = shapeptr->shape();
  ShapeVector in_shape_max = shapeptr->max_shape();
  ShapeVector in_shape_min = shapeptr->min_shape();
  ShapeVector out_shape;
  ShapeVector out_shape_max;
  ShapeVector out_shape_min;
  for (size_t i = 0; i < perm.size(); i++) {
    auto idx = LongToSize(perm[i]);
    out_shape.push_back(in_shape[idx]);
    if (!in_shape_min.empty() && !in_shape_max.empty()) {
      out_shape_max.push_back(in_shape_max[idx]);
      out_shape_min.push_back(in_shape_min[idx]);
    }
  }

  abstract::ShapePtr out_shape_ptr = std::make_shared<abstract::Shape>(out_shape, out_shape_min, out_shape_max);
  return out_shape_ptr;
}

CNodePtr InsertForInput(const FuncGraphPtr &func_graph, const CNodePtr &node, const std::vector<int64_t> &perm) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  std::vector<AnfNodePtr> new_inputs = {common::AnfAlgo::GetCNodePrimitiveNode(node)};

  auto in_node = common::AnfAlgo::GetInputNode(node, 0);
  auto type = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 0);
  auto in_shape = common::AnfAlgo::GetPrevNodeOutputDetailShape(node, 0);
  auto transpose_out_shape = InferTransposeOutputShape(in_shape, perm);

  auto ori_out_types = common::AnfAlgo::GetAllOutputInferDataTypes(node);

  std::vector<AnfNodePtr> trans_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimTranspose->name()))};
  trans_inputs.push_back(in_node);
  auto transpose = func_graph->NewCNode(trans_inputs);
  MS_EXCEPTION_IF_NULL(transpose);
  common::AnfAlgo::SetNodeAttr(kAttrPerm, MakeValue(perm), transpose);
  common::AnfAlgo::SetOutputTypeAndDetailShape({type}, {transpose_out_shape}, transpose.get());
  std::vector<std::string> transpose_input_names{"input_x", "input_perm"};
  common::AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(transpose_input_names), transpose);
  new_inputs.push_back(transpose);

  CNodePtr new_cnode = nullptr;
  // cnode changed so make a new cnode to differ from original one.
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  if (kernel_graph == nullptr) {
    new_cnode = std::make_shared<CNode>(*node);
  } else {
    new_cnode = kernel_graph->NewCNode(node);
  }
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_inputs(new_inputs);
  common::AnfAlgo::SetOutputTypeAndDetailShape(ori_out_types, {transpose_out_shape, transpose_out_shape},
                                               new_cnode.get());
  return new_cnode;
}

AnfNodePtr InsertForOutput(const FuncGraphPtr &func_graph, const CNodePtr &orig_node, const CNodePtr &node,
                           const std::vector<int64_t> &perm) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(orig_node);
  MS_EXCEPTION_IF_NULL(node);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto update_states = common::AnfAlgo::GetUpdateStateUsers(manager, orig_node);
  for (auto &update_state : update_states) {
    manager->SetEdge(update_state.first, update_state.second, node);
  }
  if (manager->node_users()[orig_node].empty()) {
    return node;
  }

  std::vector<AnfNodePtr> tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  auto out_num = common::AnfAlgo::GetOutputTensorNum(node);

  for (size_t output_idx = 0; output_idx < out_num; output_idx++) {
    auto tuple_getitem = CreatTupleGetItemNode(func_graph, node, output_idx);
    std::vector<AnfNodePtr> transpose_inputs;
    auto prim = std::make_shared<Primitive>(prim::kPrimTranspose->name());
    transpose_inputs.push_back(NewValueNode(prim));
    transpose_inputs.push_back(tuple_getitem);

    auto shape = common::AnfAlgo::GetOutputDetailShape(node, output_idx);
    auto type = common::AnfAlgo::GetOutputInferDataType(node, output_idx);
    auto transpose_out_shape = InferTransposeOutputShape(shape, perm);

    CNodePtr transpose = func_graph->NewCNode(transpose_inputs);
    MS_EXCEPTION_IF_NULL(transpose);
    common::AnfAlgo::SetNodeAttr(kAttrPerm, MakeValue(perm), transpose);
    common::AnfAlgo::SetOutputTypeAndDetailShape({type}, {transpose_out_shape}, transpose.get());
    std::vector<std::string> transpose_input_names{"input_x", "input_perm"};
    common::AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(transpose_input_names), transpose);
    tuple_inputs.push_back(transpose);
  }
  auto make_tuple = func_graph->NewCNode(tuple_inputs);
  return make_tuple;
}

AnfNodePtr InsertTranspose(const FuncGraphPtr &func_graph, const CNodePtr &node, const std::vector<size_t> &in_shape,
                           int64_t axis) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  axis = axis < 0 ? (SizeToLong(in_shape.size()) + axis) : axis;
  std::vector<int64_t> perm(in_shape.size(), 0);
  int64_t i = 0;
  std::generate(perm.begin(), perm.end(), [&] { return i++; });
  std::reverse(perm.begin() + LongToSize(axis), perm.end());

  auto new_sort = InsertForInput(func_graph, node, perm);
  common::AnfAlgo::SetNodeAttr("axis", MakeValue(IntToLong(-1)), new_sort);
  return InsertForOutput(func_graph, node, new_sort, perm);
}

const AnfNodePtr InsertTransposeForSort::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const EquivPtr &) const {
  if (node == nullptr || !AnfUtils::IsRealKernel(node)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = common::AnfAlgo::GetCNodeName(node);
  if (op_name != kSortOpName) {
    return nullptr;
  }

  auto axis = common::AnfAlgo::GetNodeAttr<int64_t>(node, "axis");
  auto in_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  if (axis == -1 || (axis > 0 && (LongToSize(axis) == in_shape.size() - 1))) {
    return nullptr;
  }
  // if the attr value axis of sort is not -1, or not the rank of sort, need insert transpose for correct execution
  return InsertTranspose(func_graph, cnode, in_shape, axis);
}
}  // namespace opt
}  // namespace mindspore

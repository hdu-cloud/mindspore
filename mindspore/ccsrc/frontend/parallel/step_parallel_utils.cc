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

#include "frontend/parallel/step_parallel_utils.h"

#include <cinttypes>
#include <algorithm>

#include <map>
#include <set>
#include <string>
#include <utility>
#include <queue>
#include <memory>

#include "utils/hash_map.h"
#include "mindspore/core/ops/core_ops.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/optimizer.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/graph_util/pipeline_split_utils.h"
#include "frontend/parallel/node_check.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/dynamic_creator.h"
#include "ir/param_info.h"
#include "ir/tensor.h"
#include "utils/trace_base.h"
#include "include/common/utils/comm_manager.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"
#include "mindspore/core/utils/parallel_node_check.h"

namespace mindspore {
namespace parallel {
using mindspore::tensor::Tensor;
size_t TOTAL_OPS = 0;
// g_RefMap, for CNode B input i is a RefKey[Parameter C],
// it will be one item in map with key: C, and value: (B, i)
std::map<AnfNodePtr, std::pair<AnfNodePtr, int64_t>> g_RefMap;

bool IsSomePrimitive(const CNodePtr &cnode, const std::string &name) {
  if (!cnode) {
    return false;
  }
  ValueNodePtr anf_node = cnode->input(0)->cast<ValueNodePtr>();
  if (!anf_node) {
    return false;
  }
  PrimitivePtr prim = anf_node->value()->cast<PrimitivePtr>();
  if (!prim) {
    return false;
  }
  return (prim->name() == name);
}

bool IsSomePrimitiveList(const CNodePtr &cnode, const std::set<string> &check_list) {
  return std::any_of(check_list.begin(), check_list.end(),
                     [cnode](const string &in) { return IsSomePrimitive(cnode, in); });
}

std::string GetPrimName(const CNodePtr &node) {
  auto prim = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(prim);
  return prim->name();
}

bool IsTraining(const FuncGraphManagerPtr &manager) {
  for (auto &fg : manager->func_graphs()) {
    if (fg->has_flag(kTraining)) {
      return true;
    }
  }
  return false;
}

TensorInfo GetInputsTensorInfo(const std::pair<AnfNodePtr, int64_t> &param_info) {
  auto user_cnode = param_info.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(user_cnode);
  auto user_input_index = param_info.second;
  OperatorInfoPtr op_info = user_cnode->user_data<OperatorInfo>();
  MS_EXCEPTION_IF_NULL(op_info);

  TensorInfo tensor_info;
  if (IsPrimitiveCNode(user_cnode, prim::kPrimSend)) {
    auto param_index = IntToSize(GetValue<int>(user_cnode->GetPrimalAttr(PARAM_INDEX)));
    tensor_info = op_info->inputs_tensor_info()[param_index];
  } else {
    size_t input_tensor_info_size = op_info->inputs_tensor_info().size();
    if (SizeToLong(input_tensor_info_size) <= user_input_index - 1) {
      MS_LOG(EXCEPTION) << op_info->name() << ": the size of inputs tensor info is " << input_tensor_info_size
                        << ", but the index is " << (user_input_index - 1);
    }
    tensor_info = op_info->inputs_tensor_info()[LongToSize(user_input_index - 1)];
  }
  return tensor_info;
}

AnfNodePtr GetRealKernelNode(const AnfNodePtr &node, int64_t get_item_index, CNodePtr *call_node) {
  if (IsPrimitiveCNode(node, prim::kPrimDepend) || IsPrimitiveCNode(node, prim::kPrimLoad) ||
      IsPrimitiveCNode(node, prim::kPrimCast)) {
    return GetRealKernelNode(node->cast<CNodePtr>()->input(1), get_item_index, call_node);
  }
  if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    auto cnode = node->cast<CNodePtr>();
    auto cur_get_item_index = LongToInt(GetTupleGetItemIndex(cnode));
    auto tuple_getitem_input = cnode->input(1);
    auto pass_through_node = GetRealKernelNode(tuple_getitem_input, cur_get_item_index, call_node);
    return GetRealKernelNode(pass_through_node, get_item_index, call_node);
  }
  if (get_item_index != -1 && IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
    auto make_tuple_cnode = node->cast<CNodePtr>();
    auto make_tuple_input = make_tuple_cnode->input(LongToSize(get_item_index + 1));
    return GetRealKernelNode(make_tuple_input, -1, call_node);
  }
  if (node->isa<CNode>() && IsValueNode<FuncGraph>(node->cast<CNodePtr>()->input(0))) {
    if (call_node != nullptr && *call_node == nullptr) {
      *call_node = node->cast<CNodePtr>();
    }
    auto cnode = node->cast<CNodePtr>();
    auto graph = GetValueNode<FuncGraphPtr>(cnode->input(0));
    auto output = GetRealKernelNode(graph->output(), get_item_index, call_node);
    MS_EXCEPTION_IF_NULL(output);
    if (output->isa<Parameter>()) {
      auto parameters = graph->parameters();
      auto pos_iter = std::find(parameters.begin(), parameters.end(), output);
      // If can't find in parameters, the parameter is a fv.
      if (pos_iter == parameters.end()) {
        return output;
      }
      auto pos = std::distance(parameters.begin(), pos_iter);
      return GetRealKernelNode(cnode->input(LongToSize(pos + 1)), -1, call_node);
    }
    return output;
  }
  return node;
}

AnfNodePtr CheckMakeTupleSplit(const AnfNodePtr &node, const FuncGraphManagerPtr &manager) {
  auto node_users = manager->node_users()[node];

  bool is_first_tensor_info = true;
  TensorInfo first_tensor_info;
  AnfNodePtr first_node;
  for (auto &node_user : node_users) {
    auto user_node = node_user.first->cast<CNodePtr>();
    if (!user_node->has_user_data<OperatorInfo>()) {
      continue;
    }
    auto tensor_info = GetInputsTensorInfo(node_user);
    if (is_first_tensor_info) {
      is_first_tensor_info = false;
      first_tensor_info = tensor_info;
      first_node = node_user.first;
      continue;
    }
    if (first_tensor_info == tensor_info) {
      continue;
    } else {
      MS_LOG(EXCEPTION) << "The node: " << node->DebugString()
                        << " has multiple users, but the TensorInfo are different";
    }
  }
  return first_node;
}

bool IsParallelCareNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  ValueNodePtr prim_node = cnode->input(0)->cast<ValueNodePtr>();
  if (prim_node == nullptr) {
    return false;
  }
  PrimitivePtr prim = prim_node->value()->cast<PrimitivePtr>();
  if (prim == nullptr) {
    return false;
  }
  if (IsInParallelBlackList(prim)) {
    MS_LOG(DEBUG) << "Parallel don't care node: " << prim->name();
    return false;
  }
  // get_next is not in the forward graph, we need mark the get_next as the forward node
  if (prim->name() == GET_NEXT || prim->name() == VIRTUAL_OUTPUT) {
    return true;
  }
  if ((prim->name() == CAST) && !cnode->has_user_data<OperatorInfo>()) {
    return false;
  }

  return cnode->in_forward_flag();
}

bool HasNestedMetaFg(const FuncGraphPtr &func_graph) {
  if (!IsPynativeParallel()) {
    return false;
  }
  AnfNodePtr ret = func_graph->get_return();
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  for (auto &node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimJ) || IsPrimitiveCNode(node, prim::kPrimVmap) ||
        IsPrimitiveCNode(node, prim::kPrimTaylor)) {
      return true;
    }
  }
  return false;
}

bool IsEmbedShardNode(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  AnfNodePtr ret = func_graph->get_return();
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  return std::any_of(all_nodes.begin(), all_nodes.end(), [&func_graph](const AnfNodePtr &node) {
    return IsPrimitiveCNode(node, prim::kPrimShard) && (node->func_graph() == func_graph);
  });
}

Shapes GetValueListShape(const AnfNodePtr &node) {
  Shapes shapes;
  std::vector<ValuePtr> inputs_seq;
  if (IsValueNode<ValueList>(node)) {
    inputs_seq = node->cast<ValueNodePtr>()->value()->cast<ValueListPtr>()->value();
  } else if (IsValueNode<ValueTuple>(node)) {
    inputs_seq = node->cast<ValueNodePtr>()->value()->cast<ValueTuplePtr>()->value();
  } else {
    MS_LOG(EXCEPTION) << "node is eigther ValueList or ValueTuple";
  }
  for (auto &ele : inputs_seq) {
    auto tensor = ele->cast<tensor::TensorPtr>();
    if (tensor == nullptr) {
      MS_LOG(WARNING) << "The value node is not a tensor";
      break;
    }
    auto one_shape = tensor->shape();
    shapes.push_back(one_shape);
  }
  return shapes;
}

bool IsControlFlowNode(const AnfNodePtr &node) {
  // Only switch or FuncCall nodes are control flow nodes
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // func node
  if (cnode->input(0)->isa<CNode>() && IsPrimitiveCNode(cnode->input(0), prim::kPrimSwitch)) {
    return true;
  }
  return false;
}

int64_t GetTupleGetItemIndex(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!cnode->input(TUPLE_GETITEM_INDEX_POS)->isa<ValueNode>()) {
    MS_LOG(EXCEPTION) << "The index of tuple getitem is not a value node";
  }

  ValuePtr tuple_index_value = GetValueNode(cnode->input(TUPLE_GETITEM_INDEX_POS));
  MS_EXCEPTION_IF_NULL(tuple_index_value);
  if (!tuple_index_value->isa<Int64Imm>()) {
    MS_LOG(EXCEPTION) << "The index of tuple getitem is not int64";
  }
  return tuple_index_value->cast<Int64ImmPtr>()->value();
}

void RedistributionNextNode(const AnfNodePtr &node, const FuncGraphManagerPtr &manager,
                            const NodeUsersMap &node_users_map, int64_t get_item_index,
                            std::vector<std::pair<std::pair<AnfNodePtr, int>, int>> *next_nodes) {
  MS_EXCEPTION_IF_NULL(node);
  if (node_users_map.count(node) == 0) {
    return;
  }
  auto node_set = node_users_map.at(node);
  for (auto &node_pair : node_set) {
    auto use_cnode = node_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(use_cnode);
    if (IsValueNode<FuncGraph>(use_cnode->input(0))) {
      auto fg = GetValueNode<FuncGraphPtr>(use_cnode->input(0));
      MS_EXCEPTION_IF_NULL(fg);
      auto fg_parameters = fg->parameters();
      auto param = fg_parameters[IntToSize(node_pair.second - 1)];
      MS_EXCEPTION_IF_NULL(param);
      RedistributionNextNode(param, manager, node_users_map, get_item_index, next_nodes);
      continue;
    }
    if (IsPrimitiveCNode(use_cnode, prim::kPrimTupleGetItem)) {
      get_item_index = LongToInt(GetTupleGetItemIndex(use_cnode));
    }
    // depend, auto monad and control flow op don't need to jump over
    if ((IsPrimitiveCNode(use_cnode, prim::kPrimDepend) && node_pair.second != 1) ||
        IsPrimitiveCNode(use_cnode, prim::kPrimUpdateState) || IsPrimitiveCNode(use_cnode, prim::kPrimSwitch)) {
      continue;
    }
    if (IsParallelCareNode(use_cnode) && use_cnode->has_user_data<OperatorInfo>()) {
      next_nodes->push_back(std::make_pair(node_pair, get_item_index));
    } else if (use_cnode->input(0)->isa<CNode>()) {
      continue;
    } else {
      // search recursively
      RedistributionNextNode(use_cnode, manager, node_users_map, get_item_index, next_nodes);
    }
  }
}

void RedistributionPreNode(const CNodePtr &cnode, const FuncGraphManagerPtr &manager,
                           std::vector<AnfNodePtr> *pre_nodes) {
  if (IsValueNode<FuncGraph>(cnode->input(0))) {
    auto fg = GetValueNode<FuncGraphPtr>(cnode->input(0));
    auto pre_node = GetRealKernelNode(fg->output(), -1, nullptr);
    if (!pre_node) {
      return;
    }
    auto pre_cnode = pre_node->cast<CNodePtr>();
    if (!pre_cnode) {
      return;
    }
    if (IsParallelCareNode(pre_cnode) && pre_cnode->has_user_data<OperatorInfo>()) {
      pre_nodes->push_back(pre_cnode);
    } else {
      RedistributionPreNode(pre_cnode, pre_cnode->func_graph()->manager(), pre_nodes);
    }
  }
  if (IsControlFlowNode(cnode)) {
    auto switch_cnode = cnode->input(0)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(switch_cnode);
    // extract true branch, false branch is usually also a control flow graph
    auto fg = GetValueNode<FuncGraphPtr>(switch_cnode->input(2));
    MS_EXCEPTION_IF_NULL(fg);
    auto fg_out = fg->output()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(fg_out);
    // control flow node, need enter graph to find redistribution pre node.
    RedistributionPreNode(fg_out, manager, pre_nodes);
  }
  if (IsPrimitiveCNode(cnode, prim::kPrimDepend) || IsPrimitiveCNode(cnode, prim::kPrimLoad) ||
      IsPrimitiveCNode(cnode, prim::kPrimCast) || IsPrimitiveCNode(cnode, prim::kPrimAllReduce)) {
    auto cnode_input = cnode->input(1)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode_input);
    RedistributionPreNode(cnode_input, manager, pre_nodes);
  }
  if (IsParallelCareNode(cnode) && cnode->has_user_data<OperatorInfo>()) {
    pre_nodes->push_back(cnode);
  }
}

Shapes GetNodeShape(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  Shapes shapes;
  if (IsValueNode<ValueList>(node) || IsValueNode<ValueTuple>(node)) {
    return GetValueListShape(node);
  }
  BaseShapePtr base_shape_ptr = node->Shape();
  if (node->isa<CNode>() && !IsControlFlowNode(node)) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode->input(0)->isa<CNode>()) {
      if (cnode->inputs().size() < 2) {
        MS_LOG(EXCEPTION) << "GetNodeShape: " << node->ToString() << " size is smaller than 2";
      }
      base_shape_ptr = cnode->input(1)->Shape();
    }
  }
  if (base_shape_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "GetNodeShape: " << node->ToString() << " shape_ptr is nullptr, full name is "
                      << node->fullname_with_scope();
  }
  auto tuple_shape_ptr = dyn_cast<abstract::SequenceShape>(base_shape_ptr);
  if (tuple_shape_ptr != nullptr) {
    auto tuple_shape = tuple_shape_ptr->shape();
    for (auto &shape : tuple_shape) {
      auto each_shape = dyn_cast<abstract::Shape>(shape);
      MS_EXCEPTION_IF_NULL(each_shape);
      shapes.push_back(each_shape->shape());
    }
  } else {
    auto shape_ptr = dyn_cast<abstract::Shape>(base_shape_ptr);
    MS_EXCEPTION_IF_NULL(shape_ptr);
    shapes.push_back(shape_ptr->shape());
  }
  return shapes;
}

RankList FindCommonMirrorGroup(const FuncGraphPtr &root) {
  auto parameters = root->parameters();
  for (auto &parameter : parameters) {
    auto param_ptr = parameter->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    if (!(param_ptr->has_default() && ParameterRequireGrad(param_ptr))) {
      continue;
    }
    size_t allow_repeat_num = 1;
    if (ParallelContext::GetInstance()->enable_parallel_optimizer() &&
        (!param_ptr->param_info() || param_ptr->param_info()->parallel_optimizer())) {
      if (ParallelContext::GetInstance()->optimizer_weight_shard_size() == -1) {
        MS_LOG(WARNING) << "The parameter :" << param_ptr->fullname_with_scope()
                        << " is fully shard by optimizer parallel,"
                           " thus cannot find common data parallel group for this rank";
        return {g_device_manager->global_rank()};
      }
      allow_repeat_num = size_t(ParallelContext::GetInstance()->optimizer_weight_shard_size());
    }
    if (IsFullySplitParameter(param_ptr, allow_repeat_num)) {
      MS_LOG(WARNING) << "The parameter :" << param_ptr->fullname_with_scope()
                      << " is fully shard, thus cannot find common data parallel group for this rank";
      return {g_device_manager->global_rank()};
    }
  }
  AnfNodePtr ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<int64_t> common_group_list;
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  bool is_first_group = true;
  for (auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimMirror) && !IsPrimitiveCNode(node, prim::kPrimMirrorMicroStep) &&
        !IsPrimitiveCNode(node, prim::kPrimMirrorMiniStep)) {
      continue;
    }
    auto prim = GetCNodePrimitive(node);
    if (!prim->HasAttr(GROUP)) {
      MS_LOG(EXCEPTION) << "The mirror operator dose not have group attr : " << node->DebugString();
    }
    std::string group_name = GetValue<std::string>(prim->GetAttr(GROUP));
    std::vector<int64_t> group_list = g_device_manager->FindRankListByHashName(group_name);
    if (is_first_group) {
      common_group_list = group_list;
      is_first_group = false;
    } else {
      std::vector<int64_t> new_comm_group_list;
      (void)std::set_intersection(common_group_list.begin(), common_group_list.end(), group_list.begin(),
                                  group_list.end(), std::back_inserter(new_comm_group_list));
      common_group_list = new_comm_group_list;
    }
  }
  MS_LOG(INFO) << "The common mirror group is:" << common_group_list;
  return common_group_list;
}

std::string CreateInstanceName(const CNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsValueNode<Primitive>(node->input(0))) {
    MS_LOG(EXCEPTION) << "CreateInstanceName: " << node->ToString() << " doesn't have primitive";
  }
  std::string name_base = node->fullname_with_scope();
  std::string name = name_base + "_" + std::to_string(index);
  std::string instance_name = HashInstanceName(name);
  return instance_name;
}

void SetCommunicationOpGroupLabel(std::vector<AnfNodePtr> new_node_input) {
  if (new_node_input.empty()) {
    return;
  }

  auto prim_anf_node = new_node_input[0]->cast<ValueNodePtr>();
  auto prim = GetValueNode<PrimitivePtr>(prim_anf_node);
  MS_EXCEPTION_IF_NULL(prim);

  auto attrs = prim->attrs();
  auto iter = attrs.find(GROUP);
  if (iter != attrs.end()) {
    auto value = iter->second;
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<StringImm>()) {
      std::string hash_name = value->cast<StringImmPtr>()->value();
      MS_EXCEPTION_IF_NULL(g_device_manager);
      std::string rank_list_name = g_device_manager->FindRankListNameByHashName(hash_name);
      (void)prim->AddAttr(GROUP_RANKS, MakeValue(rank_list_name));
    }
  }
}

std::vector<AnfNodePtr> ReplaceOpInput(const Operator &replace_op, const std::string &instance_name,
                                       const CNodePtr &node) {
  OperatorArgs arg_replace_op = replace_op.second;
  ValuePtr pyop_instance = CreateOpInstance(arg_replace_op.first, replace_op.first, instance_name);
  if (pyop_instance == nullptr) {
    MS_LOG(EXCEPTION) << "Failure: " << replace_op.first << " CreateOpInstance failed";
  }
  OperatorParams params = arg_replace_op.second;
  if (node->inputs().size() < 2) {
    // GetNext operator dose not has input
    if (node->inputs().size() == 1) {
      return {NewValueNode(pyop_instance)};
    }
    MS_LOG(EXCEPTION) << "Failure: " << node->ToString() << " size is smaller than 2";
  }
  std::vector<AnfNodePtr> replace_input = {NewValueNode(pyop_instance), node->input(1)};

  if (replace_op.first == EMBEDDING_LOOKUP) {
    replace_input = {NewValueNode(pyop_instance), node->input(1), node->input(2)};
  }

  if (!params.empty()) {
    Param param_first = *(params.begin());
    int64_t first_position = param_first.second;
    if (first_position == 1) {
      replace_input.pop_back();
    }
    for (auto &param : params) {
      AnfNodePtr val = NewValueNode(param.first.second);
      if (val == nullptr) {
        MS_LOG(EXCEPTION) << "Failure:val is nullptr";
      }
      int64_t position = param.second;
      (void)replace_input.insert(replace_input.cbegin() + position, val);
    }
  } else if (replace_op.first == SYNC_BATCH_NORM) {
    for (size_t i = 2; i < node->inputs().size(); ++i) {
      replace_input.push_back(node->input(i));
    }
  }
  SetCommunicationOpGroupLabel(replace_input);
  return replace_input;
}

void SetStridedSliceSplitStrategy(const std::vector<AnfNodePtr> &all_nodes) {
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!IsPrimitiveCNode(cnode, prim::kPrimStridedSlice)) {
      continue;
    }
    auto slice_prim = GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(slice_prim);
    if (slice_prim->HasAttr(FUNC_GRAPH_FLAG_STRIDED_SLICE)) {
      if (slice_prim->HasAttr(INTERLEAVED_NUM) &&
          GetValue<int64_t>(slice_prim->GetAttr(INTERLEAVED_NUM)) == MICRO_INTERLEAVED_SIZE) {
        ParallelContext::GetInstance()->set_enable_micro_interleaved(true);
      }
      SetStridedSliceStrategy(cnode);
    }
  }
}

// Check the given tensor, return nullptr if the given type is not an TensorType
bool CheckTensorType(const TypePtr &node_type) {
  MS_EXCEPTION_IF_NULL(node_type);
  if (!node_type->isa<mindspore::TensorType>()) {
    return false;
  }
  return true;
}

// For the weight used by cast and matmul at the same time, like the followings
// weight1->mirror->cast1-> matmul1;
// weight1->add
// we will not insert the cast(FP32->FP16), as it will cause the input of the operator add to be changed to fp16.
AnfNodePtr GetChildCastNode(const AnfNodePtr &node_ptr, const NodeUsersMap &node_users_map) {
  std::queue<AnfNodePtr> visited;
  AnfNodePtr queue_node = nullptr;
  CNodePtr cnode = nullptr;
  AnfNodePtr node = nullptr;
  if (!node_ptr) {
    return nullptr;
  }
  auto users = node_users_map.at(node_ptr);
  for (auto &node_user : users) {
    cnode = node_user.first->cast<CNodePtr>();
    if (!cnode || !cnode->in_forward_flag()) {
      continue;
    }
    if (node_user.first) {
      visited.push(node_user.first);
    }
  }
  while (!visited.empty()) {
    queue_node = visited.front();
    visited.pop();
    cnode = queue_node->cast<CNodePtr>();
    // MAKE_TUPLE will not appear after the load in the forward graph
    if (IsSomePrimitive(cnode, MAKE_TUPLE)) {
      continue;
    } else if (IsInAllGatherNodeList(cnode) || IsSomePrimitiveList(cnode, {LOAD, RESHAPE})) {
      auto node_set = node_users_map.at(queue_node);
      for (auto &node_user : node_set) {
        visited.push(node_user.first);
      }
    } else if (!IsSomePrimitive(cnode, CAST)) {
      MS_LOG(INFO) << "The weight's users including the non cast node So "
                   << "will not insert cast for this parameter " << node_ptr->DebugString();
      return nullptr;
    } else if (!node) {
      node = queue_node;
    }
  }
  return node;
}
// Given the cnode ptr, find its users until we find the computation node, then return the type of the
// computation node. This function is used to find the target type for CreateFP16Cast. Only returns the target type if
// it is float16, and the source node is float32. If the situation is not matched, then return the nullptr.
TypePtr FindChildCastWithFP32ToFP16(const CNodePtr &cnode_ptr, const NodeUsersMap &node_users_map) {
  auto node_ptr = cnode_ptr->cast<AnfNodePtr>();
  if (!node_ptr) {
    return nullptr;
  }
  auto cnode_inputs = cnode_ptr->inputs();
  if (cnode_inputs.size() < TWO_INPUT_SIZE) {
    return nullptr;
  }
  // As we execute the function IsWeightValidUsed when we start to insert the mirror, so the second parameter
  // is always the parameter.
  auto weight = cnode_inputs[1];
  if (!weight->isa<Parameter>()) {
    return nullptr;
  }
  MS_LOG(INFO) << "Start to search the weight params:" << weight->DebugString();

  AnfNodePtr node = GetChildCastNode(weight, node_users_map);
  if (!node) {
    return nullptr;
  }
  // get the output dtype of the operator
  auto node_type = node->Type();
  if (!CheckTensorType(node_type)) {
    return nullptr;
  }
  auto input_element_type = node_type->cast<mindspore::TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(input_element_type);
  auto source_node_type = node_ptr->Type();
  if (!CheckTensorType(source_node_type)) {
    return nullptr;
  }
  auto source_element_type = source_node_type->cast<mindspore::TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(input_element_type);
  // We only add cast operation when the source is fp32 type, and the users is fp16 type.
  if (source_element_type->type_id() == kNumberTypeFloat32 && input_element_type->type_id() == kNumberTypeFloat16) {
    return input_element_type;
  }
  return nullptr;
}

// Create a cast node given the current node and the previous node. The target type of the the cast is from the
// compute_node_type.
// Return the new cast node with pre_node as the inputs.
AnfNodePtr CreateFP16Cast(const CNodePtr &node, const AnfNodePtr &pre_node, const TypePtr &compute_node_type) {
  const char kOpsFunctionModelName[] = "mindspore.ops.functional";
  static py::object cast_prim = python_adapter::GetPyFn(kOpsFunctionModelName, "cast");
  const auto &adapter = py::cast<PrimitivePyAdapterPtr>(cast_prim);
  MS_EXCEPTION_IF_NULL(adapter);
  MS_EXCEPTION_IF_NULL(compute_node_type);
  auto prim = adapter->attached_primitive();
  if (prim == nullptr) {
    prim = std::make_shared<PrimitivePy>(cast_prim, adapter);
  }
  // Insert cast.
  auto type_node = NewValueNode(compute_node_type);
  type_node->set_abstract(compute_node_type->ToAbstract());
  auto new_node = node->func_graph()->NewCNode({NewValueNode(prim), pre_node, type_node});
  new_node->set_abstract(node->abstract());
  new_node->set_in_forward_flag(true);
  return new_node;
}

void LabelGenMaskMicro(const FuncGraphPtr &root) {
  AnfNodePtr ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  for (auto &node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimDropoutDoMask)) {
      auto gen_mask_node = RealInputNode(node->cast<CNodePtr>(), 2);
      if (gen_mask_node->isa<CNode>()) {
        gen_mask_node->cast<CNodePtr>()->set_primal_attrs(node->cast<CNodePtr>()->primal_attrs());
      }
    }
  }
}

void SetCastForParamNotRecompute(const std::vector<AnfNodePtr> &all_nodes) {
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimCast)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto cast_input = RealInputNode(cnode, 1);
    if (cast_input->isa<Parameter>()) {
      MS_LOG(INFO) << "Cast for parameter no needs recompute to avoid redundant trans_data operator";
      PrimitivePtr prim = GetValueNode<PrimitivePtr>(cnode->input(0)->cast<ValueNodePtr>());
      (void)prim->AddAttr("recompute", MakeValue(false));
    }
  }
}

std::shared_ptr<Value> GetAttrsFromAnfNode(const std::shared_ptr<AnfNode> &node, const string &key) {
  if (!node) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  auto prim = GetCNodePrimitive(cnode);
  if (prim && prim->HasAttr(key)) {
    return prim->GetAttr(key);
  }
  return nullptr;
}

bool IsSplittableOperator(const std::string &op_name) {
  // clang-format off
  static const std::set<std::string> splittable_op =
    {MATMUL, TRANSPOSE, GELU, FAST_GELU, TANH, SOFTMAX, SUB, MUL, DIV, RESHAPE, GREATER, LOG_SOFTMAX, ACTIVATION, PRELU,
     FLOORDIV, L2_NORMALIZE, ADD, MAXPOOL, AVGPOOL, MAXPOOLV2, VIRTUAL_DATA_SET, RELU, ONEHOT, DROPOUT_DO_MASK,
     REDUCE_MAX, REDUCE_MIN, ARGMAXWITHVALUE, ARGMINWITHVALUE, REDUCE_SUM, CONV2D, FUSE_BATCH_NORM, POOLING,
     MAX_POOL_WITH_ARGMAX, SIMPLE_MEAN, FLATTEN, BATCH_NORM, LAYER_NORM, BIAS_ADD, ASSIGN_SUB, COS, ACOS, EXP, STACK,
     LOG, REDUCE_MEAN, REAL_DIV, SIGMOID, POW, MAXIMUM, MINIMUM, EQUAL, NOT_EQUAL, LOGICALNOT, GATHERV2, SQRT, CONCAT,
     STRIDEDSLICE, GET_NEXT, CAST, NEG, SQUARE, BATCH_MATMUL, EXPAND_DIMS, SQUEEZE, SPARSE_GATHERV2, TILE, DROPOUT,
     SOFTMAX_CROSS_ENTROPY_WITH_LOGITS, SIGMOID_CROSS_ENTROPY_WITH_LOGITS, SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS,
     EMBEDDING_LOOKUP, FUSE_BATCH_NORM_EX, SPLIT, BROADCAST_TO, ABS, ACOSH, ASIN, ASINH, ATAN, ATANH, CEIL, COSH,
     EXPM1, LOG1P, SIN, SINH, TAN, RSQRT, INV, RECIPROCAL, ROUND, FLOOR, SIGN, ERF, ERFC, ZEROSLIKE, ONESLIKE,
     BESSELI0E, BESSELI1E, FLOORMOD, ASSIGN, ASSIGN_ADD, ATAN2, DIVNONAN, LOGICALAND, LOGICALOR, ELU, RELU6, RELUV2,
     SOFTPLUS, SOFTSIGN, GREATEREQUAL, LESSEQUAL, LESS, APPROXIMATEEQUAL, MOD, UNIQUE, UNSORTED_SEGMENT_SUM,
     UNSORTED_SEGMENT_MIN, REPEAT_ELEMENTS, TENSOR_DOT, RANGE, UNIFORM_CANDIDATE_SAMPLER, SLICE, SELECT, GATHERD,
     UNSORTED_SEGMENT_MAX, GATHER_ND, TOPK, SCATTER_UPDATE, SCATTER_ND_UPDATE, SCATTER_ND_ADD, SCATTER_ND_SUB,
     TENSOR_SCATTER_UPDATE, TENSOR_SCATTER_ADD, TENSOR_SCATTER_SUB, TENSOR_SCATTER_MAX, TENSOR_SCATTER_MIN,
     TENSOR_SCATTER_MUL, TENSOR_SCATTER_DIV, VIRTUAL_OUTPUT, CONV2D_BACK_PROP_INPUT, CONV2D_TRANSPOSE,
     MATMUL_DDS, DSD_MATMUL, UNIFORMREAL, RESIZE_BILINEAR, RESIZE_NEAREST_NEIGHBOR, FAST_GELU, IOU, BOUNDING_BOX_ENCODE,
     RANDOM_CHOICE_WITH_MASK, CROP_AND_RESIZE, ROI_ALIGN, REDUCE_PROD, REDUCE_ANY, REDUCE_ALL, ARGMAX, ARGMIN, ARGMINV2,
     UNSORTED_SEGMENT_PROD, SQUARE_SUM_ALL, MATMUL_DDS, DSD_MATMUL, UNIFORMREAL, RESIZE_BILINEAR, UNIQUE_CONSECUTIVE,
     RESIZE_NEAREST_NEIGHBOR, CUM_SUM, FAST_GELU, IOU, BOUNDING_BOX_ENCODE, RANDOM_CHOICE_WITH_MASK, CROP_AND_RESIZE,
     ROI_ALIGN, IS_FINITE, RINT, HSHRINK, HSIGMOID, MISH, SELU, SOFT_SHRINK, XLOGY, XDIVY, CUM_PROD, BITWISE_AND,
     BITWISE_OR, BITWISE_XOR, MUL_NO_NAN, TRUNCATE_DIV, TRUNCATE_MOD, INPLACE_ADD, INPLACE_SUB, INPLACE_UPDATE,
     L2_LOSS, LERP, ADDN, CDIST, SQUARED_DIFFERENCE, ERFINV, MASKED_FILL, SPLITV, GAMMA, KLDIV_LOSS, LIN_SPACE,
     CHECK_VALID, INVERT, SCATTER_ADD, SCATTER_DIV, SCATTER_MUL, SCATTER_MAX, SCATTER_MIN, SCATTER_SUB, UNIQUE_WITH_PAD,
     POPULATION_COUNT, IDENTITY};
  // clang-format on

  auto iter = splittable_op.find(op_name);
  return (iter != splittable_op.end());
}

bool IsAutoParallelCareNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  ValueNodePtr prim_node = cnode->input(0)->cast<ValueNodePtr>();
  if (prim_node == nullptr) {
    return false;
  }
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_node);
  if (prim == nullptr) {
    return false;
  }
  if (prim->name() == SEND || prim->name() == RECEIVE) {
    return false;
  }
  bool bool_result = IsParallelCareNode(cnode) && !IsSplittableOperator(prim->name());
  if (bool_result && (prim->name() != MAKE_TUPLE) && (prim->name() != MAKE_LIST)) {
    MS_LOG(EXCEPTION) << "Should implementing OperatorInfo for: " << prim->name();
  } else if (prim->name() == CAST) {
    if (cnode->fullname_with_scope().find(OPTIMIZER_SUB_STRING) != std::string::npos) {
      // Do not care CASTs from optimizer
      return false;
    }
    return true;
  }
  return IsParallelCareNode(cnode) && IsSplittableOperator(prim->name());
}

std::string GetDisOpName(const std::string &prim_name) {
  std::string op_name = prim_name;
  if (!prim_name.empty() && (prim_name[0] == '_')) {
    op_name = prim_name.substr(1);
  }
  return op_name + "Info";
}

OperatorInfoPtr OperatorInstanceByName(const std::string &name, const PrimitiveAttrs &attrs,
                                       const std::vector<Shapes> &shape_list) {
  if (shape_list.size() != 2) {
    MS_LOG(ERROR) << "The size of shape list is not 2";
    return nullptr;
  }
  if (name.length() == 0) {
    MS_LOG(EXCEPTION) << "Length of name is zero!";
  }
  std::string distribute_opname = GetDisOpName(name);
  OperatorInfoPtr operator_ =
    (OperatorInfoPtr)DynCreator::Instance().Create(distribute_opname, shape_list[0], shape_list[1], attrs, TOTAL_OPS);
  if (operator_ == nullptr) {
    MS_LOG(INFO) << "Create " << name << " failed";
    return nullptr;
  }
  std::string origin_name = operator_->name();
  operator_->set_name(origin_name + std::to_string(TOTAL_OPS));
  MS_LOG(INFO) << "Successfully created operator " << origin_name;
  ++TOTAL_OPS;
  return operator_;
}

OperatorInfoPtr OperatorInstance(const PrimitivePtr &prim, const PrimitiveAttrs &attrs,
                                 const std::vector<Shapes> &shape_list) {
  MS_EXCEPTION_IF_NULL(prim);
  OperatorInfoPtr operator_ = OperatorInstanceByName(prim->name(), attrs, shape_list);
  if (operator_ == nullptr) {
    if (IsInBatchParallelBlackList(prim)) {
      MS_LOG(EXCEPTION) << "Operator " << prim->name() << " is not supported yet in auto parallel mode.";
    }
    MS_LOG(INFO) << "Create " << prim->name() << " failed, use batch parallel";
    operator_ = OperatorInstanceByName(BATCH_PARALLEL, attrs, shape_list);
    MS_EXCEPTION_IF_NULL(operator_);
  }
  return operator_;
}

static Shapes GetRefKeyNodeShape(const AnfNodePtr &node, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);

  std::vector<AnfNodePtr> parameters = FindParameterByRefKeyNode(node, func_graph);
  if (parameters.size() != 1) {
    MS_LOG(EXCEPTION) << "Find parameter by ref key node failed";
  }

  Shapes input_shapes = GetNodeShape(parameters[0]);
  if (input_shapes.size() != 1) {
    MS_LOG(EXCEPTION) << "Get input shape failed";
  }

  MS_LOG(INFO) << "The parameter shape is " << ShapeToString(input_shapes[0]);
  return input_shapes;
}

std::vector<Shapes> ExtractShape(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  Shapes shape_inputs, shape_outputs;
  std::vector<Shapes> shape_all;
  std::vector<AnfNodePtr> all_inputs = node->inputs();

  const int concat_size = 2;
  size_t inputs_size = all_inputs.size();
  for (size_t i = 1; i < inputs_size; ++i) {
    Shapes input_shapes;
    AnfNodePtr input = all_inputs[i];
    if (HasAbstractMonad(input)) {
      continue;
    }
    if (IsValueNode<RefKey>(input)) {
      auto func_graph = node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      std::vector<AnfNodePtr> parameters = FindParameterByRefKeyNode(input, func_graph);
      if (parameters.size() != 1) {
        MS_LOG(EXCEPTION) << "Find parameter by ref key node failed";
      }
      std::pair<AnfNodePtr, int64_t> node_pair = std::make_pair(node, SizeToLong(i));
      g_RefMap[parameters[0]] = node_pair;
      input_shapes = GetRefKeyNodeShape(input, func_graph);
    } else if (input->isa<CNode>() || IsValueNode<Tensor>(input) || input->isa<Parameter>() ||
               ((IsValueNode<ValueList>(input) || IsValueNode<ValueTuple>(input)) && (inputs_size == concat_size))) {
      input_shapes = GetNodeShape(input);
    } else {
      continue;
    }
    if (input_shapes.size() != 1) {
      if (inputs_size == concat_size) {  // like concat
        shape_inputs = input_shapes;
        break;
      } else {
        MS_LOG(EXCEPTION) << "ExtractShape: Get input shape failed";
      }
    }
    shape_inputs.push_back(input_shapes[0]);
  }
  shape_all.push_back(shape_inputs);
  // extract out shape
  shape_outputs = GetNodeShape(node);
  shape_all.push_back(shape_outputs);
  return shape_all;
}

void AddNodeFusionInfo(const CNodePtr &node, const CNodePtr &comm_node, const std::string &backward_comm_name,
                       int32_t fusion_id) {
  if (fusion_id <= 0) {
    return;
  }
  if (GetValueNode<PrimitivePtr>(comm_node->input(0))->HasAttr(GROUP)) {
    auto comm_group = GetValue<std::string>(GetValueNode<PrimitivePtr>(comm_node->input(0))->GetAttr(GROUP));
    std::string fusion_key = backward_comm_name + "_" + comm_group + "_" + std::to_string(fusion_id);
    if (!IsPrimitiveCNode(node, prim::kPrimLoad)) {
      node->AddPrimalAttr(kRelatedFusionKey, MakeValue<std::string>(fusion_key));
      node->AddPrimalAttr(kRelatedNodeId, MakeValue<std::string>(node->UniqueId()));
      node->AddAttr(kRelatedCommNodeId, MakeValue<std::string>(comm_node->UniqueId()));
      return;
    }
    auto func_graph = node->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    auto manager = func_graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    auto user_set = manager->node_users()[node];
    for (auto &pair : user_set) {
      if (!IsPrimitiveCNode(pair.first)) {
        continue;
      }
      auto next_cnode = pair.first->cast<CNodePtr>();
      next_cnode->AddPrimalAttr(kRelatedFusionKey, MakeValue<std::string>(fusion_key));
      next_cnode->AddPrimalAttr(kRelatedNodeId, MakeValue<std::string>(node->UniqueId()));
      node->AddAttr(kRelatedCommNodeId, MakeValue<std::string>(comm_node->UniqueId()));
    }
  }
}

OperatorInfoPtr CreateOperatorInfo(const CNodePtr &cnode) {
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_EXCEPTION_IF_NULL(prim);

  auto shape_list = ExtractShape(cnode);
  if (shape_list.empty()) {
    MS_LOG(EXCEPTION) << "Node: " << cnode->DebugString() << " failed to extract shape.";
  }

  auto attrs = prim->attrs();
  OperatorInfoPtr op_info = OperatorInstance(prim, attrs, shape_list);
  MS_EXCEPTION_IF_NULL(op_info);

  // When the 'inputs' contains numerical values for some operators, these values should be extracted from
  // ANF graph
  auto &inputs = cnode->inputs();
  std::vector<ValuePtr> input_value;
  for (size_t index = 1; index < inputs.size(); ++index) {
    if (inputs[index]->isa<ValueNode>()) {
      input_value.push_back(GetValueNode(inputs[index]));
      continue;
    }
    (void)input_value.emplace_back(nullptr);
  }

  (*op_info).set_input_value(input_value);
  (*op_info).set_outputs_dtype(cnode->Type());
  (*op_info).set_cnode(cnode);
  return op_info;
}

void ExtendInputArgsAbstractShape(const AbstractBasePtr &args_abstract_item, size_t index) {
  auto args_abstract_item_shape = args_abstract_item->BuildShape();
  auto shape_ptr = dyn_cast<abstract::Shape>(args_abstract_item_shape);
  if (shape_ptr == nullptr) {
    MS_LOG(WARNING) << "The input " << index << " is not a tensor.";
    return;
  }
  auto shape_value = parallel::ToFullShape(shape_ptr->shape(), index);
  auto new_shape_item = std::make_shared<abstract::Shape>(shape_value);
  args_abstract_item->set_shape(new_shape_item);
}

ShapeVector ToFullShape(const ShapeVector &input_shape, size_t index) {
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  if (ParallelContext::GetInstance()->dataset_strategy().empty()) {
    auto shape_value = input_shape;
    if (!parallel::ParallelContext::GetInstance()->full_batch()) {
      auto comm_info = parallel::GetCommInfo();
      auto world_rank_size = comm_info.device_num / ParallelContext::GetInstance()->pipeline_stage_split_num();
      shape_value[0] = shape_value[0] * SizeToLong(world_rank_size);
    }
    return shape_value;
  }
  auto dataset_strategy = ParallelContext::GetInstance()->dataset_strategy();
  if (index >= dataset_strategy.size()) {
    MS_LOG(EXCEPTION) << "The input shapes size is not equal to dataset strategy size " << dataset_strategy.size();
  }
  auto dataset_strategy_item = dataset_strategy[index];
  if (input_shape.size() != dataset_strategy_item.size()) {
    MS_LOG(EXCEPTION) << "The input_shapes[" << index << "]'s size" << input_shape.size()
                      << " is not equal to dataset_strategy[" << index << "]'s size " << dataset_strategy_item.size();
  }
  ShapeVector shape_value;
  for (size_t i = 0; i < dataset_strategy_item.size(); ++i) {
    shape_value.push_back(input_shape[i] * dataset_strategy_item[i]);
  }
  return shape_value;
}

CommInfo GetCommInfo() {
  int64_t device_num = ParallelContext::GetInstance()->device_num();
  int64_t global_rank = ParallelContext::GetInstance()->global_rank();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string backend = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  std::string world_group;
  std::string communication_backend;
  if (backend == kAscendDevice || backend == kDavinciDevice) {
    world_group = HCCL_WORLD_GROUP;
    communication_backend = HCCL_BACKEND;
  } else if (backend == kGPUDevice) {
    world_group = NCCL_WORLD_GROUP;
    communication_backend = NCCL_BACKEND;
  } else {
    MS_LOG(EXCEPTION) << "Invalid communication backend: " << backend
                      << " for semi_auto_parallel/auto_parallel mode,"
                         " currently only support Ascend/GPU backend.";
  }
  uint32_t world_rank_size = 0;
  if (!CommManager::GetInstance().GetRankSize(world_group, &world_rank_size)) {
    MS_LOG(EXCEPTION) << "Get rank size failed";
  }

  if (!ParallelContext::GetInstance()->device_num_is_set()) {
    device_num = UintToInt(world_rank_size);
    MS_LOG(INFO) << "Get device num from communication model, the device num is  " << device_num;
  }
#if (!defined(_WIN32) && !defined(__APPLE__) && !(defined(ENABLE_TESTCASES) || defined(ENABLE_TEST)))
  if (ParallelContext::GetInstance()->device_num_is_set() && world_rank_size != device_num &&
      !ParallelContext::GetInstance()->hccl_test_available()) {
    // hccl_test_available is used when we compile graphs in real ascend card environment, but with hccl_test.
    MS_LOG(EXCEPTION) << "The device_num " << device_num << " set in the context is not consist with "
                      << world_rank_size << " devices you have"
                      << ". Please check your rank_table file(for Ascend) or host file(for GPU).";
  }
#endif
  uint32_t rank_id = 0;
  if (!ParallelContext::GetInstance()->global_rank_is_set()) {
    if (!CommManager::GetInstance().GetRankID(world_group, &rank_id)) {
      MS_LOG(EXCEPTION) << "Get rank id failed";
    }
    global_rank = UintToInt(rank_id);
    MS_LOG(INFO) << "Get global rank from communication model, the global rank is  " << global_rank;
  }
  CommInfo comm_info{device_num, global_rank, world_group, communication_backend};
  return comm_info;
}

bool IsPynativeParallel() {
  auto parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  auto execution_mode = MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE);
  return (execution_mode == kPynativeMode) && (parallel_mode == kSemiAutoParallel || parallel_mode == kAutoParallel);
}

bool IsAutoParallelCareGraph(const FuncGraphPtr &func_graph) {
  // compile graph order:
  // 1, ParallelParameterContextInitShape/Ckpt/Restore
  // 2, PynativeShard: find 'shard' node and set 'pynative_shard' flag for root graph
  // 3, PipelineSplit: insert virtual dataset
  // 4, StepAutoParallel
  // 5, StepParallel
  // if IsPynativeParallel() is true, it maybe has some graphs that we no care, so need to check 'pynative_shard' flag
  MS_EXCEPTION_IF_NULL(func_graph);
  if (func_graph->has_flag(kSkipAutoParallelCompile)) {
    return false;
  }

  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  std::string parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode != kAutoParallel && parallel_mode != kSemiAutoParallel) {
    return false;
  }

  if (IsPynativeParallel() && !func_graph->has_flag(kPynativeShard)) {
    return false;
  }
  return true;
}
}  // namespace parallel
}  // namespace mindspore

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
#include "plugin/device/ascend/optimizer/ir_fission/diag_fission.h"
#include <algorithm>
#include <memory>
#include <vector>
#include <set>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "frontend/optimizer/opt.h"
#include "backend/common/optimizer/helper.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kDiagInputNum = 1;
constexpr size_t kDiagInputMaxDim = 4;

template <typename T>
void SetAssistTensorData(void *data, const T &value, int64_t dims_size) {
  MS_EXCEPTION_IF_NULL(data);
  auto tensor_data = reinterpret_cast<T *>(data);
  for (size_t i = 0; i < static_cast<size_t>(dims_size); ++i) {
    tensor_data[(1 + static_cast<size_t>(dims_size)) * i] = value;
  }
}
}  // namespace

ValueNodePtr DiagFission::CreateAssistNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                           const ShapeVector &ori_shape) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  ShapeVector output_shape(ori_shape);
  ShapeValueDType dims = 1;
  for (size_t i = 0; i < ori_shape.size(); i++) {
    dims = dims * ori_shape[i];
  }
  (void)output_shape.insert(output_shape.cend(), ori_shape.cbegin(), ori_shape.cend());
  auto type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type, output_shape);
  AbstractBasePtr x_abstract;
  if (type == kNumberTypeInt32) {
    SetAssistTensorData<int32_t>(tensor->data_c(), 1, dims);
    x_abstract = std::make_shared<abstract::AbstractTensor>(kInt32, output_shape);
  } else if (type == kNumberTypeInt64) {
    SetAssistTensorData<int64_t>(tensor->data_c(), 1, dims);
    x_abstract = std::make_shared<abstract::AbstractTensor>(kInt64, output_shape);
  } else if (type == kNumberTypeFloat16) {
    SetAssistTensorData<float16>(tensor->data_c(), float16(static_cast<float>(1)), dims);
    x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat16, output_shape);
  } else if (type == kNumberTypeFloat32) {
    SetAssistTensorData<float>(tensor->data_c(), static_cast<float>(1), dims);
    x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat, output_shape);
  } else {
    MS_EXCEPTION(TypeError) << "The type of node [" << node->DebugString()
                            << "] must be int32, float16 or float32, but got " << node->Type()->ToString() << ".";
  }
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto assist_value_node = kernel_graph->NewValueNode(x_abstract, tensor);
  kernel_graph->AddValueNodeToGraph(assist_value_node);
  common::AnfAlgo::SetOutputInferTypeAndShape({type}, {output_shape}, assist_value_node.get());
  return assist_value_node;
}

const BaseRef DiagFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto diag_prim = std::make_shared<Primitive>(prim::kPrimDiag->name());
  return VectorRef({diag_prim, Xs});
}

const AnfNodePtr DiagFission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  auto diag_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(diag_cnode);

  auto type = common::AnfAlgo::GetOutputInferDataType(node, kIndex0);
  if (!CheckOpAICoreSupported(type)) {
    MS_LOG(INFO) << "Diag fission failed for aicore, check to aicpu.";
    return nullptr;
  }
  if (common::AnfAlgo::IsDynamicShape(diag_cnode)) {
    MS_LOG(EXCEPTION) << "Diag don't support dynamic shape, node: " << diag_cnode->fullname_with_scope();
  }
  if (diag_cnode->size() != kDiagInputNum + 1) {
    MS_LOG(INFO) << "The node " << diag_cnode->DebugString() << " is not equal to " << kDiagInputNum << " inputs";
    return nullptr;
  }
  auto input_shape = common::AnfAlgo::GetOutputInferShape(diag_cnode->inputs()[kIndex1], 0);
  if (input_shape.size() > kDiagInputMaxDim) {
    MS_EXCEPTION(ValueError) << "For Diag, rank of input should be less than 5, but got: " << input_shape.size();
  }
  std::vector<AnfNodePtr> new_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimDiag->name()))};
  auto assist_const = CreateAssistNode(graph, diag_cnode, input_shape);
  (void)new_inputs.insert(new_inputs.cend(), diag_cnode->inputs().cbegin() + 1, diag_cnode->inputs().cend());
  new_inputs.push_back(assist_const);
  CNodePtr new_cnode = NewCNode(new_inputs, graph);
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_abstract(diag_cnode->abstract());
  new_cnode->set_scope(diag_cnode->scope());
  if (kernel_graph != nullptr) {
    kernel_graph->AddValueNodeToGraph(assist_const);
    MS_LOG(INFO) << "Add assist tensor for diag op success.";
  }
  return new_cnode;
}

bool DiagFission::CheckOpAICoreSupported(const TypeId &type) const {
  if (type == kNumberTypeInt64) {
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    return context_ptr->get_param<bool>(MS_CTX_ENABLE_REDUCE_PRECISION);
  }

  const std::set<TypeId> aicore_supported_types = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeInt32};
  if (aicore_supported_types.find(type) == aicore_supported_types.end()) {
    return false;
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore

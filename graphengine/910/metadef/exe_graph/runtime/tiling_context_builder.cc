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
#include "exe_graph/runtime/tiling_context_builder.h"

#include "exe_graph/lowering/bg_kernel_context_extend.h"
#include "data_dependent_interpreter.h"
#include "graph/compute_graph.h"
#include "graph/operator.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils_ex.h"
#include "graph/debug/ge_util.h"
#include "graph/def_types.h"
#include "common/checker.h"
#include "graph/debug/ge_util.h"

namespace gert {
namespace {
void GetStorageShape(const ge::GeTensorDesc &tensor_desc, gert::StorageShape &storage_shape) {
  const auto &storage_dims = tensor_desc.GetShape().GetDims();
  for (const auto &dim : storage_dims) {
    (void)storage_shape.MutableStorageShape().AppendDim(dim);
  }
  const auto &origin_dims = tensor_desc.GetOriginShape().GetDims();
  for (const auto &dim : origin_dims) {
    (void)storage_shape.MutableOriginShape().AppendDim(dim);
  }
}
} // namespace

TilingContextBuilder &TilingContextBuilder::CompileInfo(void *compile_info) {
  compile_info_ = compile_info;
  return *this;
}
TilingContextBuilder &TilingContextBuilder::PlatformInfo(void *platform_info) {
  platform_info_ = platform_info;
  return *this;
}
TilingContextBuilder &TilingContextBuilder::TilingData(void *tiling_data) {
  outputs_[TilingContext::kOutputTilingData] = tiling_data;
  return *this;
}
TilingContextBuilder &TilingContextBuilder::Workspace(ContinuousVector *workspace) {
  outputs_[TilingContext::kOutputWorkspace] = workspace;
  return *this;
}
TilingContextBuilder &TilingContextBuilder::SpaceRegistry(const gert::OpImplSpaceRegistryPtr &space_registry) {
  space_registry_ = space_registry;
  return *this;
}

ge::graphStatus TilingContextBuilder::GetDependInputTensorAddr(const ge::Operator &op, const size_t input_idx,
                                                               TensorAddress &address) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GE_ASSERT_NOTNULL(op_desc);
  auto depend_tensor = ge::ComGraphMakeUnique<ge::Tensor>();
  depend_ge_tensor_holders_.emplace_back(std::move(depend_tensor));
  GE_ASSERT_NOTNULL(depend_ge_tensor_holders_.back());
  auto input_name = op_desc->GetInputNameByIndex(static_cast<uint32_t>(input_idx));
  GE_ASSERT_GRAPH_SUCCESS(op.GetInputConstData(input_name.c_str(), *(depend_ge_tensor_holders_.back().get())));
  address = depend_ge_tensor_holders_.back()->GetData();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingContextBuilder::BuildRtTensor(const ge::GeTensorDesc &tensor_desc,
                                                    ConstTensorAddressPtr address,
                                                    std::unique_ptr<uint8_t[]> &rt_tensor_holder) const {
  gert::StorageShape storage_shape;
  GetStorageShape(tensor_desc, storage_shape);

  rt_tensor_holder = ge::ComGraphMakeUnique<uint8_t[]>(sizeof(gert::Tensor));
  GE_ASSERT_NOTNULL(rt_tensor_holder, "Create context holder inputs failed.");
  auto rt_tensor = ge::PtrToPtr<uint8_t, gert::Tensor>(rt_tensor_holder.get());
  rt_tensor->SetDataType(tensor_desc.GetDataType());
  rt_tensor->MutableStorageShape() = storage_shape.GetStorageShape();
  rt_tensor->MutableOriginShape() = storage_shape.GetOriginShape();
  rt_tensor->MutableFormat().SetStorageFormat(tensor_desc.GetFormat());
  rt_tensor->MutableFormat().SetOriginFormat(tensor_desc.GetOriginFormat());
  (void)rt_tensor->MutableTensorData().SetAddr(address, nullptr);
  rt_tensor->MutableTensorData().SetPlacement(gert::kOnHost);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingContextBuilder::BuildRTInputTensors(const ge::Operator &op) {
  const auto node = ge::NodeUtilsEx::GetNodeFromOperator(op);
  auto shared_node = const_cast<ge::Node *>(node.get())->shared_from_this();
  const DataDependentInterpreter ddi(shared_node, space_registry_);
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);

  const size_t input_num = node->GetInDataNodesAndAnchors().size();
  for (size_t i = 0U; i < input_num; ++i) {
    TensorAddress address = nullptr;
    bool is_data_dependent = false;
    GE_ASSERT_SUCCESS(ddi.IsDataDependent(static_cast<int32_t>(i), is_data_dependent));
    if (is_data_dependent) {
      GE_ASSERT_GRAPH_SUCCESS(GetDependInputTensorAddr(op, i, address));
    }
    std::unique_ptr<uint8_t[]> tensor_holder;
    GE_ASSERT_GRAPH_SUCCESS(BuildRtTensor(op_desc->GetInputDesc(i), address, tensor_holder));
    rt_tensor_holders_.emplace_back(std::move(tensor_holder));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingContextBuilder::BuildRTOutputShapes(const ge::Operator &op) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GE_ASSERT_NOTNULL(op_desc);
  for (size_t i = 0U; i < op_desc->GetOutputsSize(); ++i) {
    gert::StorageShape storage_shape;
    GetStorageShape(op_desc->GetOutputDesc(i), storage_shape);
    std::unique_ptr<uint8_t[]> tensor_holder;
    GE_ASSERT_GRAPH_SUCCESS(BuildRtTensor(op_desc->GetOutputDesc(i), nullptr, tensor_holder));
    GE_ASSERT_NOTNULL(tensor_holder, "Create context holder outputs failed, op[%s]", op_desc->GetName().c_str());
    rt_tensor_holders_.emplace_back(std::move(tensor_holder));
  }
  return ge::GRAPH_SUCCESS;
}

// 0-n input tensors
// n-m output shapes
// m + 1 compile info
// m + 2 tiling func
// 其中 n为输入个数总和，m为输入输出个数总和
KernelContextHolder TilingContextBuilder::Build(const ge::Operator &op) {
  KernelContextHolder holder;
  if (compile_info_ == nullptr) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "Please give tiling context builder compile info.");
    return holder;
  }
  if (platform_info_ == nullptr) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "Please give tiling context builder platform info.");
    return holder;
  }
  auto node = ge::NodeUtilsEx::GetNodeFromOperator(op);
  std::vector<void *> context_inputs;
  auto ret = BuildRTInputTensors(op);
  if (ret != ge::GRAPH_SUCCESS) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "Fail to BuildRTInputTensors.");
    return holder;
  }
  ret = BuildRTOutputShapes(op);
  if (ret != ge::GRAPH_SUCCESS) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "Fail to BuildRTOutputShapes.");
    return holder;
  }
  for (const auto &input_holder : rt_tensor_holders_) {
    context_inputs.emplace_back(input_holder.get());
  }
  context_inputs.emplace_back(compile_info_);
  context_inputs.emplace_back(platform_info_);
  context_inputs.emplace_back(nullptr);

  return base_builder_.Inputs(context_inputs).Outputs(outputs_).Build(node->GetOpDesc());
}

AtomicTilingContextBuilder &AtomicTilingContextBuilder::CompileInfo(void *compile_info) {
  compile_info_ = compile_info;
  return *this;
}

AtomicTilingContextBuilder &AtomicTilingContextBuilder::CleanWorkspaceSizes(ContinuousVector *workspace_sizes) {
  worksapce_sizes_ = reinterpret_cast<void *>(workspace_sizes);
  return *this;
}

AtomicTilingContextBuilder &AtomicTilingContextBuilder::CleanOutputSizes(const std::vector<int64_t> &output_sizes) {
  clean_output_sizes_ = output_sizes;
  return *this;
}

AtomicTilingContextBuilder &AtomicTilingContextBuilder::TilingData(void *tiling_data) {
  outputs_[TilingContext::kOutputTilingData] = tiling_data;
  return *this;
}
AtomicTilingContextBuilder &AtomicTilingContextBuilder::Workspace(ContinuousVector *workspace) {
  outputs_[TilingContext::kOutputWorkspace] = workspace;
  return *this;
}
// 0 atomic op workspace
// 1~n  待清零的output size
// n+1  compile info
// n+2  atomic tiling func
// 其中 n 为待清零的输出个数，
KernelContextHolder AtomicTilingContextBuilder::Build(const ge::Operator &op) {
  KernelContextHolder holder;
  if (compile_info_ == nullptr) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "Please give tiling context builder compile info.");
    return holder;
  }
  std::vector<void *> context_inputs;
  context_inputs.emplace_back(worksapce_sizes_);
  for (const int64_t out_size : clean_output_sizes_) {
    context_inputs.emplace_back(reinterpret_cast<void *>(out_size));
  }
  context_inputs.emplace_back(compile_info_);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  return base_builder_.Inputs(context_inputs).Outputs(outputs_).Build(op_desc);
}
}  // namespace gert

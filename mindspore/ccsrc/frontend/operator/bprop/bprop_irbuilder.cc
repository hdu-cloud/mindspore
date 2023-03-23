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

#include "frontend/operator/bprop/bprop_irbuilder.h"

#include <algorithm>
#include <vector>
#include <limits>
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "frontend/operator/bprop/grad/common_utils.h"

namespace mindspore {
namespace expander {
namespace bprop {
namespace {
constexpr size_t kMaxDims = 8;
}  // namespace

bool BpropIRBuilder::Run(const NodePtrList &inputs, const DAttr &attrs, CNodePtrList *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  if (!BpropIRBuilderFactory::Instance().HasOp(name())) {
    return false;
  }
  inputs_ptr_ = &inputs;
  attrs_ptr_ = &attrs;
  auto func = BpropIRBuilderFactory::Instance().GetBuilder(name());
  auto output_nodes = func(this);
  outputs->reserve(output_nodes.size());
  (void)std::transform(output_nodes.cbegin(), output_nodes.cend(), std::back_inserter(*outputs),
                       [](const NodePtr &node) {
                         auto cnode = node->get<CNodePtr>();
                         MS_EXCEPTION_IF_NULL(cnode);
                         return cnode;
                       });
  return true;
}

ValuePtr BpropIRBuilder::GetAttr(const std::string &attr) const {
  auto iter = attrs_ptr_->find(attr);
  if (iter != attrs_ptr_->end()) {
    return iter->second;
  }
  MS_LOG(WARNING) << "The attr " << attr << " does not exist in op " << name();
  return nullptr;
}

NodePtr BpropIRBuilder::GetInput(size_t i) const {
  if (i >= inputs_ptr_->size()) {
    MS_LOG(EXCEPTION) << "For " << name_ << ", the index " << i << " is out of range of inputs size "
                      << inputs_ptr_->size();
  }
  return (*inputs_ptr_)[i];
}

ValuePtr BpropIRBuilder::GetAttr(const NodePtr &node, const std::string &attr) const {
  auto p = GetCNodePrimitive(node->get());
  MS_EXCEPTION_IF_NULL(p);
  return p->GetAttr(attr);
}

int64_t BpropIRBuilder::GetSize(const NodePtr &node) const {
  auto shape = GetShape(node);
  return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
}

std::string BpropIRBuilder::GetTargetFromContext() const {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  return context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
}

NodePtr BpropIRBuilder::TensorGetItem(const NodePtr &node, int64_t idx) const {
  auto data_shape = GetShape(node);
  auto n = data_shape.size();
  if (n < 1 || n > kMaxDims) {
    MS_EXCEPTION(ValueError) << "Expect Tensor to have dimension between 1 and " << kMaxDims << ", but got: " << n;
  }
  std::vector<int64_t> begin_strides(n, 0);
  std::vector<int64_t> end_strides = data_shape;
  std::vector<int64_t> step_strides(n, 1);
  begin_strides[0] = CheckRange(idx, data_shape[0]);
  end_strides[0] = begin_strides[0] + 1;
  constexpr int64_t begin_mask = 252;  // sum 2^i, i in [2, 8)
  constexpr int64_t end_mask = 252;
  constexpr int64_t ellipsis_mask = 0;
  constexpr int64_t new_axis_mask = 0;
  constexpr int64_t shrink_axis_mask = 1;
  return Emit(
    prim::kStridedSlice,
    {node, EmitValue(MakeValue(begin_strides)), EmitValue(MakeValue(end_strides)), EmitValue(MakeValue(step_strides))},
    {{kAttrBeginMask, MakeValue(begin_mask)},
     {kAttrEndMask, MakeValue(end_mask)},
     {kAttrEllipsisMask, MakeValue(ellipsis_mask)},
     {kAttrNewAxisMask, MakeValue(new_axis_mask)},
     {kAttrShrinkAxisMask, MakeValue(shrink_axis_mask)}});
}

NodePtr BpropIRBuilder::StridedSlice(const NodePtr &x, const std::map<int64_t, std::vector<int64_t>> &slices) const {
  auto data_shape = GetShape(x);
  auto n = data_shape.size();
  std::vector<int64_t> begin_strides(n, 0);
  std::vector<int64_t> end_strides = data_shape;
  std::vector<int64_t> step_strides(n, 1);
  int64_t shrink_axis_mask = 0;
  int64_t end_mask = 0;
  auto zero = MakeValue<int64_t>(0);
  for (const auto &[_axis, slice] : slices) {
    auto axis = CheckRange(_axis, static_cast<int64_t>(n));
    if (slice.size() >= kDim2) {
      begin_strides[axis] = slice[kIndex0];
      end_strides[axis] = slice[kIndex1];
      if (end_strides[axis] == LLONG_MAX) {
        end_mask |= (1 << axis);
      }
      if (slice.size() >= kDim3) {
        step_strides[axis] = slice[kIndex2];
      }
    } else {
      if (slice.size() == 1) {
        begin_strides[axis] = slice[kIndex0];
        end_strides[axis] = begin_strides[axis] + 1;
        shrink_axis_mask |= (1 << axis);
      }
    }
  }
  return Emit(prim::kStridedSlice, {x, Value(begin_strides), Value(end_strides), Value(step_strides)},
              {{kAttrBeginMask, zero},
               {kAttrEndMask, MakeValue(end_mask)},
               {kAttrEllipsisMask, zero},
               {kAttrNewAxisMask, zero},
               {kAttrShrinkAxisMask, MakeValue(shrink_axis_mask)}});
}

#ifdef _MSC_VER
void RegGradArrayOps();
void RegGradClipOps();
void RegGradCommOps();
void RegGradDebugOps();
void RegGradImageOps();
void RegGradImplementationsOps();
void RegGradInnerOps();
void RegGradLinalgOps();
void RegGradMathOps();
void RegGradNnOps();
void RegGradOtherOps();
void RegGradQuantOps();
void RegGradScipyOps();
void RegGradSparseOps();

void RegBpropExpanders() {
  RegGradArrayOps();
  RegGradClipOps();
  RegGradCommOps();
  RegGradDebugOps();
  RegGradImageOps();
  RegGradImplementationsOps();
  RegGradInnerOps();
  RegGradLinalgOps();
  RegGradMathOps();
  RegGradNnOps();
  RegGradOtherOps();
  RegGradQuantOps();
  RegGradScipyOps();
  RegGradSparseOps();
}
#endif
}  // namespace bprop
}  // namespace expander
}  // namespace mindspore

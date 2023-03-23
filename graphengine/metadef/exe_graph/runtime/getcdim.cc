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

#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <bitset>
#include "graph/types.h"
#include "exe_graph/runtime/getcdim.h"
#include "axis_constants.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "exe_graph/runtime/shape.h"
#include <iostream>
namespace gert {
  const std::map<ge::Format, const int32_t> CDIM_INDEX_OF_FORMAT {
    {ge::FORMAT_NCHW, transformer::AXIS_NCHW_DIM_C},
    {ge::FORMAT_HWCN, transformer::AXIS_HWCN_DIM_C},
    {ge::FORMAT_NHWC, transformer::AXIS_NHWC_DIM_C},
    {ge::FORMAT_CHWN, transformer::AXIS_CHWN_DIM_C},
    {ge::FORMAT_NDHWC, transformer::NDHWC_DIM_C},
    {ge::FORMAT_NCDHW, transformer::NCDHW_DIM_C},
    {ge::FORMAT_DHWCN, transformer::DHWCN_DIM_C},
    {ge::FORMAT_DHWNC, transformer::DHWNC_DIM_C}
  };
  int64_t GetCDim(TilingContext *context, const size_t index, const bool is_input) {
    if (context == nullptr) {
      return -1;
    }
    auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
    auto compute_node_info = extend_context->GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return -1;
    }
    auto kernel_context = reinterpret_cast<KernelContext *>(context);
    const CompileTimeTensorDesc *td = nullptr;
    StorageShape *storage_shape = nullptr;
    if (is_input) {
      td = compute_node_info->GetInputTdInfo(index);
      storage_shape = kernel_context->MutableInputPointer<StorageShape>(index);
    } else {
      td = compute_node_info->GetOutputTdInfo(index);
      storage_shape = kernel_context->GetOutputPointer<StorageShape>(index);
    }
    if (td == nullptr || storage_shape == nullptr) {
      return -1;
    }
    auto original_format = td->GetOriginFormat();
    auto iter = CDIM_INDEX_OF_FORMAT.find(original_format);
    if (iter == CDIM_INDEX_OF_FORMAT.end()) {
      return -1;
    }
    Shape &origin_shape = storage_shape->MutableOriginShape();
    auto expend_dims = td->GetExpandDimsType();
    Shape expand_shape;
    (void)expend_dims.Expand(origin_shape, expand_shape);

    if (static_cast<size_t>(iter->second) >= expand_shape.GetDimNum()) {
      return -1;
    }
    if (expand_shape.GetDimNum() == origin_shape.GetDimNum()) {
      return static_cast<int64_t>(origin_shape.GetDim(iter->second));
    } else {
      return static_cast<int64_t>(expand_shape.GetDim(iter->second));
    }
  }

  int64_t GetInputCDim(TilingContext *context, const size_t index) {
    return GetCDim(context, index, true);
  }
  int64_t GetOutputCDim(TilingContext *context, const size_t index) {
    return GetCDim(context, index, false);
  }
}  // namespace gert
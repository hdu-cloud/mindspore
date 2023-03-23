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
#ifndef INC_50EA5B1AAF3341A28036E698708ADB64_H
#define INC_50EA5B1AAF3341A28036E698708ADB64_H
#include <cstdint>
#include <string>
#include <unordered_set>
#include "graph/ge_error_codes.h"
#include "kernel_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_shape_range_context.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "graph/any_value.h"

namespace gert {
struct OpImplKernelRegistry {
  typedef UINT32 (*InferShapeKernelFunc)(InferShapeContext *);
  typedef UINT32 (*InferShapeRangeKernelFunc)(InferShapeRangeContext *);
  typedef UINT32 (*TilingKernelFunc)(TilingContext *);
  typedef UINT32 (*InferDataTypeKernelFunc)(InferDataTypeContext *);
  using OpType = std::string;
  using PrivateAttrList = std::vector<std::pair<std::string, ge::AnyValue>>;
  using PrivateAttrSet = std::unordered_set<std::string>;

  struct OpImplFunctions {
    bool HasDataDependency() const {
      return (inputs_dependency != 0U);
    }
    /*
     * param index: must be ir index
     */
    bool IsInputDataDependency(int32_t index) const {
      if (index < 0 || static_cast<size_t>(index) >= sizeof(inputs_dependency) * kInt64ByteCount) {
        return false;
      }
      return inputs_dependency & static_cast<uint64_t>(1) << index;
    }
    ge::graphStatus SetInputDataDependency(int32_t index) {
      if (index < 0 || static_cast<size_t>(index) >= sizeof(inputs_dependency) * kInt64ByteCount) {
        return ge::GRAPH_FAILED;
      }
      inputs_dependency |= static_cast<uint64_t>(1) << index;
      return ge::GRAPH_SUCCESS;
    }

    InferShapeKernelFunc infer_shape;
    InferShapeRangeKernelFunc infer_shape_range;
    InferDataTypeKernelFunc infer_datatype;
    TilingKernelFunc tiling;
    KernelRegistry::KernelFunc tiling_parse;
    void *(*compile_info_creator)();
    void (*compile_info_deleter)(void *);
    size_t max_tiling_data_size;
    uint64_t inputs_dependency;
    static constexpr size_t kInt64ByteCount = 8;
    PrivateAttrList private_attrs;
    PrivateAttrSet unique_private_attrs;
  };
  virtual ~OpImplKernelRegistry() {}
  virtual const OpImplFunctions *GetOpImpl(const std::string &op_type) const = 0;
  virtual const PrivateAttrList &GetPrivateAttrs(const std::string &op_type) const = 0;
};
}  // namespace gert

#endif

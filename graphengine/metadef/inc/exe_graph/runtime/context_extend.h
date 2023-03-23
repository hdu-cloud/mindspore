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

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_CONTEXT_EXTEND_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_CONTEXT_EXTEND_H_
#include <type_traits>
#include <memory>
#include "compute_node_info.h"

namespace gert {
class KernelExtendInfo {
 public:
  /**
   * 获取kernel name
   * @return kernel name
   */
  const char *GetKernelName() const {
    return kernel_name_;
  }
  /**
   * 设置kernel name
   * @param kernel_name kernel name
   */
  void SetKernelName(const char *kernel_name) {
    kernel_name_ = kernel_name;
  }
  /**
   * 获取kernel type
   * @return kernel type
   */
  const char *GetKernelType() const {
    return kernel_type_;
  }
  /**
   * 设置kernel type
   * @param kernel_type kernel type
   */
  void SetKernelType(const char *kernel_type) {
    kernel_type_ = kernel_type;
  }

  /**
   * 设置kernel_type_idx_,用于profiling
   * @param kernel_type_idx idx of kernel type in profiling
   */
  void SetKernelTypeIdx(uint64_t kernel_type_idx) {
    kernel_type_idx_ = kernel_type_idx;
  }

  /**
   * 设置compute_node_name_idx_,用于profiling
   * @param compute_node_name_idx idx of node name in profiling
   */
  void SetNodeNameIdx(uint64_t compute_node_name_idx) {
    compute_node_name_idx_ = compute_node_name_idx;
  }

  /**
   * 获取compute_node_name_idx_,用于profiling
   * @return compute_node_name_idx_ idx of node name in profiling
   */
  uint64_t GetNodeNameIdx() const {
    return compute_node_name_idx_;
  }

  /**
   * 获取kernel_type_idx_,用于profiling
   * @param kernel_type_idx_ idx of kernel type in profiling
   */
  uint64_t GetKernelTypeIdx() const {
    return kernel_type_idx_;
  }

  KernelExtendInfo() = delete;
  KernelExtendInfo(const KernelExtendInfo &) = delete;
  KernelExtendInfo(KernelExtendInfo &&) = delete;
  KernelExtendInfo &operator=(const KernelExtendInfo &) = delete;
  KernelExtendInfo &operator=(KernelExtendInfo &&) = delete;

 private:
  const char *kernel_name_;
  const char *kernel_type_;
  uint64_t compute_node_name_idx_;
  uint64_t kernel_type_idx_;
};
static_assert(std::is_standard_layout<KernelExtendInfo>::value, "The class KernelExtendInfo must be a POD");

}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_CONTEXT_EXTEND_H_

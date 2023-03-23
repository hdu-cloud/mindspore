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

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_KERNEL_REGISTRY_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_KERNEL_REGISTRY_H_
#include <unordered_map>
#include <string>

#include "kernel_registry.h"

namespace gert {
class KernelRegistryImpl : public KernelRegistry {
 public:
  static KernelRegistryImpl &GetInstance();
  void RegisterKernel(std::string kernel_type, KernelFuncs func) override;
  const KernelFuncs *FindKernelFuncs(const std::string &kernel_type) const override;

  const std::unordered_map<std::string, KernelFuncs> &GetAll() const;

 private:
  std::unordered_map<std::string, KernelFuncs> types_to_func_;
};
}

#endif //AIR_CXX_RUNTIME_V2_KERNEL_KERNEL_REGISTRY_H_

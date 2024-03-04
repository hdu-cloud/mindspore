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

#ifndef METADEF_CXX_REGISTER_KERNEL_REGISTER_DATA_H_
#define METADEF_CXX_REGISTER_KERNEL_REGISTER_DATA_H_
#include <string>
#include "register/kernel_registry.h"
namespace gert {
class KernelRegisterData {
 public:
  explicit KernelRegisterData(const ge::char_t *kernel_type);

  KernelRegistry::KernelFuncs &GetFuncs() {
    return funcs_;
  }

  const std::string &GetKernelType() const {
    return kernel_type_;
  }

  std::string &GetCriticalSection() {
    return critical_section_;
  }

 private:
  std::string critical_section_;
  std::string kernel_type_;
  KernelRegistry::KernelFuncs funcs_;
};
}  // namespace gert

#endif  // METADEF_CXX_REGISTER_KERNEL_REGISTER_DATA_H_

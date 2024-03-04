/*
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

#include "register/ffts_node_calculater_registry.h"
#include "common/hyper_status.h"

namespace gert {
FFTSNodeCalculaterRegistry &FFTSNodeCalculaterRegistry::GetInstance() {
  static FFTSNodeCalculaterRegistry registry;
  return registry;
}

FFTSNodeCalculaterRegistry::NodeCalculater FFTSNodeCalculaterRegistry::FindNodeCalculater(const string &func_name) {
  auto iter = names_to_calculater_.find(func_name);
  if (iter == names_to_calculater_.end()) {
    return nullptr;
  }
  return iter->second;
}

void FFTSNodeCalculaterRegistry::Register(const string &func_name,
                                          const FFTSNodeCalculaterRegistry::NodeCalculater func) {
  names_to_calculater_[func_name] = func;
}

FFTSNodeCalculaterRegister::FFTSNodeCalculaterRegister(const string &func_name,
    FFTSNodeCalculaterRegistry::NodeCalculater func) noexcept {
  FFTSNodeCalculaterRegistry::GetInstance().Register(func_name, func);
}
}  // namespace gert
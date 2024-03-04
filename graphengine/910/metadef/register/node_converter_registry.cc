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

#include "register/node_converter_registry.h"
#include "common/hyper_status.h"

namespace gert {
NodeConverterRegistry &NodeConverterRegistry::GetInstance() {
  static NodeConverterRegistry registry;
  return registry;
}

NodeConverterRegistry::NodeConverter NodeConverterRegistry::FindNodeConverter(const string &func_name) {
  auto data = FindRegisterData(func_name);
  if (data == nullptr) {
    return nullptr;
  }
  return data->converter;
}
void NodeConverterRegistry::RegisterNodeConverter(const std::string &func_name, NodeConverter func) {
  names_to_register_data_[func_name] = {func, -1};
}
const NodeConverterRegistry::ConverterRegisterData *NodeConverterRegistry::FindRegisterData(
    const string &func_name) const {
  auto iter = names_to_register_data_.find(func_name);
  if (iter == names_to_register_data_.end()) {
    return nullptr;
  }
  return &iter->second;
}
void NodeConverterRegistry::Register(const string &func_name,
                                     const NodeConverterRegistry::ConverterRegisterData &data) {
  names_to_register_data_[func_name] = data;
}
NodeConverterRegister::NodeConverterRegister(const char *lower_func_name,
                                             NodeConverterRegistry::NodeConverter func) noexcept {
  NodeConverterRegistry::GetInstance().Register(lower_func_name, {func, -1});
}
NodeConverterRegister::NodeConverterRegister(const char *lower_func_name, int32_t require_placement,
                                             NodeConverterRegistry::NodeConverter func) noexcept {
  NodeConverterRegistry::GetInstance().Register(lower_func_name, {func, require_placement});
}
}  // namespace gert
/*
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

#include "register/kernel_registry_impl.h"
#include <utility>
#include "graph/debug/ge_log.h"
namespace gert {
namespace {
ge::graphStatus NullCreator(const ge::Node *node, KernelContext *context) {
  (void) node;
  (void) context;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus NullDestoryer(const ge::Node *node, KernelContext *context) {
  (void) node;
  (void) context;
  return ge::GRAPH_SUCCESS;
}
std::shared_ptr<KernelRegistry> g_user_defined_registry = nullptr;
}  // namespace

KernelRegistry &KernelRegistry::GetInstance() {
  if (g_user_defined_registry != nullptr) {
    return *g_user_defined_registry;
  } else {
    return KernelRegistryImpl::GetInstance();
  }
}
void KernelRegistry::ReplaceKernelRegistry(std::shared_ptr<KernelRegistry> registry) {
  g_user_defined_registry = std::move(registry);
}

KernelRegistryImpl &KernelRegistryImpl::GetInstance() {
  static KernelRegistryImpl registry;
  return registry;
}
void KernelRegistryImpl::RegisterKernel(std::string kernel_type, KernelRegistry::KernelFuncs func) {
  types_to_func_[std::move(kernel_type)] = std::move(func);
}

const KernelRegistry::KernelFuncs *KernelRegistryImpl::FindKernelFuncs(const std::string &kernel_type) const {
  auto iter = types_to_func_.find(kernel_type);
  if (iter == types_to_func_.end()) {
    return nullptr;
  }
  return &iter->second;
}
const std::unordered_map<std::string, KernelRegistry::KernelFuncs> &KernelRegistryImpl::GetAll() const {
  return types_to_func_;
}
KernelRegister::KernelRegister(const char *kernel_type) : kernel_type_(kernel_type) {
  kernel_funcs_.outputs_creator = NullCreator;
  kernel_funcs_.outputs_initializer = NullDestoryer;
}
KernelRegister &KernelRegister::RunFunc(KernelRegistry::KernelFunc func) {
  kernel_funcs_.run_func = func;
  return *this;
}
KernelRegister &KernelRegister::OutputsCreator(KernelRegistry::CreateOutputsFunc func) {
  kernel_funcs_.outputs_creator = std::move(func);
  return *this;
}
KernelRegister &KernelRegister::OutputsInitializer(KernelRegistry::CreateOutputsFunc func) {
  kernel_funcs_.outputs_initializer = std::move(func);
  return *this;
}
KernelRegister::KernelRegister(const KernelRegister &other) {
  if (other.kernel_type_.size() > 1 && other.kernel_type_[0] == '"') {
    GELOGW("The kernel type starts with \", that maybe a mistake");
  }
  KernelRegistry::GetInstance().RegisterKernel(other.kernel_type_, other.kernel_funcs_);
}
}  // namespace gert
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

#include "register/op_kernel_registry.h"
#include <mutex>
#include <map>
#include "graph/debug/ge_log.h"
#include "graph/debug/ge_util.h"

namespace ge {
class OpKernelRegistry::OpKernelRegistryImpl {
 public:
  void RegisterHostCpuOp(const std::string &op_type, const OpKernelRegistry::CreateFn create_fn) {
    const std::lock_guard<std::mutex> lock(mu_);
    create_fns_[op_type] = create_fn;
  }

  OpKernelRegistry::CreateFn GetCreateFn(const std::string &op_type) {
    const std::lock_guard<std::mutex> lock(mu_);
    const auto it = create_fns_.find(op_type);
    if (it == create_fns_.end()) {
      return nullptr;
    }

    return it->second;
  }

 private:
  std::mutex mu_;
  std::map<std::string, OpKernelRegistry::CreateFn> create_fns_;
};

OpKernelRegistry::OpKernelRegistry() {
  impl_ = ge::ComGraphMakeUnique<OpKernelRegistryImpl>();
}

OpKernelRegistry::~OpKernelRegistry() = default;

OpKernelRegistry& OpKernelRegistry::GetInstance() {
  static OpKernelRegistry instance;
  return instance;
}

bool OpKernelRegistry::IsRegistered(const std::string &op_type) {
  if (impl_ == nullptr) {
    GELOGE(MEMALLOC_FAILED,
           "[Check][Param:impl_]Failed to invoke IsRegistered %s, OpKernelRegistry is not properly initialized",
           op_type.c_str());
    return false;
  }

  return impl_->GetCreateFn(op_type) != nullptr;
}

void OpKernelRegistry::RegisterHostCpuOp(const std::string &op_type, const CreateFn create_fn) {
  if (impl_ == nullptr) {
    GELOGE(MEMALLOC_FAILED,
           "[Check][Param:impl_]Failed to register %s, OpKernelRegistry is not properly initialized",
           op_type.c_str());
    return;
  }

  impl_->RegisterHostCpuOp(op_type, create_fn);
}
std::unique_ptr<HostCpuOp> OpKernelRegistry::CreateHostCpuOp(const std::string &op_type) {
  if (impl_ == nullptr) {
    GELOGE(MEMALLOC_FAILED,
           "[Check][Param:impl_]Failed to create op for %s, OpKernelRegistry is not properly initialized",
           op_type.c_str());
    return nullptr;
  }

  const auto create_fn = impl_->GetCreateFn(op_type);
  if (create_fn == nullptr) {
    GELOGD("Host Cpu op is not registered. op type = %s", op_type.c_str());
    return nullptr;
  }

  return std::unique_ptr<HostCpuOp>(create_fn());
}

HostCpuOpRegistrar::HostCpuOpRegistrar(const char_t *const op_type, HostCpuOp *(*const create_fn)()) {
  if (op_type == nullptr) {
    GELOGE(PARAM_INVALID, "[Check][Param:op_type]is null,Failed to register host cpu op");
    return;
  }

  OpKernelRegistry::GetInstance().RegisterHostCpuOp(op_type, create_fn);
}
} // namespace ge
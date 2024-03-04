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

#include "register/kernel_registry_impl.h"
#include <utility>
#include "graph/debug/ge_log.h"
#include "kernel_register_data.h"
namespace gert {
namespace {
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
void KernelRegistryImpl::RegisterKernel(std::string kernel_type, KernelInfo kernel_infos) {
  kernel_infos_[std::move(kernel_type)] = std::move(kernel_infos);
}

const KernelRegistry::KernelFuncs *KernelRegistryImpl::FindKernelFuncs(const std::string &kernel_type) const {
  const auto iter = kernel_infos_.find(kernel_type);
  if (iter == kernel_infos_.end()) {
    return nullptr;
  }
  return &iter->second.func;
}
const KernelRegistry::KernelInfo *KernelRegistryImpl::FindKernelInfo(const std::string &kernel_type) const {
  const auto iter = kernel_infos_.find(kernel_type);
  if (iter == kernel_infos_.end()) {
    return nullptr;
  }
  return &iter->second;
}
const std::unordered_map<std::string, KernelRegistryImpl::KernelInfo> &KernelRegistryImpl::GetAll() const {
  return kernel_infos_;
}

KernelRegisterV2::KernelRegisterV2(const char *kernel_type)
    : register_data_(new(std::nothrow) KernelRegisterData(kernel_type)) {}
KernelRegisterV2::~KernelRegisterV2() = default;
KernelRegisterV2 &KernelRegisterV2::RunFunc(KernelRegistry::KernelFunc func) {
  if (register_data_ != nullptr) {
    register_data_->GetFuncs().run_func = func;
  }
  return *this;
}
KernelRegisterV2 &KernelRegisterV2::ConcurrentCriticalSectionKey(const std::string &critical_section_key) {
  if (register_data_ != nullptr) {
    register_data_->GetCriticalSection() = critical_section_key;
  }
  return *this;
}
KernelRegisterV2 &KernelRegisterV2::OutputsCreator(KernelRegistry::CreateOutputsFunc func) {
  GELOGW("The OutputsCreator is no longer supported, use OutputsCreatorFunc instead");
  (void)func;
  return *this;
}
KernelRegisterV2 &KernelRegisterV2::OutputsCreatorFunc(KernelRegistry::OutputsCreatorFunc func) {
  if (register_data_ != nullptr) {
    register_data_->GetFuncs().outputs_creator_func = func;
  }
  return *this;
}
KernelRegisterV2 &KernelRegisterV2::OutputsInitializer(KernelRegistry::CreateOutputsFunc func) {
  GELOGW("The OutputsInitializer is no longer supported");
  (void)func;
  return *this;
}
KernelRegisterV2 &KernelRegisterV2::TracePrinter(KernelRegistry::TracePrinter func) {
  if (register_data_ != nullptr) {
    register_data_->GetFuncs().trace_printer = func;
  }
  return *this;
}

KernelRegisterV2 &KernelRegisterV2::ProfilingInfoFiller(KernelRegistry::ProfilingInfoFiller func) {
  if (register_data_ != nullptr) {
    register_data_->GetFuncs().profiling_info_filler = func;
  }
  return *this;
}

KernelRegisterV2 &KernelRegisterV2::DataDumpInfoFiller(KernelRegistry::DataDumpInfoFiller func) {
  if (register_data_ != nullptr) {
    register_data_->GetFuncs().data_dump_info_filler = func;
  }
  return *this;
}

KernelRegisterV2 &KernelRegisterV2::ExceptionDumpInfoFiller(KernelRegistry::ExceptionDumpInfoFiller func) {
  if (register_data_ != nullptr) {
    register_data_->GetFuncs().exception_dump_info_filler = func;
  }
  return *this;
}

KernelRegisterV2::KernelRegisterV2(const KernelRegisterV2 &other) : register_data_(nullptr) {
  const auto register_data = other.register_data_.get();
  if (register_data == nullptr) {
    GE_LOGE("The register_data_ in register object is nullptr, failed to register funcs");
    return;
  }
  GELOGD("GERT kernel type %s registered", register_data->GetKernelType().c_str());
  KernelRegistry::GetInstance().RegisterKernel(register_data->GetKernelType(),
                                               {register_data->GetFuncs(), register_data->GetCriticalSection()});
}
}  // namespace gert
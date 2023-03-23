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
#ifndef B369E37D560547C2B8DC137404F9713E_H
#define B369E37D560547C2B8DC137404F9713E_H
#include <functional>
#include <string>
#include <memory>
#include "graph/ge_error_codes.h"
#include "exe_graph/runtime/base_type.h"
#include "exe_graph/runtime/kernel_context.h"
#include "graph/node.h"

namespace gert {
class KernelRegistry {
 public:
  static KernelRegistry &GetInstance();
  static void ReplaceKernelRegistry(std::shared_ptr<KernelRegistry> registry);

  using CreateOutputsFunc = std::function<ge::graphStatus(const ge::Node *, KernelContext *)>;
  typedef UINT32 (*KernelFunc)(KernelContext *context);
  struct KernelFuncs {
    KernelFunc run_func;
    CreateOutputsFunc outputs_creator;
    CreateOutputsFunc outputs_initializer;
  };

  virtual ~KernelRegistry() = default;
  virtual const KernelFuncs *FindKernelFuncs(const std::string &kernel_type) const = 0;
  virtual void RegisterKernel(std::string kernel_type, KernelFuncs func) {
    (void) kernel_type;
    (void) func;
  };
};

class KernelRegister {
 public:
  explicit KernelRegister(const char *kernel_type);
  KernelRegister(const KernelRegister &other);

  KernelRegister &RunFunc(KernelRegistry::KernelFunc func);

  KernelRegister &OutputsCreator(KernelRegistry::CreateOutputsFunc func);
  KernelRegister &OutputsInitializer(KernelRegistry::CreateOutputsFunc func);

 private:
  std::string kernel_type_;
  KernelRegistry::KernelFuncs kernel_funcs_;
};
}  // namespace gert

#define REGISTER_KERNEL_COUNTER2(type, counter) static auto g_register_kernel_##counter = gert::KernelRegister(#type)
#define REGISTER_KERNEL_COUNTER(type, counter) REGISTER_KERNEL_COUNTER2(type, counter)
#define REGISTER_KERNEL(type) REGISTER_KERNEL_COUNTER(type, __COUNTER__)

#endif

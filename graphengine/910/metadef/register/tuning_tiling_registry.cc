/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include "register/tuning_tiling_registry.h"
#include "common/ge_common/debug/ge_log.h"

namespace tuningtiling {
ge::AscendString TuningTilingDef::GetClassName() const {
  return class_name_;
}

std::map<ge::AscendString, TuningTilingDefConstructor> &TuningTilingClassFactory::RegisterInfo() {
  static std::map<ge::AscendString, TuningTilingDefConstructor> instance;
  return instance;
}

void TuningTilingClassFactory::RegisterTilingData(const ge::AscendString &optype,
                                                  TuningTilingDefConstructor const constructor) {
  if (constructor == nullptr) {
    return;
  }
  auto &instance = TuningTilingClassFactory::RegisterInfo();
  instance[optype] = constructor;
  GELOGI("RegisterTuningTilingData: optype: %s, registered count: %zu", optype.GetString(), instance.size());
}

std::shared_ptr<TuningTilingDef> TuningTilingClassFactory::CreateTilingDataInstance(const ge::AscendString &optype) {
  const auto &instance = TuningTilingClassFactory::RegisterInfo();
  const auto it = instance.find(optype);
  if (it == instance.cend()) {
    GELOGW("CreateTilingDataInstance: can not find optype: %s", optype.GetString());
    return nullptr;
  }

  TuningTilingDefConstructor const constructor = it->second;

  if (constructor == nullptr) {
    GELOGW("CreateTilingDataInstance: constructor is nullptr");
    return nullptr;
  }

  return (*constructor)();
}
}  // namespace tuningtiling

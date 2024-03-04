/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#include "register/ffts_plus_update_manager.h"
#include "graph/debug/ge_util.h"
#include "common/plugin/plugin_manager.h"

namespace ge {
namespace {
void GetLibPaths(std::string &lib_paths) {
  const std::string base_path = GetModelPath();
  GELOGI("base path is %s", base_path.c_str());
  std::vector<std::string> ffts_depend_libs;
  ffts_depend_libs.push_back("libffts_executor.so");
  ffts_depend_libs.push_back("libascendcpu_executor.so");
  for (const auto &lib_name : ffts_depend_libs) {
    (void)lib_paths.append(base_path);
    (void)lib_paths.append(lib_name);
    (void)lib_paths.append(":");
  }
  GELOGI("Get lib paths for load. paths = %s", lib_paths.c_str());
}
}  // namespace

FftsPlusUpdateManager &FftsPlusUpdateManager::Instance() {
  static FftsPlusUpdateManager instance;
  return instance;
}

FftsCtxUpdatePtr FftsPlusUpdateManager::GetUpdater(const std::string &core_type) const {
  const std::map<std::string, FftsCtxUpdateCreatorFun>::const_iterator it = creators_.find(core_type);
  if (it == creators_.cend()) {
    GELOGW("Cannot find creator for core type: %s.", core_type.c_str());
    return nullptr;
  }

  return it->second();
}

void FftsPlusUpdateManager::RegisterCreator(const std::string &core_type, const FftsCtxUpdateCreatorFun &creator) {
  if (creator == nullptr) {
    GELOGW("Register null creator for core type: %s", core_type.c_str());
    return;
  }

  const auto it = creators_.find(core_type);
  if (it != creators_.end()) {
    GELOGW("Creator already exist for core type: %s", core_type.c_str());
    return;
  }

  GELOGI("Register creator for core type: %s", core_type.c_str());
  creators_[core_type] = creator;
}

Status FftsPlusUpdateManager::Initialize() {
  const std::unique_lock<std::mutex> lk(init_mutex_);
  if (is_init_) {
    return SUCCESS;
  }
  std::string lib_paths;
  GetLibPaths(lib_paths);
  plugin_manager_ = ComGraphMakeUnique<PluginManager>();
  GE_CHECK_NOTNULL(plugin_manager_);
  GE_CHK_STATUS_RET(plugin_manager_->LoadSo(lib_paths), "[Load][Libs]Failed, lib_paths=%s.", lib_paths.c_str());
  is_init_ = true;
  return SUCCESS;
}

FftsPlusUpdateManager::~FftsPlusUpdateManager() {
  creators_.clear();  // clear must be called before `plugin_manager_.reset` which would close so
  plugin_manager_.reset();
}
}  // namespace ge

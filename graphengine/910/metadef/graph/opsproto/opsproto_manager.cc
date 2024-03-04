/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "graph/opsproto_manager.h"
#include <cstdlib>
#include <functional>
#include "common/ge_common/debug/ge_log.h"
#include "graph/debug/ge_log.h"
#include "graph/types.h"
#include "graph/def_types.h"
#include "graph/operator_factory_impl.h"
#include "mmpa/mmpa_api.h"
#include "common/plugin/plugin_manager.h"

namespace ge {
OpsProtoManager *OpsProtoManager::Instance() {
  static OpsProtoManager instance;
  return &instance;
}

bool OpsProtoManager::Initialize(const std::map<std::string, std::string> &options) {
  const std::lock_guard<std::mutex> lock(mutex_);

  if (is_init_) {
    GELOGI("OpsProtoManager is already initialized.");
    return true;
  }

  const std::map<std::string, std::string>::const_iterator iter = options.find("ge.opsProtoLibPath");
  if (iter == options.end()) {
    GELOGW("[Initialize][CheckOption] Option \"ge.opsProtoLibPath\" not set");
    return false;
  }

  pluginPath_ = iter->second;
  LoadOpsProtoPluginSo(pluginPath_);

  is_init_ = true;

  return true;
}

void OpsProtoManager::Finalize() {
  const std::lock_guard<std::mutex> lock(mutex_);

  if (!is_init_) {
    GELOGI("OpsProtoManager is not initialized.");
    return;
  }

  for (const auto handle : handles_) {
    if (handle != nullptr) {
      if (mmDlclose(handle) != 0) {
        const char_t *error = mmDlerror();
        error = (error == nullptr) ? "" : error;
        GELOGW("[Finalize][CloseHandle] close handle failed, reason:%s", error);
        continue;
      }
      GELOGI("close opsprotomanager handler success");
    } else {
      GELOGW("[Finalize][CheckHandle] handler is null");
    }
  }

  is_init_ = false;
}

static std::vector<std::string> SplitStr(const std::string &str, const char_t delim) {
  std::vector<std::string> elems;
  if (str.empty()) {
    elems.emplace_back("");
    return elems;
  }

  std::stringstream str_stream(str);
  std::string item;

  while (getline(str_stream, item, delim)) {
    elems.push_back(item);
  }

  const auto str_size = str.size();
  if ((str_size > 0UL) && (str[str_size - 1UL] == delim)) {
    elems.emplace_back("");
  }

  return elems;
}

void GetOpsProtoSoFileList(const std::string &path, std::vector<std::string> &file_list) {
  // Support multi lib directory with ":" as delimiter
  const std::vector<std::string> v_path = SplitStr(path, ':');

  std::string os_type;
  std::string cpu_type;
  PluginManager::GetCurEnvPackageOsAndCpuType(os_type, cpu_type);

  for (auto i = 0UL; i < v_path.size(); ++i) {
    const std::string new_path = v_path[i] + "lib/" + os_type + "/" + cpu_type + "/";
    char_t resolved_path[MMPA_MAX_PATH] = {};
    const INT32 result = mmRealPath(new_path.c_str(), &(resolved_path[0U]), MMPA_MAX_PATH);
    if (result == EN_OK) {
      PluginManager::GetFileListWithSuffix(new_path, ".so", file_list);
    } else {
      GELOGW("[FindSo][Check] Get path with os&cpu type [%s] failed, reason:%s", new_path.c_str(), strerror(errno));
      PluginManager::GetFileListWithSuffix(v_path[i], ".so", file_list);
    }
  }
}

void OpsProtoManager::LoadOpsProtoPluginSo(const std::string &path) {
  if (path.empty()) {
    REPORT_INNER_ERROR("E18888", "filePath is empty. please check your text file.");
    GELOGE(GRAPH_FAILED, "[Check][Param] filePath is empty. please check your text file.");
    return;
  }
  std::vector<std::string> file_list;

  // If there is .so file in the lib path
  GetOpsProtoSoFileList(path, file_list);

  // Not found any .so file in the lib path
  if (file_list.empty()) {
    GELOGW("[LoadSo][Check] OpsProtoManager can not find any plugin file in pluginPath: %s \n", path.c_str());
    return;
  }
  // Warning message
  GELOGW("[LoadSo][Check] Shared library will not be checked. Please make sure that the source of shared library is "
         "trusted.");

  // Load .so file
  for (const auto &elem : file_list) {
    OperatorFactoryImpl::SetRegisterOverridable(true);
    void *const handle = mmDlopen(elem.c_str(), static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) |
        static_cast<uint32_t>(MMPA_RTLD_GLOBAL)));
    OperatorFactoryImpl::SetRegisterOverridable(false);
    if (handle == nullptr) {
      const char_t *error = mmDlerror();
      error = (error == nullptr) ? "" : error;
      GELOGW("[LoadSo][Open] OpsProtoManager dlopen failed, plugin name:%s. Message(%s).", elem.c_str(), error);
      continue;
    } else {
      // Close dl when the program exist, not close here
      GELOGI("OpsProtoManager plugin load %s success.", elem.c_str());
      handles_.push_back(handle);
    }
  }
}
}  // namespace ge

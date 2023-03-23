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

#ifndef METADEF_CXX_GRAPH_COMMON_PLUGIN_SO_MANAGER_H_
#define METADEF_CXX_GRAPH_COMMON_PLUGIN_SO_MANAGER_H_

#include <iostream>
#include <map>
#include <vector>

#include "mmpa/mmpa_api.h"
#include "external/ge/ge_api_error_codes.h"
#include "external/graph/types.h"
#include "graph/debug/ge_log.h"

namespace ge {
class PluginSoManager {
 public:
  PluginSoManager() = default;

  ~PluginSoManager();

  void SplitPath(const std::string &mutil_path, std::vector<std::string> &path_vec) const;

  Status LoadSo(const std::string &path, const std::vector<std::string> &func_check_list = {});

  Status Load(const std::string &path, const std::vector<std::string> &func_check_list = {});

 private:
  void ClearHandles() noexcept;
  Status ValidateSo(const std::string &file_path, const int64_t size_of_loaded_so, int64_t &file_size) const;
  std::vector<std::string> so_list_;
  std::map<std::string, void *> handles_;
};

inline std::string GetModelPath() {
  mmDlInfo dl_info;
  if ((mmDladdr(reinterpret_cast<void *>(&GetModelPath), &dl_info) != EN_OK) || (dl_info.dli_fname == nullptr)) {
    GELOGW("Failed to read the shared library file path! errmsg:%s", mmDlerror());
    return std::string();
  }

  if (strnlen(dl_info.dli_fname, MMPA_MAX_PATH) >= MMPA_MAX_PATH) {
    GELOGW("The shared library file path is too long!");
    return std::string();
  }

  char_t path[MMPA_MAX_PATH]{};
  if (mmRealPath(dl_info.dli_fname, &path[0], MMPA_MAX_PATH) != EN_OK) {
    const size_t max_error_strlen = 128U;
    char_t err_buf[max_error_strlen + 1U]{};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], max_error_strlen);
    GELOGW("Failed to get realpath of %s, errmsg:%s", dl_info.dli_fname, err_msg);
    return std::string();
  }

  std::string so_path = path;
  so_path = so_path.substr(0U, so_path.rfind('/') + 1U);
  return so_path;
}
}  // namespace ge
#endif  //METADEF_CXX_GRAPH_COMMON_PLUGIN_SO_MANAGER_H_

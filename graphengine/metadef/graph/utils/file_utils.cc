/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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

#include "graph/utils/file_utils.h"

#include <cerrno>
#include "graph/types.h"
#include "graph/debug/ge_log.h"
#include "mmpa/mmpa_api.h"

namespace ge {
std::string RealPath(const char_t *path) {
  if (path == nullptr) {
    REPORT_INNER_ERROR("E18888", "path is nullptr, check invalid");
    GELOGE(FAILED, "[Check][Param] path pointer is NULL.");
    return "";
  }
  if (strnlen(path, static_cast<size_t>(MMPA_MAX_PATH)) >= static_cast<size_t>(MMPA_MAX_PATH)) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19002", {"filepath", "size"},
                                                    {path, std::to_string(MMPA_MAX_PATH)});
    GELOGE(FAILED, "[Check][Param]Path[%s] len is too long, it must be less than %d", path, MMPA_MAX_PATH);
    return "";
  }

  // Nullptr is returned when the path does not exist or there is no permission
  // Return absolute path when path is accessible
  std::string res;
  char_t resolved_path[MMPA_MAX_PATH] = {};
  if (mmRealPath(path, &(resolved_path[0U]), MMPA_MAX_PATH) == EN_OK) {
    res = &(resolved_path[0]);
  } else {
    GELOGW("[Util][realpath] Get real_path for %s failed, reason:%s", path, strerror(errno));
  }

  return res;
}

inline int32_t CheckAndMkdir(const char_t *tmp_dir_path, mmMode_t mode) {
  if (mmAccess2(tmp_dir_path, M_F_OK) != EN_OK) {
    const int32_t ret = mmMkdir(tmp_dir_path, mode);  // 700
    if (ret != 0) {
      REPORT_CALL_ERROR("E18888",
                        "Can not create directory %s. Make sure the directory "
                        "exists and writable. errmsg:%s",
                        tmp_dir_path, strerror(errno));
      GELOGW(
          "[Util][mkdir] Create directory %s failed, reason:%s. Make sure the "
          "directory exists and writable.",
          tmp_dir_path, strerror(errno));
      return ret;
    }
  }
  return 0;
}

/**
 *  @ingroup domi_common
 *  @brief Create directory, support to create multi-level directory
 *  @param [in] directory_path  Path, can be multi-level directory
 *  @return -1 fail
 *  @return 0 success
 */
int32_t CreateDirectory(const std::string &directory_path) {
  GE_CHK_BOOL_EXEC(!directory_path.empty(),
                   REPORT_INNER_ERROR("E18888", "directory path is empty, check invalid");
                       return -1, "[Check][Param] directory path is empty.");
  const auto dir_path_len = directory_path.length();
  if (dir_path_len >= static_cast<size_t>(MMPA_MAX_PATH)) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19002", {"filepath", "size"},
                                                    {directory_path, std::to_string(MMPA_MAX_PATH)});
    GELOGW("[Util][mkdir] Path %s len is too long, it must be less than %d", directory_path.c_str(), MMPA_MAX_PATH);
    return -1;
  }
  char_t tmp_dir_path[MMPA_MAX_PATH] = {};
  const auto mkdir_mode = static_cast<mmMode_t>(static_cast<uint32_t>(M_IRUSR) |
                                          static_cast<uint32_t>(M_IWUSR) |
                                          static_cast<uint32_t>(M_IXUSR));
  for (size_t i = 0U; i < dir_path_len; i++) {
    tmp_dir_path[i] = directory_path[i];
    if ((tmp_dir_path[i] == '\\') || (tmp_dir_path[i] == '/')) {
      const int32_t ret = CheckAndMkdir(&(tmp_dir_path[0U]), mkdir_mode);
      if (ret != 0) {
        return ret;
      }
    }
  }
  return CheckAndMkdir(directory_path.c_str(), mkdir_mode);
}
}

/**
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

#include "common/plugin/plugin_manager.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <regex>
#include <string>
#include <sys/stat.h>

#include "common/checker.h"
#include "common/ge_common/debug/log.h"
#include "common/ge_common/util.h"
#include "graph/def_types.h"

namespace ge {
namespace {
const int32_t kMaxNumOfSo = 64;
const int32_t kMaxSizeOfSo = 209100800;        // = 200M(unit is Byte)
const int32_t kMaxSizeOfLoadedSo = 522752000;  // = 500M(unit is Byte)
const char_t *const kExt = ".so";              // supported extension of shared object
const char_t *const kOppEnvName = "ASCEND_OPP_PATH";
const char_t *const kBuiltIn = "built-in";     // opp built-in directory name
const char_t *const kVendors = "vendors";      // opp vendors directory name
const char_t *const kConfig = "config.ini";    // opp vendors config file name
const size_t kVendorConfigPartsCount = 2U;
const char_t *kHostCpuLibRelativePathV01 = "/op_impl/built-in/host_cpu";
const char_t *kHostCpuLibRelativePathV02 = "/built-in/op_impl/host_cpu";
const size_t kMaxErrorStrLen = 128U;
const uint32_t kLibFirstLayer = 0U;
const uint32_t kLibSecondLayer = 1U;
const char_t *const kOppPath = "opp/";
const char_t *const kRuntimePath = "runtime/";
const char_t *const kCompilerPath = "compiler/";
const char_t *const kScene = "scene.info";
const size_t kSceneValueCount = 2U;
const size_t kSceneKeyIndex = 0U;
const size_t kSceneValueIndex = 1U;
const char_t *const kSceneOs = "os";
const char_t *const kSceneArch = "arch";
const char_t *const kRequiredOppAbiVersion = "required_opp_abi_version=";
const char_t *const kCompilerVersion = "compiler_version=";
const char_t *const kVersionInfo = "/version.info";
const char_t *const kOppVersion = "Version=";
const size_t kEffectiveVersionNum = 2U;

std::string TransRequiredOppAbiVersionToString(const std::vector<std::pair<uint32_t, uint32_t>> &required_version) {
  std::stringstream ss;
  for (const auto &it : required_version) {
    ss << "[" << it.first << ", " << it.second << "], ";
  }
  return ss.str();
}
}  // namespace

void PluginManager::ClearHandles_() noexcept {
  for (const auto &handle : handles_) {
    if (mmDlclose(handle.second) != 0) {
      const char_t *error = mmDlerror();
      GE_IF_BOOL_EXEC(error == nullptr, error = "");
      GELOGW("Failed to close handle of %s, errmsg:%s", handle.first.c_str(), error);
    }
  }
  handles_.clear();
}

PluginManager::~PluginManager() { ClearHandles_(); }

Status PluginManager::GetOppPath(std::string &opp_path) {
  GELOGI("Enter get opp path schedule");
  char path_env[MMPA_MAX_PATH] = {'\0'};
  int32_t ret = mmGetEnv(kOppEnvName, path_env, MMPA_MAX_PATH);
  if ((ret == EN_OK) && (strlen(path_env) > 0U)) {
    opp_path = path_env;
    std::string file_path = RealPath(opp_path.c_str());
    if (file_path.empty()) {
      GELOGW("[Call][RealPath] File path %s is invalid.", opp_path.c_str());
    } else {
      GELOGI("Get opp path from env: %s", opp_path.c_str());
    }
    if (opp_path.back() != '/') {
      opp_path += '/';
    }
  }
  if (opp_path.empty()) {
    opp_path = GetModelPath();
    GELOGI("Get opp path from model path, value is %s", opp_path.c_str());
    opp_path = opp_path.substr(0, opp_path.rfind('/'));
    opp_path = opp_path.substr(0, opp_path.rfind('/') + 1U);
    opp_path += "ops/";
  }
  return SUCCESS;
}

bool PluginManager::IsNewOppPathStruct(const std::string &opp_path) {
  return mmIsDir((opp_path + kBuiltIn).c_str()) == EN_OK;
}

Status PluginManager::GetOppPluginVendors(const std::string &vendors_config, std::vector<std::string> &vendors) {
  GELOGI("Enter get opp plugin config file schedule, config file is '%s'", vendors_config.c_str());
  std::ifstream config(vendors_config);
  if (!config.good()) {
    GELOGI("Can not open file '%s'!", vendors_config.c_str());
    return FAILED;
  }
  std::string content;
  std::getline(config, content);
  config.close();
  GE_ASSERT_TRUE(!content.empty(), "Content of file '%s' is empty!", vendors_config.c_str());
  std::vector<std::string> v_parts;
  SplitPath(content, v_parts, '=');
  GE_ASSERT_TRUE(v_parts.size() == kVendorConfigPartsCount, "Format of file content is invalid!");
  SplitPath(v_parts[1], vendors, ',');
  GE_ASSERT_TRUE(!vendors.empty(), "Format of file content is invalid!");
  (void) for_each(vendors.begin(), vendors.end(), &StringUtils::Trim);
  return SUCCESS;
}

Status PluginManager::ReversePathString(std::string &path_str) {
  GELOGI("Enter ReversePathString schedule");
  if (path_str.empty() || (path_str.find(":") == std::string::npos)) {
    return SUCCESS;
  }
  std::vector<std::string> path_vec;
  SplitPath(path_str, path_vec, ':');
  GE_ASSERT_TRUE(!path_vec.empty(), "The vector path_vec should not be empty!");
  auto it = path_vec.crbegin();
  path_str = *(it++);
  while (it != path_vec.crend()) {
    path_str += ":" + *(it++);
  }
  GELOGI("path_str is '%s'", path_str.c_str());
  return SUCCESS;
}

void PluginManager::GetPluginPathFromCustomOppPath(const std::string &sub_path, std::string &plugin_path) {
  GELOGI("Start to get plugin path from ASCEND_CUSTOM_OPP_PATH schedule.");
  plugin_path = "";
  const char *const custom_opp_path_env = std::getenv("ASCEND_CUSTOM_OPP_PATH");
  if (custom_opp_path_env == nullptr) {
    GELOGI("env ASCEND_CUSTOM_OPP_PATH is not defined.");
    return;
  }
  const std::string custom_opp_path = custom_opp_path_env;
  if (custom_opp_path.empty()) {
    GELOGW("env ASCEND_CUSTOM_OPP_PATH is defined but it's empty.");
    return;
  }
  GELOGI("value of env ASCEND_CUSTOM_OPP_PATH is %s.", custom_opp_path.c_str());
  std::vector<std::string> custom_paths = StringUtils::Split(custom_opp_path, ':');
  for (const auto &custom_path : custom_paths) {
    if ((!custom_path.empty()) && (mmIsDir((custom_path + "/" + sub_path).c_str()) == EN_OK)) {
      if (IsVendorVersionValid(custom_path)) {
        plugin_path += custom_path + "/" + sub_path + ":";
        GELOGI("custom_path '%s' is valid.", custom_path.c_str());
      }
    } else {
      GELOGI("custom_path '%s' is invalid, which is skipped.", custom_path.c_str());
    }
  }
  GELOGI("Run GetPluginPathFromCustomOppPath finished, current plugin_path is %s.", plugin_path.c_str());
}

Status PluginManager::GetOppPluginPathOld(const std::string &opp_path,
                                          const std::string &path_fmt,
                                          std::string &plugin_path,
                                          const std::string &path_fmt_custom) {
  GELOGI("Enter get opp plugin path old schedule");
  const std::string &fmt_custom  = path_fmt_custom.empty() ? path_fmt : path_fmt_custom;
  plugin_path = (opp_path + std::regex_replace(fmt_custom, std::regex("%s"), "custom") + ":")
              + (opp_path + std::regex_replace(path_fmt, std::regex("%s"), "built-in"));
  GELOGI("plugin_path is '%s'", plugin_path.c_str());
  return SUCCESS;
}

bool PluginManager::GetRequiredOppAbiVersion(std::vector<std::pair<uint32_t, uint32_t>> &required_opp_abi_version) {
  std::string model_path = GetModelPath();
  GELOGD("Current lib path is:%s", model_path.c_str());
  model_path = model_path.substr(0, model_path.rfind('/'));
  model_path = model_path.substr(0, model_path.rfind('/'));
  model_path = model_path.substr(0, model_path.rfind('/') + 1);
  GELOGD("Run package path is:%s", model_path.c_str());

  std::string version_path;
  if (mmIsDir((model_path + kCompilerPath).c_str()) == EN_OK) {
    version_path = model_path + kCompilerPath + kVersionInfo;
  } else if (mmIsDir((model_path + kRuntimePath).c_str()) == EN_OK) {
    version_path = model_path + kRuntimePath + kVersionInfo;
  } else {
    GELOGW("compiler and runtime not exited");
    return true;
  }
  GELOGI("extract required opp abi version info from %s", version_path.c_str());

  std::string version;
  if (!PluginManager::GetVersionFromPathWithName(version_path, version, kRequiredOppAbiVersion)) {
    GELOGW("Not get required_opp_abi_version from path:%s", version_path.c_str());
    return true;
  }

  // valid required_opp_abi_version: ">=6.3, <=6.4, 6.4"
  version = StringUtils::ReplaceAll(version, "\"", "");
  StringUtils::Trim(version);
  std::queue<std::string> split_version;
  for (auto &it : StringUtils::Split(version, ',')) {
    split_version.emplace(StringUtils::Trim(it));
  }
  while (!split_version.empty()) {
    auto first = split_version.front();
    split_version.pop();
    if (StringUtils::StartWith(first, ">=")) {
      if (split_version.empty() || !StringUtils::StartWith(split_version.front(), "<=")) {
        GELOGW("Format of required_opp_abi_version [%s] is invalid, start with >= but not end with <=",
               version.c_str());
        return false;
      }
      auto second = split_version.front();
      split_version.pop();
      first = first.substr(kEffectiveVersionNum, first.size() - kEffectiveVersionNum);
      second = second.substr(kEffectiveVersionNum, second.size() - kEffectiveVersionNum);
      uint32_t first_num = 0U;
      if (!GetEffectiveVersion(first, first_num)) {
        GELOGW("[InvalidVersion] Format of required_opp_abi_version [%s] is not invalid", version.c_str());
        return false;
      }
      uint32_t second_num = 0U;
      if (!GetEffectiveVersion(second, second_num)) {
        GELOGW("[InvalidVersion] Format of required_opp_abi_version [%s] is not invalid", version.c_str());
        return false;
      }
      required_opp_abi_version.emplace_back(first_num, second_num);
    } else {
      uint32_t tmp_num = 0U;
      if (!GetEffectiveVersion(first, tmp_num)) {
        GELOGW("[InvalidVersion] Format of required_opp_abi_version [%s] is not invalid", version.c_str());
        return false;
      }
      required_opp_abi_version.emplace_back(tmp_num, tmp_num);
    }
  }
  return true;
}

bool PluginManager::GetEffectiveVersion(const std::string &opp_version, uint32_t &effective_version) {
  const auto split_version = StringUtils::Split(opp_version, '.');
  GE_ASSERT_TRUE(split_version.size() >= kEffectiveVersionNum);
  std::stringstream ss;
  ss << split_version[0];        // Cann version
  ss << split_version[1];        // C version
  ss >> effective_version;
  if (ss.fail() || !ss.eof()) {
    GELOGW("Can not convert [%s] to number from %s", ss.str().c_str(), opp_version.c_str());
    return false;
  }
  GELOGD("Get effective version:%u from %s", effective_version, opp_version.c_str());
  return true;
}

bool PluginManager::IsVendorVersionValid(const std::string &vendor_path) {
  // 获取opp包版本号：内置算子包字段为opp_version， 自定义算子包字段为compiler_version
  std::string opp_version;
  std::string compiler_version;
  GetOppAndCompilerVersion(vendor_path, opp_version, compiler_version);
  if (opp_version.empty() && compiler_version.empty()) {
    GELOGW("[NotVerification] Will not verify version as the opp version and compiler version are not set");
    return true;
  }
  return IsVendorVersionValid(opp_version, compiler_version);
}

bool PluginManager::IsVendorVersionValid(const std::string &opp_version, const std::string &compiler_version) {
  // 获取compiler或者runtime包支持的版本号范围
  std::vector<std::pair<uint32_t, uint32_t>> required_opp_abi_version;
  GE_ASSERT_TRUE(GetRequiredOppAbiVersion(required_opp_abi_version), "Get required opp abi version failed.");
  if (required_opp_abi_version.empty()) {
    GELOGW("[NotVerification] Will not verify version as the required_opp_abi_version are not set");
    return true;
  }

  // 校验版本号是否在支持的版本号列表范围内
  return CheckOppAndCompilerVersions(opp_version, compiler_version, required_opp_abi_version);
}

bool IsVersionWithInRequiredRange(const uint32_t effective_version,
                                  const std::vector<std::pair<uint32_t, uint32_t>> &required_version) {
  for (const auto &it : required_version) {
    if ((effective_version >= it.first) && (effective_version <= it.second)) {
      GELOGD("[ValidVersion] Effective version:%u within the required range.", effective_version);
      return true;
    }
  }

  GELOGW("[InvalidVersion] Effective version:%u not within the required range.", effective_version);
  return false;
}

void PluginManager::GetOppAndCompilerVersion(const std::string &vendor_path, std::string &opp_version,
                                             std::string &compiler_version) {
  std::string version_path;
  if (vendor_path.find(kBuiltIn) != std::string::npos) {
    version_path = vendor_path.substr(0, vendor_path.rfind("/")) + kVersionInfo;
    (void) PluginManager::GetVersionFromPathWithName(version_path, opp_version, kOppVersion);
    GELOGD("Get opp_version:%s", opp_version.c_str());
  } else {
    version_path = vendor_path + kVersionInfo;
    (void) PluginManager::GetVersionFromPathWithName(version_path, compiler_version, kCompilerVersion);
    GELOGD("Get compiler_version:%s", compiler_version.c_str());
  }
  return;
}

bool PluginManager::CheckOppAndCompilerVersions(const std::string &opp_version, const std::string &compiler_version,
                                                const std::vector<std::pair<uint32_t, uint32_t>> &required_version) {
  std::vector<uint32_t> effective_opp_versions;
  if (!opp_version.empty()) {
    uint32_t effective_opp_version = 0U;
    if (!GetEffectiveVersion(opp_version, effective_opp_version)) {
      GELOGW("[InvalidVersion] Format of opp version [%s] is invalid", opp_version.c_str());
      return false;
    }
    if (!IsVersionWithInRequiredRange(effective_opp_version, required_version)) {
      GELOGW("opp_version:%s is not with in required_opp_abi_version:%s",
             opp_version.c_str(), TransRequiredOppAbiVersionToString(required_version).c_str());
      return false;
    }
  }

  if (!compiler_version.empty()) {
    for (const auto &it : StringUtils::Split(compiler_version, ',')) {
      uint32_t effective_compiler_version = 0U;
      if (!GetEffectiveVersion(it, effective_compiler_version)) {
        GELOGW("[InvalidVersion] Format of compiler version [%s] is invalid", it.c_str());
        return false;
      }
      if (!IsVersionWithInRequiredRange(effective_compiler_version, required_version)) {
        GELOGW("compiler version:%s is not with in required_opp_abi_version:%s",
               opp_version.c_str(), TransRequiredOppAbiVersionToString(required_version).c_str());
        return false;
      }
    }
  }

  GELOGD("[ValidVersion] opp version:%s and compiler version:%s are within the required range:%s",
         opp_version.c_str(), compiler_version.c_str(),
         TransRequiredOppAbiVersionToString(required_version).c_str());
  return true;
}

// 需要打包进om的so优先级：
// 1. ASCEND_CUSTOM_OPP_PATH
// 2. ASCEND_OPP_PATH + vendors + 厂商名
// 3. ASCEND_OPP_PATH + build-in
void PluginManager::GetPackageSoPath(std::vector<std::string> &vendors) {
  std::string custom_opp_path;
  ge::PluginManager::GetPluginPathFromCustomOppPath("", custom_opp_path);
  if (!custom_opp_path.empty()) {
    std::vector<std::string> split_custom_opp_path = ge::StringUtils::Split(custom_opp_path, ':');
    vendors.insert(vendors.end(), split_custom_opp_path.begin(), split_custom_opp_path.end());
  }

  std::string opp_path;
  if (ge::PluginManager::GetOppPath(opp_path) == ge::SUCCESS) {
    std::string vendors_path;
    (void) ge::PluginManager::GetOppPluginPathNew(opp_path, "%s", vendors_path, "");
    if (!vendors_path.empty()) {
      auto split_vendors_path = ge::StringUtils::Split(vendors_path, ':');
      vendors.insert(vendors.end(), split_vendors_path.begin(), split_vendors_path.end());
    }
  }
  return;
}

Status PluginManager::GetOppPluginPathNew(const std::string &opp_path,
                                          const std::string &path_fmt,
                                          std::string &plugin_path,
                                          const std::string &old_custom_path,
                                          const std::string &path_fmt_custom) {
  GELOGI("Enter get opp plugin path new schedule");
  const std::string vendors_config = opp_path + kVendors + "/" + kConfig;
  std::vector<std::string> vendors;
  if (GetOppPluginVendors(vendors_config, vendors) != SUCCESS) {
    GELOGI("Can not get opp plugin vendors!");
    plugin_path += opp_path + old_custom_path + ":";
  } else {
    const std::string &fmt_custom  = path_fmt_custom.empty() ? path_fmt : path_fmt_custom;
    for (const auto &vendor : vendors) {
      if (IsVendorVersionValid(opp_path + kVendors + "/" + vendor)) {
        plugin_path += opp_path + kVendors + "/" + std::regex_replace(fmt_custom, std::regex("%s"), vendor) + ":";
      }
    }
  }
  if (IsVendorVersionValid(opp_path + "/" + kBuiltIn)) {
    plugin_path += opp_path + std::regex_replace(path_fmt, std::regex("%s"), "built-in");
  }
  GELOGI("plugin_path is '%s'", plugin_path.c_str());
  return SUCCESS;
}

Status PluginManager::GetOpsProtoPath(std::string &opsproto_path) {
  GELOGI("Enter GetOpsProtoPath schedule");
  std::string opp_path;
  GE_ASSERT_TRUE(GetOppPath(opp_path) == SUCCESS, "Failed to get opp path!");
  if (!IsNewOppPathStruct(opp_path)) {
    GELOGI("Opp plugin path structure is old version!");
    return GetOppPluginPathOld(opp_path, "op_proto/%s/", opsproto_path);
  } else {
    GELOGI("Opp plugin path structure is new version!");
    GetPluginPathFromCustomOppPath("op_proto/", opsproto_path);
    return GetOppPluginPathNew(opp_path, "%s/op_proto/", opsproto_path, "op_proto/custom/");
  }
}

Status PluginManager::GetCustomOpPath(const std::string &fmk_type, std::string &customop_path) {
  GELOGI("Enter GetCustomOpPath schedule");
  std::string opp_path;
  GE_ASSERT_TRUE(GetOppPath(opp_path) == SUCCESS, "Failed to get opp path!");
  if (!IsNewOppPathStruct(opp_path)) {
    GELOGI("Opp plugin path structure is old version!");
    return GetOppPluginPathOld(opp_path, "framework/%s/" + fmk_type + "/", customop_path, "framework/%s/");
  } else {
    GELOGI("Opp plugin path structure is new version!");
    GetPluginPathFromCustomOppPath("framework/", customop_path);
    return GetOppPluginPathNew(opp_path, "%s/framework/" + fmk_type + "/", customop_path, "framework/custom/",
                               "%s/framework/");
  }
}

Status PluginManager::GetCustomCaffeProtoPath(std::string &customcaffe_path) {
  GELOGD("Enter GetCustomCaffeProtoPath schedule");
  std::string opp_path;
  GE_ASSERT_TRUE(GetOppPath(opp_path) == SUCCESS, "Failed to get opp path!");
  if (!IsNewOppPathStruct(opp_path)) {
    customcaffe_path = opp_path + "framework/custom/caffe/";
    GELOGI("Opp plugin path structure is old version! customcaffe_path is '%s'", customcaffe_path.c_str());
    return SUCCESS;
  } else {
    GELOGI("Opp plugin path structure is new version!");
    GetPluginPathFromCustomOppPath("framework/caffe/", customcaffe_path);
    const std::string vendors_config = opp_path + kVendors + "/" + kConfig;
    std::vector<std::string> vendors;
    if (GetOppPluginVendors(vendors_config, vendors) != SUCCESS) {
      GELOGI("Can not get opp plugin vendors!");
      customcaffe_path += opp_path + "framework/custom/caffe/";
    } else {
      for (const auto &vendor : vendors) {
        customcaffe_path += (customcaffe_path.empty() || (customcaffe_path.back() == ':')) ? "" : ":";
        customcaffe_path += opp_path + kVendors + "/" + vendor + "/framework/caffe/";
      }
    }
    GELOGI("customcaffe_path is '%s'", customcaffe_path.c_str());
    return SUCCESS;
  }
}

Status PluginManager::GetOpTilingForwardOrderPath(std::string &op_tiling_path) {
  GELOGI("Enter GetOpTilingPath schedule");
  std::string opp_path;
  GE_ASSERT_TRUE(GetOppPath(opp_path) == SUCCESS, "Failed to get opp path!");
  if (!IsNewOppPathStruct(opp_path)) {
    GELOGI("Opp plugin path structure is old version!");
    GE_ASSERT_TRUE(GetOppPluginPathOld(opp_path, "op_impl/%s/ai_core/tbe/", op_tiling_path) == SUCCESS,
                   "GetOppPluginPathOld failed!");
  } else {
    GELOGI("Opp plugin path structure is new version!");
    GetPluginPathFromCustomOppPath("op_impl/ai_core/tbe/", op_tiling_path);
    GE_ASSERT_TRUE(GetOppPluginPathNew(opp_path, "%s/op_impl/ai_core/tbe/", op_tiling_path,
                                       "op_impl/custom/ai_core/tbe/") == SUCCESS,
                   "GetOppPluginPathNew failed!");
  }
  return SUCCESS;
}

Status PluginManager::GetOpTilingPath(std::string &op_tiling_path) {
  GE_ASSERT_SUCCESS(GetOpTilingForwardOrderPath(op_tiling_path));
  return ReversePathString(op_tiling_path);
}

Status PluginManager::GetConstantFoldingOpsPath(const std::string &path_base, std::string &constant_folding_ops_path) {
  GELOGI("Enter GetConstantFoldingOpsPath schedule");
  std::string opp_path;
  Status ret = GetOppPath(opp_path);
  if (ret != SUCCESS) {
    GELOGW("Failed to get opp path from env and so file! use path_base as opp path");
    opp_path = path_base;
  }
  GE_ASSERT_TRUE(!opp_path.empty(), "[Check]Value of opp_path should not be empty here!");
  if (!IsNewOppPathStruct(opp_path)) {
    constant_folding_ops_path = opp_path + kHostCpuLibRelativePathV01;
  } else {
    constant_folding_ops_path = opp_path + kHostCpuLibRelativePathV02;
  }
  return SUCCESS;
}

void PluginManager::SplitPath(const std::string &mutil_path, std::vector<std::string> &path_vec, const char sep) {
  const std::string tmp_string = mutil_path + sep;
  std::string::size_type start_pos = 0U;
  std::string::size_type cur_pos = tmp_string.find(sep, 0U);
  while (cur_pos != std::string::npos) {
    const std::string path = tmp_string.substr(start_pos, cur_pos - start_pos);
    if (!path.empty()) {
      path_vec.push_back(path);
    }
    start_pos = cur_pos + 1U;
    cur_pos = tmp_string.find(sep, start_pos);
  }
}

Status PluginManager::LoadSo(const std::string &path, const std::vector<std::string> &func_check_list) {
  const int32_t flags = static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) |
      static_cast<uint32_t>(MMPA_RTLD_GLOBAL));
  return LoadSoWithFlags(path, flags, func_check_list);
}

Status PluginManager::LoadSoWithFlags(const std::string &path, const int32_t flags,
    const std::vector<std::string> &func_check_list) {
  uint32_t num_of_loaded_so = 0U;
  int64_t size_of_loaded_so = 0;
  so_list_.clear();
  ClearHandles_();

  std::vector<std::string> path_vec;
  SplitPath(path, path_vec);
  for (const auto &single_path : path_vec) {
    GE_IF_BOOL_EXEC(single_path.length() >= static_cast<ULONG>(MMPA_MAX_PATH), GELOGE(PARAM_INVALID,
                    "The shared library file path is too long!");
                    continue);
    // load break when number of loaded so reach maximum
    if (num_of_loaded_so >= static_cast<uint32_t>(kMaxNumOfSo)) {
      GELOGW("The number of dynamic libraries loaded exceeds the kMaxNumOfSo,"
             " and only the first %d shared libraries will be loaded.", kMaxNumOfSo);
      break;
    }

    std::string file_name = single_path.substr(single_path.rfind('/') + 1U, std::string::npos);
    const std::string file_path_dlopen = RealPath(single_path.c_str());
    if (file_path_dlopen.empty()) {
      GELOGW("Failed to get realpath of %s!", single_path.c_str());
      continue;
    }

    int64_t file_size = 0;
    if (ValidateSo(file_path_dlopen, size_of_loaded_so, file_size) != SUCCESS) {
      GELOGW("Failed to validate the shared library: %s", file_path_dlopen.c_str());
      continue;
    }

    GELOGI("dlopen path: %s, flags is %d", file_path_dlopen.c_str(), flags);

    // load continue when dlopen is failed
    const auto handle = mmDlopen(file_path_dlopen.c_str(), flags);
    if (handle == nullptr) {
      const char_t *error = mmDlerror();
      GE_IF_BOOL_EXEC(error == nullptr, error = "");
      REPORT_INNER_ERROR("E19999", "DLOpen SharedLibraryPath failed, path[%s]. Errormessage[%s]!",
                         file_path_dlopen.c_str(), error);
      GELOGE(PATH_INVALID,
             "[DLOpen][SharedLibraryPath]Failed, path[%s]. Errormessage[%s]!",
             file_path_dlopen.c_str(), error);
      continue;
    }

    // load continue when so is invalid
    bool is_valid = true;
    for (const auto &func_name : func_check_list) {
      const auto real_fn = reinterpret_cast<void (*)()>(mmDlsym(handle, func_name.c_str()));
      if (real_fn == nullptr) {
        const char_t *error = mmDlerror();
        GE_IF_BOOL_EXEC(error == nullptr, error = "");
        REPORT_INNER_ERROR("E19999", "[Check][So]%s is skipped since function %s is not existed! errmsg:%s",
                           func_name.c_str(), func_name.c_str(), error);
        GELOGE(PARAM_INVALID,
               "[Check][So]%s is skipped since function %s is not existed! errmsg:%s",
               func_name.c_str(), func_name.c_str(), error);
        is_valid = false;
        break;
      }
    }
    if (!is_valid) {
      if (mmDlclose(handle) != 0) {
        const char_t *error = mmDlerror();
        GE_IF_BOOL_EXEC(error == nullptr, error = "");
        GELOGE(FAILED, "[DLClose][Handle]Failed. errmsg:%s", error);
      }
      continue;
    }

    // add file to list
    size_of_loaded_so += file_size;
    so_list_.emplace_back(file_name);
    handles_[std::string(file_name)] = handle;
    num_of_loaded_so++;
  }

  GELOGI("The total number of shared libraries loaded: %u", num_of_loaded_so);
  for (const auto &name : so_list_) {
    GELOGI("load %s successfully", name.c_str());
  }

  if (num_of_loaded_so == 0U) {
    GELOGW("No loadable shared library found in the path: %s", path.c_str());
    return SUCCESS;
  }

  return SUCCESS;
}

Status PluginManager::ValidateSo(const std::string &file_path,
                                 const int64_t size_of_loaded_so, int64_t &file_size) const {
  // read file size
  struct stat stat_buf;
  if (stat(file_path.c_str(), &stat_buf) != 0) {
    GELOGW("The shared library file check failed: %s", file_path.c_str());
    return FAILED;
  }

  // load continue when the size itself reaches maximum
  file_size = stat_buf.st_size;
  if (stat_buf.st_size > kMaxSizeOfSo) {
    GELOGW("The %s is skipped since its size exceeds maximum! (size: %ldB, maximum: %dB)", file_path.c_str(), file_size,
           kMaxSizeOfSo);
    return FAILED;
  }

  // load continue if the total size of so reaches maximum when it is loaded
  if ((size_of_loaded_so + file_size) > kMaxSizeOfLoadedSo) {
    GELOGW(
        "%s is skipped because the size of loaded share library reaches maximum if it is loaded! "
        "(size: %ldB, size of loaded share library: %ldB, maximum: %dB)",
        file_path.c_str(), file_size, size_of_loaded_so, kMaxSizeOfLoadedSo);
    return FAILED;
  }

  return SUCCESS;
}

Status PluginManager::Load(const std::string &path, const std::vector<std::string> &func_check_list) {
  const int32_t flags = static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) |
      static_cast<uint32_t>(MMPA_RTLD_GLOBAL));
  return LoadWithFlags(path, flags, func_check_list);
}

Status PluginManager::LoadWithFlags(const std::string &path, const int32_t flags,
    const std::vector<std::string> &func_check_list) {
  uint32_t num_of_loaded_so = 0U;
  int64_t size_of_loaded_so = 0;
  const uint32_t is_folder = 0x4U;
  const std::string ext = kExt;
  so_list_.clear();
  ClearHandles_();

  char_t err_buf[kMaxErrorStrLen + 1U] = {};
  char_t canonical_path[MMPA_MAX_PATH] = {};
  if (mmRealPath(path.c_str(), &canonical_path[0], MMPA_MAX_PATH) != EN_OK) {
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStrLen);
    GELOGW("Failed to get realpath of %s, errmsg:%s", path.c_str(), err_msg);
    return SUCCESS;
  }

  const int32_t is_dir = mmIsDir(&canonical_path[0]);
  // Lib plugin path not exist
  if (is_dir != EN_OK) {
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStrLen);
    GELOGW("Invalid path for load: %s, errmsg:%s", path.c_str(), err_msg);
    return SUCCESS;
  }

  mmDirent **entries = nullptr;
  const auto ret = mmScandir(&canonical_path[0], &entries, nullptr, nullptr);
  if (ret < EN_OK) {
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStrLen);
    GELOGW("scan dir failed. path = %s, ret = %d, errmsg = %s", &canonical_path[0], ret, err_msg);
    return FAILED;
  }
  for (int32_t i = 0; i < ret; ++i) {
    mmDirent * const entry = entries[i];
    // read fileName and fileType
    std::string file_name = entry->d_name;
    const auto file_type = static_cast<uint32_t>(entry->d_type);

    // ignore folder
    const bool invalid_file = ((file_type == is_folder) ||
                         // ignore file whose name length is less than 3
                         (file_name.size() <= ext.size()) ||
                         // ignore file whose extension is not so
                         (file_name.compare(file_name.size() - ext.size(), ext.size(), ext) != 0));
    if (invalid_file) {
      continue;
    }

    // load break when number of loaded so reach maximum
    if (num_of_loaded_so >= static_cast<uint32_t>(kMaxNumOfSo)) {
      GELOGW("The number of dynamic libraries loaded exceeds the kMaxNumOfSo,"
             " and only the first %d shared libraries will be loaded.", kMaxNumOfSo);
      break;
    }
    const std::string canonical_path_str = (std::string(canonical_path) + "/" + file_name);
    const std::string file_path_dlopen = RealPath(canonical_path_str.c_str());
    if (file_path_dlopen.empty()) {
      GELOGW("failed to get realpath of %s", canonical_path_str.c_str());
      continue;
    }

    int64_t file_size = 0;
    if (ValidateSo(file_path_dlopen, size_of_loaded_so, file_size) != SUCCESS) {
      GELOGW("Failed to validate the shared library: %s", canonical_path_str.c_str());
      continue;
    }

    GELOGI("Dlopen path: %s. flags is %d", file_path_dlopen.c_str(), flags);

    // load continue when dlopen is failed
    const auto handle = mmDlopen(file_path_dlopen.c_str(), flags);
    if (handle == nullptr) {
      const char_t *error = mmDlerror();
      GE_IF_BOOL_EXEC(error == nullptr, error = "");
      GELOGW("Failed in dlopen %s!", error);
      continue;
    }

    GELOGW("The shared library will not be checked. Please ensure that the source of the shared library is trusted.");

    // load continue when so is invalid
    bool is_valid = true;
    for (const auto &func_name : func_check_list) {
      const auto real_fn = reinterpret_cast<void (*)()>(mmDlsym(handle, func_name.c_str()));
      if (real_fn == nullptr) {
        const char_t *error = mmDlerror();
        GE_IF_BOOL_EXEC(error == nullptr, error = "");
        GELOGW("The %s is skipped since function %s is not existed! errmsg:%s",
               file_name.c_str(), func_name.c_str(), error);
        is_valid = false;
        break;
      }
    }
    if (!is_valid) {
      if (mmDlclose(handle) != 0) {
        const char_t *error = mmDlerror();
        GE_IF_BOOL_EXEC(error == nullptr, error = "");
        GELOGE(FAILED, "[DLClose][Handle]Failed. errmsg:%s", error);
      }
      continue;
    }

    // add file to list
    size_of_loaded_so += file_size;
    so_list_.emplace_back(file_name);
    handles_[std::string(file_name)] = handle;
    num_of_loaded_so++;
  }
  mmScandirFree(entries, ret);
  if (num_of_loaded_so == 0U) {
    GELOGW("No loadable shared library found in the path: %s", path.c_str());
    return SUCCESS;
  }

  return SUCCESS;
}

void PluginManager::GetOppSupportedOsAndCpuType(
    std::unordered_map<std::string, std::unordered_set<std::string>> &opp_supported_os_cpu,
    std::string opp_path, std::string os_name, uint32_t layer) {
  if (layer > kLibSecondLayer) {
    GELOGW("The lib structure of the current opp package has only 2 layers");
    return;
  }
  GELOGD("Enter GetOppSupportedOsAndCpuType schedule");

  if (opp_path.empty()) {
    (void) GetOppPath(opp_path);
    opp_path += "built-in/op_proto/lib/";
    if (opp_path.size() >= static_cast<size_t>(MMPA_MAX_PATH)) {
      GELOGW("param path size:%zu >= max path:%d", opp_path.size(), MMPA_MAX_PATH);
      return;
    }
  }

  char_t real_path[MMPA_MAX_PATH] = {};
  if (mmRealPath(opp_path.c_str(), &(real_path[0U]), MMPA_MAX_PATH) != EN_OK) {
    GELOGW("Can not get real path:%s, it may be an old version", opp_path.c_str());
    return;
  }

  if (mmIsDir(&(real_path[0U])) != EN_OK) {
    GELOGW("Path %s is not directory, it may be an old version", real_path);
    return;
  }

  mmDirent **entries = nullptr;
  const auto ret = mmScandir(&(real_path[0U]), &entries, nullptr, nullptr);
  if (ret < EN_OK) {
    GELOGW("Can not open directory %s, it may be an old version, ret = %d", real_path, ret);
    return;
  }
  for (int32_t i = 0; i < ret; ++i) {
    const mmDirent *const dir_ent = *PtrAdd<mmDirent*>(entries, static_cast<size_t>(ret), static_cast<size_t>(i));
    if (static_cast<int32_t>(dir_ent->d_type) == DT_DIR) {
      std::string dir_name = dir_ent->d_name;
      if ((dir_name.compare(".") == 0) || (dir_name.compare("..") == 0)) {
        continue;
      }
      if ((layer == kLibFirstLayer) && (opp_supported_os_cpu.find(dir_name) == opp_supported_os_cpu.end())) {
        opp_supported_os_cpu[dir_name] = {};
        GetOppSupportedOsAndCpuType(opp_supported_os_cpu, opp_path + dir_name, dir_name, 1U);
      }
      if (layer == kLibSecondLayer) {
        opp_supported_os_cpu[os_name].emplace(dir_name);
        GELOGD("Get supported os[%s] -> cpu[%s]", os_name.c_str(), dir_name.c_str());
      }
    }
  }
  mmScandirFree(entries, ret);
  return;
}

void PluginManager::GetCurEnvPackageOsAndCpuType(std::string &host_env_os, std::string &host_env_cpu) {
  GELOGD("Enter GetCurEnvPackageOsAndCpuType schedule");
  std::string model_path = GetModelPath();
  GELOGD("Current lib path is:%s", model_path.c_str());
  model_path = model_path.substr(0, model_path.rfind('/'));
  model_path = model_path.substr(0, model_path.rfind('/'));
  model_path = model_path.substr(0, model_path.rfind('/') + 1);
  GELOGD("Run package path is:%s", model_path.c_str());

  std::string scene;
  if (mmAccess2((model_path + kOppPath + kScene).c_str(), M_R_OK) == EN_OK) {
    scene = model_path + kOppPath + kScene;
  } else if (mmAccess2((model_path + kRuntimePath + kScene).c_str(), M_R_OK) == EN_OK) {
    scene = model_path + kRuntimePath + kScene;
  } else {
    GELOGW("opp and runtime not exit");
    return;
  }
  GELOGI("extract os and cpu info from %s", scene.c_str());
  std::ifstream ifs(scene);
  if (!ifs.good()) {
    GELOGW("Can not open file:%s", scene.c_str());
    return;
  }
  std::string line;
  while (std::getline(ifs, line)) {
    line = StringUtils::Trim(line);
    std::vector<std::string> value = StringUtils::Split(line, '=');
    if (value.size() != kSceneValueCount) {
      continue;
    }
    if (value[kSceneKeyIndex].compare(kSceneOs) == 0) {
      host_env_os = value[kSceneValueIndex];
      GELOGI("Get os:%s", host_env_os.c_str());
    }
    if (value[kSceneKeyIndex].compare(kSceneArch) == 0) {
      host_env_cpu = value[kSceneValueIndex];
      GELOGI("Get cpu:%s", host_env_cpu.c_str());
    }
  }
  return;
}

bool PluginManager::GetVersionFromPathWithName(const std::string &file_path, std::string &version,
                                               const std::string version_name) {
  // Normalize the path
  std::string resolved_file_path = RealPath(file_path.c_str());
  if (resolved_file_path.empty()) {
    GELOGW("Invalid input file path [%s], make sure that the file path is correct.", file_path.c_str());
    return false;
  }
  std::ifstream fs(resolved_file_path, std::ifstream::in);
  if (!fs.is_open()) {
    GELOGW("Open %s failed.", file_path.c_str());
    return false;
  }

  std::string line;
  while (std::getline(fs, line)) {
    if (ParseVersion(line, version, version_name)) {
      GELOGD("Parse version success. content is [%s].", line.c_str());
      fs.close();
      return true;
    }
  }

  GELOGW("No version information found in the file path:%s", file_path.c_str());
  fs.close();
  return false;
}

bool PluginManager::GetVersionFromPath(const std::string &file_path, std::string &version) {
  return GetVersionFromPathWithName(file_path, version, kOppVersion);
}

// Parsing the command line
bool PluginManager::ParseVersion(std::string &line, std::string &version, const std::string version_name) {
  line = StringUtils::Trim(line);
  if (line.empty()) {
    GELOGW("line is empty.");
    return false;
  }

  std::string::size_type pos = line.find(version_name);
  if (pos == std::string::npos) {
    GELOGW("Incorrect line [%s], it must include [%s].", line.c_str(), version_name.c_str());
    return false;
  }

  if (line.size() == version_name.size()) {
    GELOGW("version information is empty. %s", line.c_str());
    return false;
  }

  version = line.substr(pos + version_name.size());

  return true;
}

void PluginManager::GetFileListWithSuffix(const std::string &path, const std::string &so_suff,
                                          std::vector<std::string> &file_list) {
  if (path.empty()) {
    GELOGI("realPath is empty");
    return;
  }
  if (path.size() >= static_cast<size_t>(MMPA_MAX_PATH)) {
    REPORT_INNER_ERROR("E18888", "param path size:%zu >= max path:%d", path.size(), MMPA_MAX_PATH);
    GELOGE(FAILED, "param path size:%zu >= max path:%d", path.size(), MMPA_MAX_PATH);
    return;
  }

  char_t resolved_path[MMPA_MAX_PATH] = {};

  // Nullptr is returned when the path does not exist or there is no permission
  // Return absolute path when path is accessible
  if (mmRealPath(path.c_str(), &(resolved_path[0U]), MMPA_MAX_PATH) != EN_OK) {
    GELOGW("[FindSo][Check] Get real_path for file %s failed, reason:%s", path.c_str(), strerror(errno));
    return;
  }

  const INT32 is_dir = mmIsDir(&(resolved_path[0U]));
  if (is_dir != EN_OK) {
    GELOGW("[FindSo][Check] Open directory %s failed, maybe it is not exit or not a dir, errmsg:%s",
           &(resolved_path[0U]), strerror(errno));
    return;
  }

  mmDirent **entries = nullptr;
  const auto file_num = mmScandir(&(resolved_path[0U]), &entries, nullptr, nullptr);
  if ((file_num < 0) || (entries == nullptr)) {
    GELOGW("[FindSo][Scan] Scan directory %s failed, ret:%d, reason:%s",
           &(resolved_path[0U]), file_num, strerror(errno));
    return;
  }
  for (int32_t i = 0; i < file_num; ++i) {
    const mmDirent *const dir_ent = entries[static_cast<size_t>(i)];
    const std::string name = std::string(dir_ent->d_name);
    if ((strcmp(name.c_str(), ".") == 0) || (strcmp(name.c_str(), "..") == 0)) {
      continue;
    }

    if ((static_cast<int32_t>(dir_ent->d_type) != DT_DIR) && (name.size() >= so_suff.size()) &&
        (name.compare(name.size() - so_suff.size(), so_suff.size(), so_suff) == 0)) {
      const std::string full_name = path + "/" + name;
      file_list.push_back(full_name);
      GELOGI("PluginManager Parse full name = %s.", full_name.c_str());
    }
  }
  mmScandirFree(entries, file_num);
  GELOGI("Found %d libs.", file_list.size());
}
}  // namespace ge

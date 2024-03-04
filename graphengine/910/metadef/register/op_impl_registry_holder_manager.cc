/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

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

#include "register/op_impl_registry_holder_manager.h"
#include <fstream>
#include "graph/debug/ge_log.h"
#include "graph/utils/file_utils.h"
#include "mmpa/mmpa_api.h"
#include "inc/graph/operator_factory_impl.h"
#include "common/checker.h"

namespace gert {
namespace {
constexpr const ge::char_t *kHomeEnvName = "HOME";
constexpr size_t kGByteSize = 1073741824U; // 1024 * 1024 * 1024
static thread_local uint32_t load_so_count = 0;
using GetImplNum = size_t (*)();
using GetImplFunctions = ge::graphStatus (*)(TypesToImpl *imp, size_t impl_num);

void CloseHandle(void *&handle) {
  if (handle != nullptr) {
    GELOGD("start close handle, handle[%p].", handle);
    if (mmDlclose(handle) != 0) {
      const ge::char_t *error = mmDlerror();
      error = (error == nullptr) ? "" : error;
      GELOGE(ge::FAILED, "[Close][Handle] failed, reason:%s", error);
      return;
    }
  }
  handle = nullptr;
}
void ReleaseOperatorFactoryBeforeCloseHandle() {
  ge::OperatorFactoryImpl::operator_infer_axis_type_info_funcs_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_axis_slice_funcs_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_value_range_paras_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_data_slice_funcs_ = nullptr;
  ge::OperatorFactoryImpl::operator_verify_funcs_ = nullptr;
  ge::OperatorFactoryImpl::operator_inferformat_funcs_ = nullptr;
  ge::OperatorFactoryImpl::operator_infershape_funcs_ = nullptr;
  ge::OperatorFactoryImpl::operator_creators_v2_ = nullptr;
  ge::OperatorFactoryImpl::operator_creators_ = nullptr;
}
}

OpImplRegistryHolder::~OpImplRegistryHolder() {
  types_to_impl_.clear();
  CloseHandle(handle_);
}

ge::graphStatus OmOpImplRegistryHolder::CreateOmOppDir(std::string &opp_dir) const {
  opp_dir.clear();
  GE_ASSERT_SUCCESS(ge::GetAscendWorkPath(opp_dir));
  if (opp_dir.empty()) {
    ge::char_t path_env[MMPA_MAX_PATH] = {'\0'};
    const int32_t ret = mmGetEnv(kHomeEnvName, path_env, MMPA_MAX_PATH);
    if ((ret != EN_OK) || (strnlen(path_env, static_cast<size_t>(MMPA_MAX_PATH)) == 0U)) {
      GELOGE(ge::FAILED, "Get %s path failed.", kHomeEnvName);
      return ge::GRAPH_FAILED;
    }

    const std::string file_path = ge::RealPath(path_env);
    if (file_path.empty()) {
      GELOGE(ge::FAILED, "[Call][RealPath] File path %s is invalid.", opp_dir.c_str());
      return ge::GRAPH_FAILED;
    }
    opp_dir = file_path;
  }
  if (opp_dir.back() != '/') {
    opp_dir += '/';
  }
  opp_dir += ".ascend_temp/.om_exe_data/"
      + std::to_string(mmGetPid())
      + "_" + std::to_string(mmGetTid())
      + "_" + std::to_string(load_so_count++)
      + "/";
  GELOGD("opp_dir is %s", opp_dir.c_str());

  GE_ASSERT_TRUE(mmAccess2(opp_dir.c_str(), M_F_OK) != EN_OK);
  GE_ASSERT_TRUE(ge::CreateDir(opp_dir) == 0);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus OmOpImplRegistryHolder::RmOmOppDir(const std::string &opp_dir) const {
  if (opp_dir.empty()) {
    GELOGD("opp dir is empty, no need remove");
    return ge::GRAPH_SUCCESS;
  }

  if (mmRmdir(opp_dir.c_str()) != 0) {
    const ge::char_t *error = mmDlerror();
    error = (error == nullptr) ? "" : error;
    GELOGE(ge::FAILED, "Failed to rm dir %s, errmsg: %s", opp_dir.c_str(), error);
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus OmOpImplRegistryHolder::SaveToFile(const std::shared_ptr<ge::OpSoBin> &so_bin,
                                                   const std::string &opp_path) const {
  constexpr mmMode_t kAccess = static_cast<mmMode_t>(static_cast<uint32_t>(M_IRUSR) |
      static_cast<uint32_t>(M_IWUSR) |
      static_cast<uint32_t>(M_UMASK_USREXEC));
  const int32_t fd = mmOpen2(opp_path.c_str(),
                             static_cast<int32_t>(static_cast<uint32_t>(M_WRONLY) |
                                 static_cast<uint32_t>(M_CREAT) |
                                 static_cast<uint32_t>(O_TRUNC)),
                             kAccess);
  if (fd < 0) {
    GELOGE(ge::FAILED, "Failed to open file, path = %s", opp_path.c_str());
    return ge::GRAPH_FAILED;
  }
  const int32_t write_count = mmWrite(fd, const_cast<uint8_t *>(so_bin->GetBinData()),
                                      static_cast<uint32_t>(so_bin->GetBinDataSize()));
  if ((write_count == EN_INVALID_PARAM) || (write_count == EN_ERROR)) {
    GELOGE(ge::FAILED, "Write data failed. mmpa error no is %d", write_count);
    GE_ASSERT_TRUE(mmClose(fd) == EN_OK);
    return ge::GRAPH_FAILED;
  }
  GE_ASSERT_TRUE(mmClose(fd) == EN_OK);
  return ge::GRAPH_SUCCESS;
}

std::unique_ptr<TypesToImpl[]> OpImplRegistryHolder::GetOpImplFunctionsByHandle(const void *handle,
                                                                                const std::string &so_path,
                                                                                size_t &impl_num) const {
  if (handle == nullptr) {
    GELOGE(ge::FAILED, "handle is nullptr");
    return nullptr;
  }

  // 兼容1.0注册方式注册自定义算子so，查找不到符号告警返回
  const auto get_impl_num = reinterpret_cast<GetImplNum>(mmDlsym(const_cast<void *>(handle), "GetRegisteredOpNum"));
  if (get_impl_num == nullptr) {
    const ge::char_t *error = mmDlerror();
    error = (error == nullptr) ? "" : error;
    GELOGW("Get registered op num functions failed, path:%s, errmsg:%s", so_path.c_str(), error);
    return nullptr;
  }
  impl_num = get_impl_num();
  GELOGD("get_impl_num: %zu", impl_num);
  GE_ASSERT_TRUE((impl_num != 0U), "get impl num is %zu", impl_num);

  const auto get_impl_funcs
      = reinterpret_cast<GetImplFunctions>(mmDlsym(const_cast<void *>(handle), "GetOpImplFunctions"));
  if (get_impl_funcs == nullptr) {
    const ge::char_t *error = mmDlerror();
    error = (error == nullptr) ? "" : error;
    GELOGW("Get op impl functions failed, path:%s, errmsg:%s", so_path.c_str(), error);
    return nullptr;
  }

  auto impl_funcs = std::unique_ptr<TypesToImpl[]>(new(std::nothrow) TypesToImpl[impl_num]);
  if (impl_funcs == nullptr) {
    GELOGE(ge::FAILED, "New unique ptr failed");
    return nullptr;
  }

  if (get_impl_funcs(reinterpret_cast<TypesToImpl *>(impl_funcs.get()), impl_num) != ge::GRAPH_SUCCESS) {
    GELOGE(ge::FAILED, "GetOpImplFunctions execute failed");
    return nullptr;
  }

  for (size_t i = 0U; i < impl_num; ++i) {
    GELOGD("impl_funcs[%zu], optype: %s", i, impl_funcs[i].op_type);
  }

  return impl_funcs;
}

void OpImplRegistryHolder::AddTypesToImpl(const gert::OpImplKernelRegistry::OpType op_type,
                                          const gert::OpImplKernelRegistry::OpImplFunctions funcs) {
  types_to_impl_[op_type] = funcs;
}

ge::graphStatus OmOpImplRegistryHolder::LoadSo(const std::shared_ptr<ge::OpSoBin> &so_bin) {
  if (so_bin->GetBinDataSize() > kGByteSize) {
    GELOGE(ge::FAILED, "The size of so bin is %zu, more than %zu", so_bin->GetBinDataSize(), kGByteSize);
    return ge::GRAPH_FAILED;
  }

  std::string opp_dir;
  GE_ASSERT_SUCCESS(CreateOmOppDir(opp_dir));

  const std::string so_path = opp_dir + so_bin->GetSoName();
  if (SaveToFile(so_bin, so_path) != ge::GRAPH_SUCCESS) {
    GE_ASSERT_SUCCESS(RmOmOppDir(opp_dir));
    return ge::GRAPH_FAILED;
  }

  void *handle = mmDlopen(so_path.c_str(),
                          static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) |
                              static_cast<uint32_t>(MMPA_RTLD_GLOBAL)));
  if (handle == nullptr) {
    const ge::char_t *error = mmDlerror();
    error = (error == nullptr) ? "" : error;
    GELOGE(ge::FAILED, "Failed to dlopen %s, errmsg: %s", so_path.c_str(), error);
    GE_ASSERT_SUCCESS(RmOmOppDir(opp_dir));
    return ge::GRAPH_FAILED;
  }

  size_t impl_num = 0;
  const auto impl_funcs = GetOpImplFunctionsByHandle(handle, so_path, impl_num);
  if (impl_funcs == nullptr) {
    CloseHandle(handle);
    GE_ASSERT_SUCCESS(RmOmOppDir(opp_dir));
    return ge::GRAPH_FAILED;
  }

  GE_ASSERT_SUCCESS(RmOmOppDir(opp_dir));

  for (size_t i = 0U; i < impl_num; ++i) {
    types_to_impl_[impl_funcs[i].op_type] = impl_funcs[i].funcs;
  }

  handle_ = handle;

  return ge::GRAPH_SUCCESS;
}

OpImplRegistryHolderManager &OpImplRegistryHolderManager::GetInstance() {
  static OpImplRegistryHolderManager instance;
  return instance;
}

void OpImplRegistryHolderManager::AddRegistry(std::string &so_data,
                                              const std::shared_ptr<OpImplRegistryHolder> &registry_holder) {
  // AddRegistry 前先刷新OpImplRegistryManager
  UpdateOpImplRegistries();
  const std::lock_guard<std::mutex> lock(map_mutex_);
  const auto iter = op_impl_registries_.find(so_data);
  if (iter == op_impl_registries_.cend()) {
    op_impl_registries_[so_data] = registry_holder;
  }
}

void OpImplRegistryHolderManager::UpdateOpImplRegistries() {
  const std::lock_guard<std::mutex> lock(map_mutex_);
  auto iter = op_impl_registries_.begin();
  while (iter != op_impl_registries_.end()) {
    if (iter->second == nullptr) {
      (void)op_impl_registries_.erase(iter++);
    } else {
      iter++;
    }
  }
}

const std::shared_ptr<OpImplRegistryHolder> OpImplRegistryHolderManager::GetOpImplRegistryHolder(std::string &so_data) {
  const std::lock_guard<std::mutex> lock(map_mutex_);
  const auto iter = op_impl_registries_.find(so_data);
  if (iter == op_impl_registries_.cend()) {
    return nullptr;
  }
  return iter->second;
}

OpImplRegistryHolderPtr OpImplRegistryHolderManager::GetOrCreateOpImplRegistryHolder(
    std::string &so_data,
    const std::string &so_name,
    const ge::SoInOmInfo &so_info,
    const std::function<OpImplRegistryHolderPtr()> create_func) {
  const std::lock_guard<std::mutex> lock(map_mutex_);
  const auto iter = op_impl_registries_.find(so_data);
  if (iter != op_impl_registries_.cend()) {
    auto holder = iter->second;
    if (holder != nullptr) {
      GEEVENT("so has been loaded, so name: %s, version:%s, cpu:%s, os:%s",
              so_name.c_str(),
              so_info.opp_version.c_str(),
              so_info.cpu_info.c_str(),
              so_info.os_info.c_str());
      return holder;
    }
  }
  if (create_func == nullptr) {
    GELOGE(ge::FAILED, "create_func is nullptr");
    return nullptr;
  }
  auto registry_holder = create_func();
  if (registry_holder == nullptr) {
    GELOGE(ge::FAILED, "create registry holder failed");
    return nullptr;
  }
  op_impl_registries_[so_data] = registry_holder;
  return registry_holder;
}
OpImplRegistryHolderManager::~OpImplRegistryHolderManager() {
  /**
   * todo 此处是临时规避方案，后续需要梳理算子的自注册机制，修改成space_registry注册机制
   * 此处临时地显示地指定这些自注册机制的static变量的析构时机(operator_infer_axis_type_info_funcs等static变量，默认在进程退出前析构)，
   * 显示地指定其在so句柄关闭之前进行析构。
   * */
  ReleaseOperatorFactoryBeforeCloseHandle();
}
}  // namespace gert

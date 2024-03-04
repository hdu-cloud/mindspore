/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "register/op_impl_registry_holder_manager.h"
#include "graph/any_value.h"
#include "graph/utils/file_utils.h"
#include "common/util/mem_utils.h"
#include "common/ge_common/debug/ge_log.h"
#include "mmpa/mmpa_api.h"
#include "tests/depends/mmpa/src/mmpa_stub.h"
#include <gtest/gtest.h>
#include <iostream>

namespace gert_test {
namespace {
#define MMPA_MAX_PATH 4096
constexpr const char *kHomeEnvName = "HOME";

ge::OpSoBinPtr CreateSoBinPtr(std::string &so_name, std::string &vendor_name) {
  std::unique_ptr<char[]> so_bin = std::unique_ptr<char[]>(new(std::nothrow) char[so_name.length()]);
  (void) memcpy_s(so_bin.get(), so_name.length(), so_name.data(), so_name.length());
  ge::OpSoBinPtr so_bin_ptr = ge::MakeShared<ge::OpSoBin>(so_name, vendor_name, std::move(so_bin), so_name.length());
  return so_bin_ptr;
}

size_t g_impl_num = 3;
char *type[] = {"Add_0", "Add_1", "Add_2"};
size_t GetRegisteredOpNum() { return g_impl_num; }
uint32_t GetOpImplFunctions(TypesToImpl *impl, size_t g_impl_num) {
  gert::OpImplKernelRegistry::OpImplFunctions funcs;
  for (int i = 0; i < g_impl_num; ++i) {
    funcs.tiling = (gert::OpImplKernelRegistry::TilingKernelFunc) (0x10 + i);
    funcs.infer_shape = (gert::OpImplKernelRegistry::InferShapeKernelFunc) (0x20 + i);
    impl[i].op_type = type[i];
    impl[i].funcs = funcs;
  }
  return 0;
}

void *mock_handle = nullptr;
bool dlsys_get_impl_func_fail = false;
class MockMmpa : public ge::MmpaStubApi {
 public:
  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "GetRegisteredOpNum") {
      return (void *) &GetRegisteredOpNum;
    } else if (std::string(func_name) == "GetOpImplFunctions") {
      if (dlsys_get_impl_func_fail) {
        dlsys_get_impl_func_fail = false;
        return nullptr;
      }
      return (void *) &GetOpImplFunctions;
    }
    return nullptr;
  }
  void *DlOpen(const char *fileName, int32_t mode) override {
    if (mock_handle == nullptr) {
      return nullptr;
    }
    return (void *) mock_handle;
  }
  int32_t DlClose(void *handle) override {
    return 0L;
  }
};
}

class OpImplRegistryHolderManagerUT : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {
    gert::OpImplRegistryHolderManager::GetInstance().ClearOpImplRegistries();
    ge::MmpaStub::GetInstance().Reset();
  }
};

TEST_F(OpImplRegistryHolderManagerUT, OmOpImplRegistryHolder_LoadSo_Succeed) {
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  mock_handle = (void *) 0xffffffff;
  g_impl_num = 1;

  std::string so_name("libopmaster.so");
  std::string vendor_name("MDC");
  ge::OpSoBinPtr so_bin_ptr = CreateSoBinPtr(so_name, vendor_name);
  gert::OmOpImplRegistryHolder om_op_impl_registry_holder;
  auto ret = om_op_impl_registry_holder.LoadSo(so_bin_ptr);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(om_op_impl_registry_holder.GetTypesToImpl().size(), 1);
}

TEST_F(OpImplRegistryHolderManagerUT, OmOpImplRegistryHolder_LoadSo_DlopenFailed) {
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  mock_handle = nullptr;
  g_impl_num = 1;

  std::string so_name("libopmaster.so");
  std::string vendor_name("MDC");
  ge::OpSoBinPtr so_bin_ptr = CreateSoBinPtr(so_name, vendor_name);
  gert::OmOpImplRegistryHolder om_op_impl_registry_holder;
  auto ret = om_op_impl_registry_holder.LoadSo(so_bin_ptr);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(om_op_impl_registry_holder.GetTypesToImpl().size(), 0);
}

TEST_F(OpImplRegistryHolderManagerUT, OmOpImplRegistryHolder_LoadSo_DlsymFailed) {
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  mock_handle = (void *) 0x7000;
  g_impl_num = 1;

  std::string so_name("libopmaster.so");
  std::string vendor_name("MDC");
  ge::OpSoBinPtr so_bin_ptr = CreateSoBinPtr(so_name, vendor_name);
  gert::OmOpImplRegistryHolder om_op_impl_registry_holder;
  auto ret = om_op_impl_registry_holder.LoadSo(so_bin_ptr);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(om_op_impl_registry_holder.GetTypesToImpl().size(), 0);
}

TEST_F(OpImplRegistryHolderManagerUT, OmOpImplRegistryHolder_LoadSo_GetImplFunc_fail) {
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  mock_handle = (void *) 0xffffffff;
  g_impl_num = 1;
  dlsys_get_impl_func_fail = true;

  std::string so_name("libopmaster.so");
  std::string vendor_name("MDC");
  ge::OpSoBinPtr so_bin_ptr = CreateSoBinPtr(so_name, vendor_name);
  gert::OmOpImplRegistryHolder om_op_impl_registry_holder;
  auto ret = om_op_impl_registry_holder.LoadSo(so_bin_ptr);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(om_op_impl_registry_holder.GetTypesToImpl().size(), 0);
}

TEST_F(OpImplRegistryHolderManagerUT, GetOrCreateOpImplRegistryHolder_Succeed_HaveRegisted) {
  std::string so_data("libopmaster.so");
  std::string so_name("libopmaster.so");
  auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
  gert::OpImplRegistryHolderManager::GetInstance().AddRegistry(so_data, registry_holder);
  ge::SoInOmInfo so_info;
  so_info.cpu_info = "x86";
  so_info.os_info = "Linux";
  so_info.opp_version = "1.84";
  auto registry_holder1 = gert::OpImplRegistryHolderManager::GetInstance().GetOrCreateOpImplRegistryHolder(so_data,
                                                                                                          so_name,
                                                                                                          so_info,
                                                                                                          nullptr);
  EXPECT_NE(registry_holder1, nullptr);
  EXPECT_EQ(gert::OpImplRegistryHolderManager::GetInstance().GetOpImplRegistrySize(), 1);
}

TEST_F(OpImplRegistryHolderManagerUT, GetOrCreateOpImplRegistryHolder_Succeed_NoRegisted) {
  std::string so_data("libopmaster.so");
  std::string so_name("libopmaster.so");
  std::string vendor_name("MDC");
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  mock_handle = (void *) 0xffffffff;
  g_impl_num = 1;
  ge::OpSoBinPtr so_bin_ptr = CreateSoBinPtr(so_name, vendor_name);
  ge::SoInOmInfo so_info;
  so_info.cpu_info = "x86";
  so_info.os_info = "Linux";
  so_info.opp_version = "1.84";
  auto create_func = [&so_bin_ptr]() ->gert::OpImplRegistryHolderPtr {
    auto om_registry_holder = std::make_shared<gert::OmOpImplRegistryHolder>();
    if (om_registry_holder == nullptr) {
      GELOGE(ge::FAILED, "make_shared om op impl registry holder failed");
      return nullptr;
    }
    if((om_registry_holder->LoadSo(so_bin_ptr)) != ge::GRAPH_SUCCESS) {
      GELOGE(ge::FAILED, "om registry holder load so failed");
      return nullptr;
    }
    return om_registry_holder;
  };
  EXPECT_EQ(gert::OpImplRegistryHolderManager::GetInstance().GetOpImplRegistrySize(), 0);
  auto registry_holder1 = gert::OpImplRegistryHolderManager::GetInstance().GetOrCreateOpImplRegistryHolder(so_data,
                                                                                                           so_name,
                                                                                                           so_info,
                                                                                                           create_func);
  EXPECT_NE(registry_holder1, nullptr);
  EXPECT_EQ(gert::OpImplRegistryHolderManager::GetInstance().GetOpImplRegistrySize(), 1);
}

TEST_F(OpImplRegistryHolderManagerUT, GetOrCreateOpImplRegistryHolder_Failed_CreateFuncFailed) {
  std::string so_data("libopmaster.so");
  std::string so_name("libopmaster.so");
  std::string vendor_name("MDC");
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  mock_handle = (void *) 0xffffffff;
  g_impl_num = 1;
  ge::OpSoBinPtr so_bin_ptr = CreateSoBinPtr(so_name, vendor_name);
  ge::SoInOmInfo so_info;
  so_info.cpu_info = "x86";
  so_info.os_info = "Linux";
  so_info.opp_version = "1.84";
  auto create_func = [&so_bin_ptr]() ->gert::OpImplRegistryHolderPtr {
    return nullptr;
  };
  EXPECT_EQ(gert::OpImplRegistryHolderManager::GetInstance().GetOpImplRegistrySize(), 0);
  auto registry_holder1 = gert::OpImplRegistryHolderManager::GetInstance().GetOrCreateOpImplRegistryHolder(so_data,
                                                                                                           so_name,
                                                                                                           so_info,
                                                                                                           create_func);
  EXPECT_EQ(registry_holder1, nullptr);
  EXPECT_EQ(gert::OpImplRegistryHolderManager::GetInstance().GetOpImplRegistrySize(), 0);
}

TEST_F(OpImplRegistryHolderManagerUT, GetOrCreateOpImplRegistryHolder_Failed_CreateFuncNull) {
  std::string so_data("libopmaster.so");
  std::string so_name("libopmaster.so");
  std::string vendor_name("MDC");
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  mock_handle = (void *) 0xffffffff;
  g_impl_num = 1;
  ge::OpSoBinPtr so_bin_ptr = CreateSoBinPtr(so_name, vendor_name);
  ge::SoInOmInfo so_info;
  so_info.cpu_info = "x86";
  so_info.os_info = "Linux";
  so_info.opp_version = "1.84";
  EXPECT_EQ(gert::OpImplRegistryHolderManager::GetInstance().GetOpImplRegistrySize(), 0);
  auto registry_holder1 = gert::OpImplRegistryHolderManager::GetInstance().GetOrCreateOpImplRegistryHolder(so_data,
                                                                                                           so_name,
                                                                                                           so_info,
                                                                                                           nullptr);
  EXPECT_EQ(registry_holder1, nullptr);
  EXPECT_EQ(gert::OpImplRegistryHolderManager::GetInstance().GetOpImplRegistrySize(), 0);
}

TEST_F(OpImplRegistryHolderManagerUT, OpImplRegistryManager_UpdateOpImplRegistries_Succeed) {
  std::string so_data("libopmaster.so");
  {
    auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
    gert::OpImplRegistryHolderManager::GetInstance().AddRegistry(so_data, registry_holder);
    auto tmp_registry_holder = gert::OpImplRegistryHolderManager::GetInstance().GetOpImplRegistryHolder(so_data);
    EXPECT_NE(tmp_registry_holder, nullptr);
    EXPECT_EQ(gert::OpImplRegistryHolderManager::GetInstance().GetOpImplRegistrySize(), 1);
  }
  auto tmp_registry_holder = gert::OpImplRegistryHolderManager::GetInstance().GetOpImplRegistryHolder(so_data);
  //  EXPECT_EQ(tmp_registry_holder, nullptr); // 最终方案适配后使用此校验
  EXPECT_NE(tmp_registry_holder, nullptr);
  EXPECT_EQ(gert::OpImplRegistryHolderManager::GetInstance().GetOpImplRegistrySize(), 1);

  gert::OpImplRegistryHolderManager::GetInstance().UpdateOpImplRegistries();
  tmp_registry_holder = gert::OpImplRegistryHolderManager::GetInstance().GetOpImplRegistryHolder(so_data);
  //  EXPECT_EQ(tmp_registry_holder, nullptr);  // 最终方案适配后使用此校验
  //  EXPECT_EQ(gert::OpImplRegistryHolderManager::GetInstance().GetOpImplRegistrySize(), 0); // 最终方案适配后使用此校验
  EXPECT_NE(tmp_registry_holder, nullptr);
  EXPECT_EQ(gert::OpImplRegistryHolderManager::GetInstance().GetOpImplRegistrySize(), 1);
}

TEST_F(OpImplRegistryHolderManagerUT, OpImplRegistryManager_UpdateOpImplRegistries_Succeed2) {
  std::string so_data("libopmaster.so");

  gert::OpImplRegistryHolderManager::GetInstance().AddRegistry(so_data, nullptr);
  auto tmp_registry_holder = gert::OpImplRegistryHolderManager::GetInstance().GetOpImplRegistryHolder(so_data);
  EXPECT_EQ(tmp_registry_holder, nullptr);
  EXPECT_EQ(gert::OpImplRegistryHolderManager::GetInstance().GetOpImplRegistrySize(), 1);

  gert::OpImplRegistryHolderManager::GetInstance().UpdateOpImplRegistries();
  tmp_registry_holder = gert::OpImplRegistryHolderManager::GetInstance().GetOpImplRegistryHolder(so_data);
  EXPECT_EQ(tmp_registry_holder, nullptr);
  EXPECT_EQ(gert::OpImplRegistryHolderManager::GetInstance().GetOpImplRegistrySize(), 0);
}

TEST_F(OpImplRegistryHolderManagerUT, OmOpImplRegistryHolder_LoadSo_HomeEnv_Null_Invalid) {
  ge::char_t home_env[MMPA_MAX_PATH] = {'\0'};
  mmGetEnv(kHomeEnvName, home_env, MMPA_MAX_PATH);
  mmSetEnv(kHomeEnvName, "", 1);
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  mock_handle = (void *) 0xffffffff;
  g_impl_num = 1;

  std::string so_name("libopmaster.so");
  std::string vendor_name("MDC");
  ge::OpSoBinPtr so_bin_ptr = CreateSoBinPtr(so_name, vendor_name);
  gert::OmOpImplRegistryHolder om_op_impl_registry_holder;
  auto ret = om_op_impl_registry_holder.LoadSo(so_bin_ptr);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  mmSetEnv(kHomeEnvName, home_env, MMPA_MAX_PATH);
}

TEST_F(OpImplRegistryHolderManagerUT, OmOpImplRegistryHolder_LoadSo_HomeEnv_Invalid) {
  ge::char_t home_env[MMPA_MAX_PATH] = {'\0'};
  mmGetEnv(kHomeEnvName, home_env, MMPA_MAX_PATH);
  mmSetEnv("HOME", "*&()", 1);
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  mock_handle = (void *) 0xffffffff;
  g_impl_num = 1;

  std::string so_name("libopmaster.so");
  std::string vendor_name("MDC");
  ge::OpSoBinPtr so_bin_ptr = CreateSoBinPtr(so_name, vendor_name);
  gert::OmOpImplRegistryHolder om_op_impl_registry_holder;
  auto ret = om_op_impl_registry_holder.LoadSo(so_bin_ptr);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  mmSetEnv(kHomeEnvName, home_env, MMPA_MAX_PATH);
}

TEST_F(OpImplRegistryHolderManagerUT, OmOpImplRegistryHolder_LoadSo_AscendWorkPathEnv_Invalid) {
  mmSetEnv("ASCEND_WORK_PATH", "", 1);
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  mock_handle = (void *) 0xffffffff;
  g_impl_num = 1;

  std::string so_name("libopmaster.so");
  std::string vendor_name("MDC");
  ge::OpSoBinPtr so_bin_ptr = CreateSoBinPtr(so_name, vendor_name);
  gert::OmOpImplRegistryHolder om_op_impl_registry_holder;
  auto ret = om_op_impl_registry_holder.LoadSo(so_bin_ptr);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  unsetenv("ASCEND_WORK_PATH");
}

TEST_F(OpImplRegistryHolderManagerUT, OmOpImplRegistryHolder_LoadSo_AscendWorkPathEnv_Valid) {
  ge::char_t current_path[MMPA_MAX_PATH] = {'\0'};
  getcwd(current_path, MMPA_MAX_PATH);
  mmSetEnv("ASCEND_WORK_PATH", current_path, 1);
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  mock_handle = (void *) 0xffffffff;
  g_impl_num = 1;

  std::string so_name("libopmaster.so");
  std::string vendor_name("MDC");
  ge::OpSoBinPtr so_bin_ptr = CreateSoBinPtr(so_name, vendor_name);
  gert::OmOpImplRegistryHolder om_op_impl_registry_holder;
  auto ret = om_op_impl_registry_holder.LoadSo(so_bin_ptr);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  std::string opp_dir = current_path;
  opp_dir += "/.ascend_temp/.om_exe_data";
  auto real_opp_dir = ge::RealPath(opp_dir.c_str());
  EXPECT_EQ(real_opp_dir, opp_dir);
  unsetenv("ASCEND_WORK_PATH");
}

TEST_F(OpImplRegistryHolderManagerUT, OmOpImplRegistryHolder_LoadSo_CreateAscendWorkPath_Success) {
  std::string work_path = "./test_work_path";
  mmSetEnv("ASCEND_WORK_PATH", work_path.c_str(), 1);
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  mock_handle = (void *) 0xffffffff;
  g_impl_num = 1;

  std::string so_name("libopmaster.so");
  std::string vendor_name("MDC");
  ge::OpSoBinPtr so_bin_ptr = CreateSoBinPtr(so_name, vendor_name);
  gert::OmOpImplRegistryHolder om_op_impl_registry_holder;
  auto ret = om_op_impl_registry_holder.LoadSo(so_bin_ptr);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  std::string real_work_path = ge::RealPath(work_path.c_str());
  EXPECT_EQ(mmAccess(real_work_path.c_str()), EN_OK);
  std::string opp_dir = real_work_path + "/.ascend_temp/.om_exe_data";
  auto real_opp_dir = ge::RealPath(opp_dir.c_str());
  EXPECT_EQ(real_opp_dir, opp_dir);
  unsetenv("ASCEND_WORK_PATH");
}
}  // namespace gert_test

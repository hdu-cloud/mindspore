// Copyright (c) 2021 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#define private public
#define protected public
#include "common/plugin/plugin_manager.h"

#include <iostream>

using namespace testing;
using namespace std;

namespace ge {
namespace {
const char *const kEnvName = "ASCEND_OPP_PATH";
const char *const kEnvNameCustom = "ASCEND_CUSTOM_OPP_PATH";

void WriteRequiredVersion(const std::string &version, std::string &path, std::string &path1) {
  path ="./runtime";
  system(("mkdir -p " + path).c_str());
  auto version_path = path + "/version.info";
  system(("touch " + version_path).c_str());
  system(("echo 'required_opp_abi_version=" + version +"' > " + version_path).c_str());

  path1 ="./../runtime";
  system(("mkdir -p " + path1).c_str());
  auto version_path1 = path1 + "/version.info";
  system(("touch " + version_path1).c_str());
  system(("echo 'required_opp_abi_version=" + version +"' > " + version_path1).c_str());
}
}
class UtestPluginManager : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(UtestPluginManager, test_plugin_manager_load) {
  PluginManager manager;
  auto test_so_name = "test_fail.so";
  manager.handles_[test_so_name] = nullptr;
  manager.ClearHandles_();
  ASSERT_EQ(manager.handles_.size(), 0);
  EXPECT_EQ(manager.LoadSo("./", {}), SUCCESS);
  int64_t file_size = 1;
  EXPECT_EQ(manager.ValidateSo("./", 1, file_size), SUCCESS);

  std::string path = GetModelPath();
  std::vector<std::string> pathList;
  pathList.push_back(path);

  std::vector<std::string> funcChkList;
  const std::string file_name = "libcce.so";
  funcChkList.push_back(file_name);
  funcChkList.push_back("invalid_func");

  system(("touch " + path + file_name).c_str());

  EXPECT_EQ(manager.Load("", pathList), SUCCESS);
  EXPECT_EQ(manager.Load(path, {}), SUCCESS);
  EXPECT_EQ(manager.LoadSo(path, {}), SUCCESS);
  EXPECT_EQ(manager.LoadSo(path, funcChkList), SUCCESS);

  path += file_name + "/";
  pathList.push_back(path);
  EXPECT_EQ(manager.LoadSo(path, pathList), SUCCESS);

  unlink(path.c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_load_so_fail) {
  PluginManager manager;
  std::vector<std::string> funcs;
  const std::string path = "./tests/depends/mmpa/libmmpa.so";
  funcs.push_back("invalid_func");
  EXPECT_EQ(manager.LoadSo(path, funcs), SUCCESS);
  EXPECT_EQ(manager.Load("./tests/depends/mmpa/", funcs), SUCCESS);
}

TEST_F(UtestPluginManager, test_plugin_manager_getopp_plugin_vendors_01) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());

  std::vector<std::string> vendors;
  Status ret = PluginManager::GetOppPluginVendors(path_config, vendors);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(vendors[0], "customize");
  EXPECT_EQ(vendors[1], "mdc");
  EXPECT_EQ(vendors[2], "lhisi");
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_getopp_plugin_vendors_02) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo '' > " + path_config).c_str());

  std::vector<std::string> vendors;
  Status ret = PluginManager::GetOppPluginVendors(path_config, vendors);
  EXPECT_NE(ret, SUCCESS);
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_getopp_plugin_vendors_03) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority' > " + path_config).c_str());

  std::vector<std::string> vendors;
  Status ret = PluginManager::GetOppPluginVendors(path_config, vendors);
  EXPECT_NE(ret, SUCCESS);
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_getopp_plugin_vendors_04) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("rm -rf " + path_config).c_str());

  std::vector<std::string> vendors;
  Status ret = PluginManager::GetOppPluginVendors(path_config, vendors);
  EXPECT_NE(ret, SUCCESS);
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_getopp_plugin_vendors_05) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo ' load_priority = customize , mdc , lhisi ' > " + path_config).c_str());

  std::vector<std::string> vendors;
  Status ret = PluginManager::GetOppPluginVendors(path_config, vendors);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(vendors[0], "customize");
  EXPECT_EQ(vendors[1], "mdc");
  EXPECT_EQ(vendors[2], "lhisi");
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetOpsProtoPath_01) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  system(("rm -rf " + path_builtin).c_str());

  std::string opsproto_path;
  Status ret = PluginManager::GetOpsProtoPath(opsproto_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(opsproto_path,
      opp_path + "op_proto/custom/:" + opp_path + "op_proto/built-in/"
  );
}

TEST_F(UtestPluginManager, test_plugin_manager_GetOpsProtoPath_02) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());

  std::string opsproto_path;
  Status ret = PluginManager::GetOpsProtoPath(opsproto_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(opsproto_path,
    path_vendors + "/customize/op_proto/:" +
    path_vendors + "/mdc/op_proto/:" +
    path_vendors + "/lhisi/op_proto/:" +
    opp_path + "built-in/op_proto/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetOpsProtoPath_03) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority' > " + path_config).c_str());

  std::string opsproto_path;
  Status ret = PluginManager::GetOpsProtoPath(opsproto_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(opsproto_path,
    opp_path + "op_proto/custom/:" +
    opp_path + "built-in/op_proto/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetOpsProtoPath_04) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path = opp_path + "custom_opp_path";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, custom_opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path + "/op_proto").c_str());

  std::string opsproto_path;
  Status ret = PluginManager::GetOpsProtoPath(opsproto_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(opsproto_path,
    custom_opp_path + "/op_proto/:" +
    path_vendors + "/customize/op_proto/:" +
    path_vendors + "/mdc/op_proto/:" +
    path_vendors + "/lhisi/op_proto/:" +
    opp_path + "built-in/op_proto/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetOpsProtoPath_05) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path = opp_path + "custom_opp_path";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, "", 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path + "/op_proto").c_str());

  std::string opsproto_path;
  Status ret = PluginManager::GetOpsProtoPath(opsproto_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(opsproto_path,
    path_vendors + "/customize/op_proto/:" +
    path_vendors + "/mdc/op_proto/:" +
    path_vendors + "/lhisi/op_proto/:" +
    opp_path + "built-in/op_proto/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetOpsProtoPath_06) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path_01 = opp_path + "custom_opp_path_01";
  std::string custom_opp_path_02 = opp_path + "custom_opp_path_02";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, (custom_opp_path_01 + ":" + custom_opp_path_02).c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path_01 + "/op_proto").c_str());
  system(("mkdir -p " + custom_opp_path_02 + "/op_proto").c_str());

  std::string opsproto_path;
  Status ret = PluginManager::GetOpsProtoPath(opsproto_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(opsproto_path,
    custom_opp_path_01 + "/op_proto/:" +
    custom_opp_path_02 + "/op_proto/:" +
    path_vendors + "/customize/op_proto/:" +
    path_vendors + "/mdc/op_proto/:" +
    path_vendors + "/lhisi/op_proto/:" +
    opp_path + "built-in/op_proto/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetOpsProtoPath_07) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path = opp_path + "custom_opp_path";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, custom_opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("mkdir -p " + custom_opp_path + "/op_proto").c_str());

  std::string opsproto_path;
  Status ret = PluginManager::GetOpsProtoPath(opsproto_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(opsproto_path,
    custom_opp_path + "/op_proto/:" +
    opp_path + "op_proto/custom/:" +
    opp_path + "built-in/op_proto/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetOpsProtoPath_08) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path_01 = opp_path + "custom_opp_path_01";
  std::string custom_opp_path_invalid_01 = opp_path + "custom_opp_path_invalid_01";
  std::string custom_opp_path_empty = "";
  std::string custom_opp_path_02 = opp_path + "custom_opp_path_02";
  std::string custom_opp_path_invalid_02 = opp_path + "custom_opp_path_invalid_02";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, (custom_opp_path_01 + ":" +
                            custom_opp_path_invalid_01 + ":" +
                            custom_opp_path_empty + ":" +
                            custom_opp_path_02 + ":" +
                            custom_opp_path_invalid_02
                           ).c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path_01 + "/op_proto").c_str());
  system(("mkdir -p " + custom_opp_path_02 + "/op_proto").c_str());

  std::string opsproto_path;
  Status ret = PluginManager::GetOpsProtoPath(opsproto_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(opsproto_path,
    custom_opp_path_01 + "/op_proto/:" +
    custom_opp_path_02 + "/op_proto/:" +
    path_vendors + "/customize/op_proto/:" +
    path_vendors + "/mdc/op_proto/:" +
    path_vendors + "/lhisi/op_proto/:" +
    opp_path + "built-in/op_proto/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetOpsProtoPath_09) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path_01 = opp_path + "custom_opp_path_01";
  std::string custom_opp_path_02 = opp_path + "custom_opp_path_02";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, (custom_opp_path_01 + ":" +
                            custom_opp_path_02 + ":" +
                            custom_opp_path_01 + ":" +
                            custom_opp_path_02
                           ).c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path_01 + "/op_proto").c_str());
  system(("mkdir -p " + custom_opp_path_02 + "/op_proto").c_str());

  std::string opsproto_path;
  Status ret = PluginManager::GetOpsProtoPath(opsproto_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(opsproto_path,
    custom_opp_path_01 + "/op_proto/:" +
    custom_opp_path_02 + "/op_proto/:" +
    custom_opp_path_01 + "/op_proto/:" +
    custom_opp_path_02 + "/op_proto/:" +
    path_vendors + "/customize/op_proto/:" +
    path_vendors + "/mdc/op_proto/:" +
    path_vendors + "/lhisi/op_proto/:" +
    opp_path + "built-in/op_proto/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetOpTilingPath_01) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  system(("rm -rf " + path_builtin).c_str());

  std::string op_tiling_path;
  Status ret = PluginManager::GetOpTilingPath(op_tiling_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_tiling_path,
      opp_path + "op_impl/built-in/ai_core/tbe/:" + opp_path + "op_impl/custom/ai_core/tbe/"
  );
}

TEST_F(UtestPluginManager, test_plugin_manager_GetOpTilingPath_02) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());

  std::string op_tiling_path;
  Status ret = PluginManager::GetOpTilingPath(op_tiling_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_tiling_path,
    opp_path + "built-in/op_impl/ai_core/tbe/:" +
    path_vendors + "/lhisi/op_impl/ai_core/tbe/:" +
    path_vendors + "/mdc/op_impl/ai_core/tbe/:" +
    path_vendors + "/customize/op_impl/ai_core/tbe/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetOpTilingPath_03) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority' > " + path_config).c_str());

  std::string op_tiling_path;
  Status ret = PluginManager::GetOpTilingPath(op_tiling_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_tiling_path,
    opp_path + "built-in/op_impl/ai_core/tbe/:" +
    opp_path + "op_impl/custom/ai_core/tbe/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetOpTilingPath_04) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path = opp_path + "custom_opp_path";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, custom_opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path + "/op_impl/ai_core/tbe").c_str());

  std::string op_tiling_path;
  Status ret = PluginManager::GetOpTilingPath(op_tiling_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_tiling_path,
    opp_path + "built-in/op_impl/ai_core/tbe/:" +
    path_vendors + "/lhisi/op_impl/ai_core/tbe/:" +
    path_vendors + "/mdc/op_impl/ai_core/tbe/:" +
    path_vendors + "/customize/op_impl/ai_core/tbe/:" +
    custom_opp_path + "/op_impl/ai_core/tbe/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetOpTilingPath_05) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path = opp_path + "custom_opp_path";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, "", 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path + "/op_impl/ai_core/tbe").c_str());

  std::string op_tiling_path;
  Status ret = PluginManager::GetOpTilingPath(op_tiling_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_tiling_path,
    opp_path + "built-in/op_impl/ai_core/tbe/:" +
    path_vendors + "/lhisi/op_impl/ai_core/tbe/:" +
    path_vendors + "/mdc/op_impl/ai_core/tbe/:" +
    path_vendors + "/customize/op_impl/ai_core/tbe/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetOpTilingPath_06) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path_01 = opp_path + "custom_opp_path_01";
  std::string custom_opp_path_02 = opp_path + "custom_opp_path_02";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, (custom_opp_path_01 + ":" + custom_opp_path_02).c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path_01 + "/op_impl/ai_core/tbe").c_str());
  system(("mkdir -p " + custom_opp_path_02 + "/op_impl/ai_core/tbe").c_str());

  std::string op_tiling_path;
  Status ret = PluginManager::GetOpTilingPath(op_tiling_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_tiling_path,
    opp_path + "built-in/op_impl/ai_core/tbe/:" +
    path_vendors + "/lhisi/op_impl/ai_core/tbe/:" +
    path_vendors + "/mdc/op_impl/ai_core/tbe/:" +
    path_vendors + "/customize/op_impl/ai_core/tbe/:" +
    custom_opp_path_02 + "/op_impl/ai_core/tbe/:" +
    custom_opp_path_01 + "/op_impl/ai_core/tbe/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetOpTilingPath_07) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path = opp_path + "custom_opp_path";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, custom_opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("mkdir -p " + custom_opp_path + "/op_impl/ai_core/tbe").c_str());

  std::string op_tiling_path;
  Status ret = PluginManager::GetOpTilingPath(op_tiling_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_tiling_path,
    opp_path + "built-in/op_impl/ai_core/tbe/:" +
    opp_path + "op_impl/custom/ai_core/tbe/:" +
    custom_opp_path + "/op_impl/ai_core/tbe/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetOpTilingPath_08) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path_01 = opp_path + "custom_opp_path_01";
  std::string custom_opp_path_invalid_01 = opp_path + "custom_opp_path_invalid_01";
  std::string custom_opp_path_empty = "";
  std::string custom_opp_path_02 = opp_path + "custom_opp_path_02";
  std::string custom_opp_path_invalid_02 = opp_path + "custom_opp_path_invalid_02";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, (custom_opp_path_01 + ":" +
                            custom_opp_path_invalid_01 + ":" +
                            custom_opp_path_empty + ":" +
                            custom_opp_path_02 + ":" +
                            custom_opp_path_invalid_02
                           ).c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path_01 + "/op_impl/ai_core/tbe").c_str());
  system(("mkdir -p " + custom_opp_path_02 + "/op_impl/ai_core/tbe").c_str());

  std::string op_tiling_path;
  Status ret = PluginManager::GetOpTilingPath(op_tiling_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_tiling_path,
    opp_path + "built-in/op_impl/ai_core/tbe/:" +
    path_vendors + "/lhisi/op_impl/ai_core/tbe/:" +
    path_vendors + "/mdc/op_impl/ai_core/tbe/:" +
    path_vendors + "/customize/op_impl/ai_core/tbe/:" +
    custom_opp_path_02 + "/op_impl/ai_core/tbe/:" +
    custom_opp_path_01 + "/op_impl/ai_core/tbe/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetOpTilingPath_09) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path_01 = opp_path + "custom_opp_path_01";
  std::string custom_opp_path_02 = opp_path + "custom_opp_path_02";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, (custom_opp_path_01 + ":" +
                            custom_opp_path_02 + ":" +
                            custom_opp_path_01 + ":" +
                            custom_opp_path_02
                           ).c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path_01 + "/op_impl/ai_core/tbe").c_str());
  system(("mkdir -p " + custom_opp_path_02 + "/op_impl/ai_core/tbe").c_str());

  std::string op_tiling_path;
  Status ret = PluginManager::GetOpTilingPath(op_tiling_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_tiling_path,
    opp_path + "built-in/op_impl/ai_core/tbe/:" +
    path_vendors + "/lhisi/op_impl/ai_core/tbe/:" +
    path_vendors + "/mdc/op_impl/ai_core/tbe/:" +
    path_vendors + "/customize/op_impl/ai_core/tbe/:" +
    custom_opp_path_02 + "/op_impl/ai_core/tbe/:" +
    custom_opp_path_01 + "/op_impl/ai_core/tbe/:" +
    custom_opp_path_02 + "/op_impl/ai_core/tbe/:" +
    custom_opp_path_01 + "/op_impl/ai_core/tbe/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetCustomOpPath_01) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  system(("rm -rf " + path_builtin).c_str());

  std::string customop_path;
  Status ret = PluginManager::GetCustomOpPath("tensorflow", customop_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(customop_path,
    opp_path + "framework/custom/:" + opp_path + "framework/built-in/tensorflow/"
  );
}

TEST_F(UtestPluginManager, test_plugin_manager_GetCustomOpPath_02) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());

  std::string customop_path;
  Status ret = PluginManager::GetCustomOpPath("tensorflow", customop_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(customop_path,
    path_vendors + "/customize/framework/:" +
    path_vendors + "/mdc/framework/:" +
    path_vendors + "/lhisi/framework/:" +
    opp_path + "built-in/framework/tensorflow/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetCustomOpPath_03) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("rm -rf " + path_config).c_str());

  std::string customop_path;
  Status ret = PluginManager::GetCustomOpPath("tensorflow", customop_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(customop_path,
    opp_path + "framework/custom/:" +
    opp_path + "built-in/framework/tensorflow/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetCustomOpPath_04) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path = opp_path + "custom_opp_path";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, custom_opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path + "/framework").c_str());

  std::string customop_path;
  Status ret = PluginManager::GetCustomOpPath("tensorflow", customop_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(customop_path,
    custom_opp_path + "/framework/:" +
    path_vendors + "/customize/framework/:" +
    path_vendors + "/mdc/framework/:" +
    path_vendors + "/lhisi/framework/:" +
    opp_path + "built-in/framework/tensorflow/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetCustomOpPath_05) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path = opp_path + "custom_opp_path";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, "", 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path + "/framework").c_str());

  std::string customop_path;
  Status ret = PluginManager::GetCustomOpPath("tensorflow", customop_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(customop_path,
    path_vendors + "/customize/framework/:" +
    path_vendors + "/mdc/framework/:" +
    path_vendors + "/lhisi/framework/:" +
    opp_path + "built-in/framework/tensorflow/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetCustomOpPath_06) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path_01 = opp_path + "custom_opp_path_01";
  std::string custom_opp_path_02 = opp_path + "custom_opp_path_02";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, (custom_opp_path_01 + ":" + custom_opp_path_02).c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path_01 + "/framework").c_str());
  system(("mkdir -p " + custom_opp_path_02 + "/framework").c_str());

  std::string customop_path;
  Status ret = PluginManager::GetCustomOpPath("tensorflow", customop_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(customop_path,
    custom_opp_path_01 + "/framework/:" +
    custom_opp_path_02 + "/framework/:" +
    path_vendors + "/customize/framework/:" +
    path_vendors + "/mdc/framework/:" +
    path_vendors + "/lhisi/framework/:" +
    opp_path + "built-in/framework/tensorflow/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetCustomOpPath_07) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path = opp_path + "custom_opp_path";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, custom_opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("mkdir -p " + custom_opp_path + "/framework").c_str());

  std::string customop_path;
  Status ret = PluginManager::GetCustomOpPath("tensorflow", customop_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(customop_path,
    custom_opp_path + "/framework/:" +
    opp_path + "framework/custom/:" +
    opp_path + "built-in/framework/tensorflow/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetCustomOpPath_08) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path_01 = opp_path + "custom_opp_path_01";
  std::string custom_opp_path_invalid_01 = opp_path + "custom_opp_path_invalid_01";
  std::string custom_opp_path_empty = "";
  std::string custom_opp_path_02 = opp_path + "custom_opp_path_02";
  std::string custom_opp_path_invalid_02 = opp_path + "custom_opp_path_invalid_02";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, (custom_opp_path_01 + ":" +
                            custom_opp_path_invalid_01 + ":" +
                            custom_opp_path_empty + ":" +
                            custom_opp_path_02 + ":" +
                            custom_opp_path_invalid_02
                           ).c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path_01 + "/framework").c_str());
  system(("mkdir -p " + custom_opp_path_02 + "/framework").c_str());

  std::string customop_path;
  Status ret = PluginManager::GetCustomOpPath("tensorflow", customop_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(customop_path,
    custom_opp_path_01 + "/framework/:" +
    custom_opp_path_02 + "/framework/:" +
    path_vendors + "/customize/framework/:" +
    path_vendors + "/mdc/framework/:" +
    path_vendors + "/lhisi/framework/:" +
    opp_path + "built-in/framework/tensorflow/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetCustomOpPath_09) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path_01 = opp_path + "custom_opp_path_01";
  std::string custom_opp_path_02 = opp_path + "custom_opp_path_02";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, (custom_opp_path_01 + ":" +
                            custom_opp_path_02 + ":" +
                            custom_opp_path_01 + ":" +
                            custom_opp_path_02
                           ).c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path_01 + "/framework").c_str());
  system(("mkdir -p " + custom_opp_path_02 + "/framework").c_str());

  std::string customop_path;
  Status ret = PluginManager::GetCustomOpPath("tensorflow", customop_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(customop_path,
    custom_opp_path_01 + "/framework/:" +
    custom_opp_path_02 + "/framework/:" +
    custom_opp_path_01 + "/framework/:" +
    custom_opp_path_02 + "/framework/:" +
    path_vendors + "/customize/framework/:" +
    path_vendors + "/mdc/framework/:" +
    path_vendors + "/lhisi/framework/:" +
    opp_path + "built-in/framework/tensorflow/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetCustomCaffeProtoPath_01) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  system(("rm -rf " + path_builtin).c_str());
  system(("rm -rf " + path_vendors).c_str());

  std::string customcaffe_path;
  Status ret = PluginManager::GetCustomCaffeProtoPath(customcaffe_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(customcaffe_path,
    opp_path + "framework/custom/caffe/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetCustomCaffeProtoPath_02) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());

  std::string customcaffe_path;
  Status ret = PluginManager::GetCustomCaffeProtoPath(customcaffe_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(customcaffe_path,
    path_vendors + "/customize/framework/caffe/:" +
    path_vendors + "/mdc/framework/caffe/:" +
    path_vendors + "/lhisi/framework/caffe/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetCustomCaffeProtoPath_03) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("rm -rf " + path_config).c_str());

  std::string customcaffe_path;
  Status ret = PluginManager::GetCustomCaffeProtoPath(customcaffe_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(customcaffe_path,
    opp_path + "framework/custom/caffe/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetCustomCaffeProtoPath_04) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path = opp_path + "custom_opp_path";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, custom_opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path + "/framework/caffe").c_str());

  std::string customcaffe_path;
  Status ret = PluginManager::GetCustomCaffeProtoPath(customcaffe_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(customcaffe_path,
    custom_opp_path + "/framework/caffe/:" +
    path_vendors + "/customize/framework/caffe/:" +
    path_vendors + "/mdc/framework/caffe/:" +
    path_vendors + "/lhisi/framework/caffe/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetCustomCaffeProtoPath_05) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path = opp_path + "custom_opp_path";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, "", 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path + "/framework/caffe").c_str());

  std::string customcaffe_path;
  Status ret = PluginManager::GetCustomCaffeProtoPath(customcaffe_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(customcaffe_path,
    path_vendors + "/customize/framework/caffe/:" +
    path_vendors + "/mdc/framework/caffe/:" +
    path_vendors + "/lhisi/framework/caffe/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetCustomCaffeProtoPath_06) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path_01 = opp_path + "custom_opp_path_01";
  std::string custom_opp_path_02 = opp_path + "custom_opp_path_02";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, (custom_opp_path_01 + ":" + custom_opp_path_02).c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path_01 + "/framework/caffe").c_str());
  system(("mkdir -p " + custom_opp_path_02 + "/framework/caffe").c_str());

  std::string customcaffe_path;
  Status ret = PluginManager::GetCustomCaffeProtoPath(customcaffe_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(customcaffe_path,
    custom_opp_path_01 + "/framework/caffe/:" +
    custom_opp_path_02 + "/framework/caffe/:" +
    path_vendors + "/customize/framework/caffe/:" +
    path_vendors + "/mdc/framework/caffe/:" +
    path_vendors + "/lhisi/framework/caffe/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetCustomCaffeProtoPath_07) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path = opp_path + "custom_opp_path";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, custom_opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("mkdir -p " + custom_opp_path + "/framework/caffe").c_str());

  std::string customcaffe_path;
  Status ret = PluginManager::GetCustomCaffeProtoPath(customcaffe_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(customcaffe_path,
    custom_opp_path + "/framework/caffe/:" +
    opp_path + "framework/custom/caffe/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetCustomCaffeProtoPath_08) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path_01 = opp_path + "custom_opp_path_01";
  std::string custom_opp_path_invalid_01 = opp_path + "custom_opp_path_invalid_01";
  std::string custom_opp_path_empty = "";
  std::string custom_opp_path_02 = opp_path + "custom_opp_path_02";
  std::string custom_opp_path_invalid_02 = opp_path + "custom_opp_path_invalid_02";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, (custom_opp_path_01 + ":" +
                            custom_opp_path_invalid_01 + ":" +
                            custom_opp_path_empty + ":" +
                            custom_opp_path_02 + ":" +
                            custom_opp_path_invalid_02
                           ).c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path_01 + "/framework/caffe").c_str());
  system(("mkdir -p " + custom_opp_path_02 + "/framework/caffe").c_str());

  std::string customcaffe_path;
  Status ret = PluginManager::GetCustomCaffeProtoPath(customcaffe_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(customcaffe_path,
    custom_opp_path_01 + "/framework/caffe/:" +
    custom_opp_path_02 + "/framework/caffe/:" +
    path_vendors + "/customize/framework/caffe/:" +
    path_vendors + "/mdc/framework/caffe/:" +
    path_vendors + "/lhisi/framework/caffe/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetCustomCaffeProtoPath_09) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path_01 = opp_path + "custom_opp_path_01";
  std::string custom_opp_path_02 = opp_path + "custom_opp_path_02";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  mmSetEnv(kEnvNameCustom, (custom_opp_path_01 + ":" +
                            custom_opp_path_02 + ":" +
                            custom_opp_path_01 + ":" +
                            custom_opp_path_02
                           ).c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path_01 + "/framework/caffe").c_str());
  system(("mkdir -p " + custom_opp_path_02 + "/framework/caffe").c_str());

  std::string customcaffe_path;
  Status ret = PluginManager::GetCustomCaffeProtoPath(customcaffe_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(customcaffe_path,
    custom_opp_path_01 + "/framework/caffe/:" +
    custom_opp_path_02 + "/framework/caffe/:" +
    custom_opp_path_01 + "/framework/caffe/:" +
    custom_opp_path_02 + "/framework/caffe/:" +
    path_vendors + "/customize/framework/caffe/:" +
    path_vendors + "/mdc/framework/caffe/:" +
    path_vendors + "/lhisi/framework/caffe/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetConstantFoldingOpsPath_01) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  system(("rm -rf " + path_builtin).c_str());

  std::string customop_path;
  Status ret = PluginManager::GetConstantFoldingOpsPath("/tmp", customop_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(customop_path, opp_path + "/op_impl/built-in/host_cpu");
}

TEST_F(UtestPluginManager, test_plugin_manager_GetConstantFoldingOpsPath_02) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  system(("mkdir -p " + path_builtin).c_str());

  std::string customop_path;
  Status ret = PluginManager::GetConstantFoldingOpsPath("/tmp", customop_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(customop_path, opp_path + "/built-in/op_impl/host_cpu");
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, GetOppSupportedOsAndCpuType_SUCCESS) {
  std::unordered_map<std::string, std::unordered_set<std::string>> opp_supported_os_cpu;
  opp_supported_os_cpu["linux"] = {"x86_64"};
  opp_supported_os_cpu["windows"] = {"x86_64", "aarch64"};

  std::string opp_path = "./test_os_and_cpu/";
  // construct dir
  for (auto it0 : opp_supported_os_cpu) {
    std::string first_layer = opp_path + "built-in/op_proto/lib/" + it0.first;
    cout<<"====first layer: "<<first_layer<<endl;
    for (auto it1 : it0.second) {
      std::string second_layer = first_layer + "/" + it1;
      system(("mkdir -p " + second_layer).c_str());
      cout<<"====second layer: "<<second_layer<<endl;
    }
  }

  // get dir and check
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  std::unordered_map<std::string, std::unordered_set<std::string>> opp_supported_os_cpu_tmp;
  PluginManager::GetOppSupportedOsAndCpuType(opp_supported_os_cpu_tmp);
  for (auto it0 : opp_supported_os_cpu_tmp) {
    cout<<"====get first layer: "<<it0.first<<endl;
    ASSERT_NE(opp_supported_os_cpu.count(it0.first), 0);
    for (auto it1 : opp_supported_os_cpu_tmp[it0.first]) {
      cout<<"====get second layer: "<<it1<<endl;
      ASSERT_NE(opp_supported_os_cpu[it0.first].count(it1), 0);
    }
  }

  // del dir
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestPluginManager, GetOppSupportedOsAndCpuType_FAIL) {
  std::unordered_map<std::string, std::unordered_set<std::string>> opp_supported_os_cpu_tmp;
  PluginManager::GetOppSupportedOsAndCpuType(opp_supported_os_cpu_tmp, "./", "linux", 2);
  ASSERT_EQ(opp_supported_os_cpu_tmp.empty(), true);

  PluginManager::GetOppSupportedOsAndCpuType(opp_supported_os_cpu_tmp, "./not_exit", "linux", 1);
  ASSERT_EQ(opp_supported_os_cpu_tmp.empty(), true);

  std::string path = "./scene.info";
  system(("touch " + path).c_str());
  PluginManager::GetOppSupportedOsAndCpuType(opp_supported_os_cpu_tmp, path, "linux", 1);
  ASSERT_EQ(opp_supported_os_cpu_tmp.empty(), true);
  system(("rm -f " + path).c_str());
}

TEST_F(UtestPluginManager, GetVersionFromPath_fail) {
  std::string opp_version = "./version.info";
  system(("touch " + opp_version).c_str());
  std::string version;
  ASSERT_EQ(PluginManager::GetVersionFromPath(opp_version, version), false);
  ASSERT_EQ(version.empty(), true);

  system(("echo '=3.20.T100.0.B356' > " + opp_version).c_str());
  ASSERT_EQ(PluginManager::GetVersionFromPath(opp_version, version), false);
  ASSERT_EQ(version.empty(), true);

  system(("echo 'Version=' > " + opp_version).c_str());
  ASSERT_EQ(PluginManager::GetVersionFromPath(opp_version, version), false);
  ASSERT_EQ(version.empty(), true);

  system(("rm -f " + opp_version).c_str());
}

TEST_F(UtestPluginManager, GetCurEnvPackageOsAndCpuType) {
  std::string path ="./opp";
  system(("mkdir -p " + path).c_str());
  path += "/scene.info";
  system(("touch " + path).c_str());
  system(("echo 'os=linux' > " + path).c_str());
  system(("echo 'arch=x86_64' >> " + path).c_str());

  std::string path1 ="./../opp";
  system(("mkdir -p " + path1).c_str());
  path1 += "/scene.info";
  system(("touch " + path1).c_str());
  system(("echo 'os=linux' > " + path1).c_str());
  system(("echo 'arch=x86_64' >> " + path1).c_str());

  system("pwd");
  system(("realpath " + path).c_str());
  system(("realpath " + path1).c_str());

  std::string cur_env_os;
  std::string cur_env_cpu;
  PluginManager::GetCurEnvPackageOsAndCpuType(cur_env_os, cur_env_cpu);
  ASSERT_EQ(cur_env_os, "linux");
  ASSERT_EQ(cur_env_cpu, "x86_64");

  system(("echo -e 'error' > " + path).c_str());
  system(("echo -e 'error' > " + path1).c_str());
  cur_env_os.clear();
  cur_env_cpu.clear();
  PluginManager::GetCurEnvPackageOsAndCpuType(cur_env_os, cur_env_cpu);
  ASSERT_EQ(cur_env_os.empty(), true);
  ASSERT_EQ(cur_env_cpu.empty(), true);

  system(("rm -f " + path).c_str());
  system(("rm -f " + path1).c_str());
  cur_env_os.clear();
  cur_env_cpu.clear();
  PluginManager::GetCurEnvPackageOsAndCpuType(cur_env_os, cur_env_cpu);
  ASSERT_EQ(cur_env_os.empty(), true);
  ASSERT_EQ(cur_env_cpu.empty(), true);
}

int super_system(const char *cmd, char *retmsg, int msg_len) {
  FILE *fp;
  int res = -1;
  if (cmd == NULL || retmsg == NULL || msg_len < 0) {
    GELOGD("Err: Func:%s sys param invalid!", __func__);
    return 1;
  }
  if ((fp = popen(cmd, "r")) == NULL) {
    perror("popen");
    GELOGD("Err: Func:%s popen error:%s!", __func__, strerror(errno));
    return 2;
  } else {
    memset(retmsg, 0, msg_len);
    while (fgets(retmsg, msg_len, fp)) {
      GELOGD("Func:%s fgets buf is %s!", __func__, retmsg);
    }
    if ((res == pclose(fp)) == -1) {
      GELOGD("Func:%s close popen file pointer fp error!", __func__);
      return 3;
    }
    retmsg[strlen(retmsg) - 1] = '\0';
    return 0;
  }
}

TEST_F(UtestPluginManager, GetFileListWithSuffix_Success) {
  std::string path = "./test";
  system(("mkdir -p " + path).c_str());
  std::string file_path = path + "/a.so";
  system(("touch " + file_path).c_str());
  file_path = path + "/b.txt";
  system(("touch " + file_path).c_str());
  file_path = path + "/cc.so";
  system(("touch " + file_path).c_str());
  std::string command = "pwd";
  char retmsg[1024];
  int ret = super_system(command.c_str(), retmsg, sizeof(retmsg));
  path = retmsg;
  path += "/test";

  std::vector<std::string> file_list;
  PluginManager::GetFileListWithSuffix(path, ".so", file_list);
  ASSERT_EQ(file_list.size(), 2);

  file_list.clear();
  PluginManager::GetFileListWithSuffix(path, "c.so", file_list);
  ASSERT_EQ(file_list.size(), 1);

  file_list.clear();
  PluginManager::GetFileListWithSuffix(path, ".o", file_list);
  ASSERT_EQ(file_list.size(), 0);

  system(("rm -f " + path).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetVersion_Valid) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc' > " + path_config).c_str());

  system(("mkdir -p " + opp_path).c_str());
  system(("echo 'Version=6.4.T5.0.B121' > " + opp_path + "/version.info").c_str());

  std::string customize1_opp_path = path_vendors + "/customize/";
  system(("mkdir -p " + customize1_opp_path).c_str());
  system(("echo 'compiler_version=6.4.T5.0.B121' > " + customize1_opp_path + "/version.info").c_str());

  std::string customize2_opp_path = path_vendors + "/mdc/";
  system(("mkdir -p " + customize2_opp_path).c_str());
  system(("echo 'compiler_version=6.4.T5.0.B121' > " + customize2_opp_path + "/version.info").c_str());

  std::string customize_opp_path = opp_path + "lhisi";
  std::string customize_opp_path_version_info = customize_opp_path + "/version.info";
  system(("mkdir -p " + customize_opp_path + "/op_impl/ai_core/tbe/").c_str());
  system(("echo 'compiler_version=6.4.T5.0.B121' > " + customize_opp_path_version_info).c_str());
  mmSetEnv(kEnvNameCustom, customize_opp_path.c_str(), 1);

  std::string path;
  std::string path1;
  WriteRequiredVersion(">=6.4,<=6.4", path, path1);

  std::string op_tiling_path;
  Status ret = PluginManager::GetOpTilingPath(op_tiling_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_tiling_path,
            opp_path + "built-in/op_impl/ai_core/tbe/:" +
                path_vendors + "/mdc/op_impl/ai_core/tbe/:" +
                path_vendors + "/customize/op_impl/ai_core/tbe/:" +
                customize_opp_path + "/op_impl/ai_core/tbe/"
  );

  std::vector<std::string> vendors;
  PluginManager::GetPackageSoPath(vendors);
  std::vector<std::string> expect_vendors{customize_opp_path + "/",
                                          "",
                                          path_vendors + "/customize",
                                          path_vendors + "/mdc",
                                          opp_path + "built-in"};

  EXPECT_EQ(vendors.size(), expect_vendors.size());
  for(size_t i = 0U; i < vendors.size(); ++i) {
    EXPECT_EQ(vendors[i], expect_vendors[i]);
  }

  system(("rm -rf " + opp_path).c_str());
  system(("rm -rf " + path).c_str());
  system(("rm -rf " + path1).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_IsVendorVersionValid) {
  std::string path, path1;
  WriteRequiredVersion("\">=6.4,<=6.4\"", path, path1);
  EXPECT_EQ(PluginManager::IsVendorVersionValid("6.4.T5.0.B121", "6.4.T5.0.B121"), true);
  system(("rm -rf " + path + " " + path1).c_str());

  WriteRequiredVersion("6.5", path, path1);
  EXPECT_EQ(PluginManager::IsVendorVersionValid("6.4.T5.0.B121", "6.4.T5.0.B121"), false);
  system(("rm -rf " + path + " " + path1).c_str());

  WriteRequiredVersion("6.#", path, path1);
  EXPECT_EQ(PluginManager::IsVendorVersionValid("6.4.T5.0.B121", "6.4.T5.0.B121"), false);
  system(("rm -rf " + path + " " + path1).c_str());

  WriteRequiredVersion("", path, path1);
  EXPECT_EQ(PluginManager::IsVendorVersionValid("6.4.T5.0.B121", "6.4.T5.0.B121"), true);
  system(("rm -rf " + path + " " + path1).c_str());
}

TEST_F(UtestPluginManager, test_plugin_manager_GetRequiredOppAbiVersion_InValid) {
  std::string compiler_path ="./compiler";
  system(("mkdir -p " + compiler_path).c_str());
  auto path = compiler_path + "/version.info";
  system(("touch " + path).c_str());
  system(("echo 'required_opp_abi_version=\">=6.4\"' > " + path).c_str());

  std::string compiler_path1 ="./../compiler";
  system(("mkdir -p " + compiler_path1).c_str());
  auto path1 = compiler_path1 + "/version.info";
  system(("touch " + path1).c_str());
  system(("echo 'required_opp_abi_version=\">=6.4\"' > " + path1).c_str());

  std::vector<std::pair<uint32_t, uint32_t>> required_opp_abi_version;
  EXPECT_EQ(PluginManager::GetRequiredOppAbiVersion(required_opp_abi_version), false);

  system(("echo 'required_opp_abi_version=\">=6.#\"' > " + path).c_str());
  system(("echo 'required_opp_abi_version=\">=6.#\"' > " + path1).c_str());
  EXPECT_EQ(PluginManager::GetRequiredOppAbiVersion(required_opp_abi_version), false);

  system(("echo 'required_opp_abi_version=>=\"6.#,<=6.4\"' > " + path).c_str());
  system(("echo 'required_opp_abi_version=>=\"6.#,<=6.4\"' > " + path1).c_str());
  EXPECT_EQ(PluginManager::GetRequiredOppAbiVersion(required_opp_abi_version), false);

  system(("echo 'required_opp_abi_version=\">=6.4,<=6.#\"' > " + path).c_str());
  system(("echo 'required_opp_abi_version=\">=6.4,<=6.#\"' > " + path1).c_str());
  EXPECT_EQ(PluginManager::GetRequiredOppAbiVersion(required_opp_abi_version), false);

  system(("echo 'required_opp_abi_version=\">=6.4,<=6.4\"' > " + path).c_str());
  system(("echo 'required_opp_abi_version=\">=6.4,<=6.4\"' > " + path1).c_str());
  EXPECT_EQ(PluginManager::IsVendorVersionValid("6.#.T5.0.B121", ""), false);
  EXPECT_EQ(PluginManager::IsVendorVersionValid("", "6.4.T5.0.B121,6.#.T5.0.B121"), false);
  EXPECT_EQ(PluginManager::IsVendorVersionValid("", "6.3.T5.0.B121"), false);

  system(("echo 'required_opp_abi_version=\">=6.3, <=6.4\"' > " + path).c_str());
  system(("echo 'required_opp_abi_version=\">=6.3, <=6.4\"' > " + path1).c_str());
  EXPECT_EQ(PluginManager::IsVendorVersionValid("6.4.T5.0.B121", "6.3.T5.0.B121, 6.4.T5.0.B121"), true);

  system(("rm -rf " + compiler_path).c_str());
  system(("rm -rf " + compiler_path1).c_str());
}

TEST_F(UtestPluginManager, GetVersionFromPath_succ) {
  std::string opp_version = "./version.info";
  system(("touch " + opp_version).c_str());
  std::string version;
  system(("echo 'Version=6.4.T5.0.B121' > " + opp_version).c_str());
  ASSERT_EQ(PluginManager::GetVersionFromPath(opp_version, version), true);
  ASSERT_EQ(version, "6.4.T5.0.B121");

  system(("echo 'required_opp_abi_version=\">=6.4,<=6.4\"' > " + opp_version).c_str());
  ASSERT_EQ(PluginManager::GetVersionFromPathWithName(opp_version, version, "required_opp_abi_version="), true);
  ASSERT_EQ(version, "\">=6.4,<=6.4\"");

  system(("rm -f " + opp_version).c_str());
}
}  // namespace ge

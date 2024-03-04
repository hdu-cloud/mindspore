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

#ifndef AIR_CXX_TESTS_UT_GE_RUNTIME_V2_SPACE_REGISTRY_FAKER_H_
#define AIR_CXX_TESTS_UT_GE_RUNTIME_V2_SPACE_REGISTRY_FAKER_H_
#include "register/op_impl_registry_api.h"
#include "register/op_impl_registry_holder_manager.h"
#include "register/op_impl_space_registry.h"
#include "graph/any_value.h"
#include "common/ge_common/debug/ge_log.h"

namespace gert {
int SuperSystem(const char *cmd, char *retmsg, int msg_len);
std::vector<std::string> CreateSceneInfo();
void CreateVersionInfo();
void DestroyVersionInfo();
void LoadDefaultSpaceRegistry();
void UnLoadDefaultSpaceRegistry();
void CreateOpmasterSoEnvInfoFunc(std::string opp_path);
void CreateVendorsOppSo(const std::string &opp_path, const std::string &customize_1 = "",
                        const std::string &customize_2 = "");

class SpaceRegistryFaker {
 public:
  OpImplSpaceRegistryPtr Build() {
    auto impl_num = GetRegisteredOpNum();
    auto impl_funcs = std::unique_ptr<TypesToImpl[]>(new(std::nothrow) TypesToImpl[impl_num]);
    auto ret = GetOpImplFunctions(reinterpret_cast<TypesToImpl *>(impl_funcs.get()), impl_num);
    if (ret != ge::GRAPH_SUCCESS) {
      GELOGE(ge::FAILED, "Get functions from OpImplRegistry failed!");
      return nullptr;
    }
    auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
    auto &types_to_impl = registry_holder->GetTypesToImpl();
    for (size_t i = 0U; i < impl_num; ++i) {
      types_to_impl[impl_funcs[i].op_type] = impl_funcs[i].funcs;
    }
    GELOGI("ori size:%zu, after size:%zu", types_to_impl.size(), registry_holder->GetTypesToImpl().size());
    auto space_registry = std::make_shared<gert::OpImplSpaceRegistry>();
    space_registry->AddRegistry(registry_holder);
    return space_registry;
  }

  static void UpdateOpImplToDefaultSpaceRegistry() {
    auto space_registry = gert::DefaultOpImplSpaceRegistry::GetInstance().GetDefaultSpaceRegistry();
    if (space_registry != nullptr) {
      GELOGD("Default Space registry already exist!");
      return;
    }
    auto impl_num = GetRegisteredOpNum();
    auto impl_funcs = std::unique_ptr<TypesToImpl[]>(new(std::nothrow) TypesToImpl[impl_num]);
    auto ret = GetOpImplFunctions(reinterpret_cast<TypesToImpl *>(impl_funcs.get()), impl_num);
    if (ret !=  ge::GRAPH_SUCCESS) {
      GELOGE(ge::FAILED, "Get functions from OpImplRegistry failed!");
      return;
    }
    auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
    auto &types_to_impl = registry_holder->GetTypesToImpl();
    for (size_t i = 0U; i < impl_num; ++i) {
      types_to_impl[impl_funcs[i].op_type] = impl_funcs[i].funcs;
    }
    GELOGI("ori size:%zu, after size:%zu", types_to_impl.size(), registry_holder->GetTypesToImpl().size());
    space_registry = std::make_shared<OpImplSpaceRegistry>();
    if (space_registry == nullptr) {
      GELOGE(ge::FAILED, "Create space registry failed!");
      return;
    }
    space_registry->AddRegistry(registry_holder);
    gert::DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(space_registry);
  }

  static void SetefaultSpaceRegistryNull() {
    gert::DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(nullptr);
  }
};
}  // namespace gert
#endif  //AIR_CXX_TESTS_UT_GE_RUNTIME_V2_SPACE_REGISTRY_FAKER_H_

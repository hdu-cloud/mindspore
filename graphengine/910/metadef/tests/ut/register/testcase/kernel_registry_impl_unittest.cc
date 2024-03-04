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
#include "register/kernel_registry_impl.h"
#include <gtest/gtest.h>
namespace gert {
namespace {
ge::graphStatus TestFuncCreator(const ge::Node *, KernelContext *) {
 return ge::GRAPH_SUCCESS;
}
KernelStatus TestFunc(KernelContext *) {
 return 0;
}
std::vector<std::string> TestTraceFunc(const gert::KernelContext *) {
 return {""};
}
}
class KernelRegistryImplUT : public testing::Test {};
TEST_F(KernelRegistryImplUT, RegisterAndFind_Ok_AllFuncRegistered) {
 KernelRegistryImpl registry;
 registry.RegisterKernel("Foo", {{TestFunc, nullptr, nullptr, TestFuncCreator, TestTraceFunc}, ""});
 ASSERT_NE(registry.FindKernelFuncs("Foo"), nullptr);
 ASSERT_EQ(registry.FindKernelFuncs("Foo")->run_func, &TestFunc);
 ASSERT_EQ(registry.FindKernelFuncs("Foo")->outputs_creator_func, &TestFuncCreator);
 ASSERT_EQ(registry.FindKernelFuncs("Foo")->trace_printer, &TestTraceFunc);
}
TEST_F(KernelRegistryImplUT, RegisterAndFind_Ok_OnlyRegisterRunFunc) {
 KernelRegistryImpl registry;
 registry.RegisterKernel("Foo", {{TestFunc, nullptr, nullptr}, ""});
 ASSERT_NE(registry.FindKernelFuncs("Foo"), nullptr);
 ASSERT_EQ(registry.FindKernelFuncs("Foo")->run_func, &TestFunc);
 ASSERT_EQ(registry.FindKernelFuncs("Foo")->outputs_creator_func, nullptr);
 ASSERT_EQ(registry.FindKernelFuncs("Foo")->trace_printer, nullptr);
}
TEST_F(KernelRegistryImplUT, FailedToFindWhenNotRegister) {
 KernelRegistryImpl registry;
 ASSERT_EQ(registry.FindKernelFuncs("Foo"), nullptr);
 ASSERT_EQ(registry.FindKernelInfo("Foo"), nullptr);
}
TEST_F(KernelRegistryImplUT, GetAll_Ok) {
 KernelRegistryImpl registry;
 registry.RegisterKernel("Foo", {{TestFunc, nullptr, nullptr}, "memory"});
 std::unordered_map<std::string, KernelRegistry::KernelInfo> expect_kernel_infos = {
     {"Foo", {{TestFunc, nullptr, nullptr, nullptr, nullptr}, "memory"}},
     {"Bar", {{TestFunc, nullptr, nullptr, TestFuncCreator, TestTraceFunc}, "memory"}}
 };
 registry.RegisterKernel("Foo", {{TestFunc, nullptr, nullptr, nullptr, nullptr}, "memory"});
 registry.RegisterKernel("Bar", {{TestFunc, nullptr, nullptr, TestFuncCreator, TestTraceFunc}, "memory"});
 ASSERT_EQ(registry.GetAll().size(), expect_kernel_infos.size());
 for (const auto &key_to_infos : registry.GetAll()) {
   ASSERT_TRUE(expect_kernel_infos.count(key_to_infos.first) > 0);
   ASSERT_EQ(key_to_infos.second.func.run_func, expect_kernel_infos[key_to_infos.first].func.run_func);
   ASSERT_EQ(key_to_infos.second.critical_section, expect_kernel_infos[key_to_infos.first].critical_section);
 }
}
}  // namespace gert
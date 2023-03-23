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
#include "exe_graph/runtime/tiling_parse_context.h"
#include <gtest/gtest.h>
#include <vector>
#include "faker/kernel_run_context_faker.h"
namespace gert {
class TilingParseContextUT : public testing::Test {};
struct CompiledInfo1 {
  uint64_t a;
  uint64_t b;
};
struct CompileDInfo2 {
  uint32_t a;
  uint32_t b;
};
struct CompiledInfo3 {
  int32_t core_num;
};

TEST_F(TilingParseContextUT, GetIoOk) {
  char *json_str = "{}";
  CompiledInfo1 ci = {10, 20};
  auto context_holder = KernelRunContextFaker().KernelIONum(1, 1).Inputs({json_str}).Outputs({&ci}).Build();
  auto context = context_holder.GetContext<TilingParseContext>();
  ASSERT_NE(context, nullptr);
  EXPECT_STREQ(context->GetCompiledJson(), "{}");
  ASSERT_NE(context->GetCompiledInfo<CompiledInfo1>(), nullptr);
  EXPECT_EQ(context->GetCompiledInfo<CompiledInfo1>()->a, 10);
  EXPECT_EQ(context->GetCompiledInfo<CompiledInfo1>()->b, 20);
}
TEST_F(TilingParseContextUT, SetCompiledInfoOk) {
  char *json_str = "{}";
  CompiledInfo1 ci = {10, 20};
  auto context_holder = KernelRunContextFaker().KernelIONum(1, 1).Inputs({json_str}).Outputs({nullptr}).Build();

}

TEST_F(TilingParseContextUT, CompiledInfoLessThan8Bytes) {
  char *json_str = "{}";
  CompiledInfo3 ci = {2};
  auto context_holder = KernelRunContextFaker().KernelIONum(1, 1).Inputs({json_str}).Outputs({&ci}).Build();
  auto context = context_holder.GetContext<TilingParseContext>();
  ASSERT_NE(context, nullptr);
  EXPECT_STREQ(context->GetCompiledJson(), "{}");
  ASSERT_NE(context->GetCompiledInfo<CompiledInfo3>(), nullptr);
  EXPECT_EQ(context->GetCompiledInfo<CompiledInfo3>()->core_num, 2);
}
}  // namespace gert
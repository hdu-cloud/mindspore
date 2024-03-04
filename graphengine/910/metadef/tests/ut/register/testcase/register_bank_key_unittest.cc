/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <gtest/gtest.h>
#include <iostream>
#include "nlohmann/json.hpp"
#include "register/tuning_bank_key_registry.h"

namespace tuningtiling {
struct DynamicRnnInputArgsV2 {
  int64_t batch;
  int32_t dims;
};
bool ConvertTilingContext(const gert::TilingContext* context,
                          std::shared_ptr<void> &input_args, size_t &size) {
  if (context == nullptr) {
    auto rnn = std::make_shared<DynamicRnnInputArgsV2>();
    rnn->batch = 0;
    rnn->dims = 1;
    size = sizeof(DynamicRnnInputArgsV2);
    input_args = rnn;
    return false;
  }
  return true;
}

DECLARE_STRUCT_RELATE_WITH_OP(DynamicRNN, DynamicRnnInputArgsV2,
  batch, dims);
REGISTER_OP_BANK_KEY_CONVERT_FUN(DynamicRNN, ConvertTilingContext);
class RegisterOPBankKeyUT : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(RegisterOPBankKeyUT, convert_tiling_context) {
  auto& func = OpBankKeyFuncRegistry::RegisteredOpFuncInfo();
  auto iter = func.find("DynamicRNN");
  nlohmann::json test;
  test["batch"] = 12;
  test["dims"] = 2;
  ASSERT_TRUE(iter != func.cend());
  const OpBankLoadFun& load_func = iter->second.GetBankKeyLoadFunc();
  std::shared_ptr<void> ld = nullptr;
  size_t len = 0;
  EXPECT_TRUE(load_func(ld, len, test));
  EXPECT_TRUE(ld != nullptr);
  const auto &parse_func = iter->second.GetBankKeyParseFunc();
  nlohmann::json test2;
  EXPECT_TRUE(parse_func(ld, len, test2));
  EXPECT_EQ(test, test2);
  const auto &convert_func = iter->second.GetBankKeyConvertFunc();
  std::shared_ptr<void> op_key = nullptr;
  size_t s = 0U;
  EXPECT_FALSE(convert_func(nullptr, op_key, s));
  EXPECT_TRUE(s !=0);
  EXPECT_TRUE(op_key != nullptr);
  auto rnn_ky = std::static_pointer_cast<DynamicRnnInputArgsV2>(op_key);
  EXPECT_EQ(rnn_ky->batch, 0);

}
}  // namespace tuningtiling

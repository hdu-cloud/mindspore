/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.

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

#include "register/tuning_bank_key_registry.h"
#include "common/ge_common/debug/ge_log.h"

namespace tuningtiling {
OpBankKeyFuncInfo::OpBankKeyFuncInfo(const ge::AscendString &optype) : optype_(optype) {}

void OpBankKeyFuncInfo::SetOpConvertFunc(const OpBankKeyConvertFun &convert_func) {
  convert_func_ = convert_func;
}

void OpBankKeyFuncInfo::SetOpParseFunc(const OpBankParseFun &parse_func) {
  parse_func_ = parse_func;
}

void OpBankKeyFuncInfo::SetOpLoadFunc(const OpBankLoadFun &load_func) {
  load_func_ = load_func;
}

const OpBankKeyConvertFun& OpBankKeyFuncInfo::OpBankKeyFuncInfo::GetBankKeyConvertFunc() const {
  return convert_func_;
}

const OpBankParseFun& OpBankKeyFuncInfo::GetBankKeyParseFunc() const {
  return parse_func_;
}

const OpBankLoadFun& OpBankKeyFuncInfo::GetBankKeyLoadFunc() const {
  return load_func_;
}

std::unordered_map<ge::AscendString, OpBankKeyFuncInfo> &OpBankKeyFuncRegistry::RegisteredOpFuncInfo() {
  static std::unordered_map<ge::AscendString, OpBankKeyFuncInfo> op_func_map;
  return op_func_map;
}

OpBankKeyFuncRegistry::OpBankKeyFuncRegistry(const ge::AscendString &optype, const OpBankKeyConvertFun &convert_func) {
  auto &op_func_map = RegisteredOpFuncInfo();
  const auto iter = op_func_map.find(optype);
  if (iter == op_func_map.cend()) {
    OpBankKeyFuncInfo op_func_info(optype);
    op_func_info.SetOpConvertFunc(convert_func);
    (void)op_func_map.emplace(optype, op_func_info);
  } else {
    iter->second.SetOpConvertFunc(convert_func);
  }
  GELOGI("Register op bank key convert function for optype:%s", optype.GetString());
}

OpBankKeyFuncRegistry::OpBankKeyFuncRegistry(const ge::AscendString &optype,
                                             const OpBankParseFun &parse_func, const OpBankLoadFun &load_func) {
  auto &op_func_map = RegisteredOpFuncInfo();
  const auto iter = op_func_map.find(optype);
  if (iter == op_func_map.cend()) {
    OpBankKeyFuncInfo op_func_info(optype);
    op_func_info.SetOpParseFunc(parse_func);
    op_func_info.SetOpLoadFunc(load_func);
    (void)op_func_map.emplace(optype, op_func_info);
  } else {
    iter->second.SetOpParseFunc(parse_func);
    iter->second.SetOpLoadFunc(load_func);
  }
  GELOGI("Register op bank key parse and load function for optype:%s", optype.GetString());
}
}  // namespace tuningtiling

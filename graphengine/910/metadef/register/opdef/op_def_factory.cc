/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#include <iostream>
#include <algorithm>
#include <map>
#include "register/op_def.h"
#include "op_def_impl.h"
#include "register/op_def_factory.h"

namespace ops {
static std::map<ge::AscendString, OpDefCreator> g_opsdef_creator;
static std::vector<ge::AscendString> g_ops_list;

int OpDefFactory::OpDefRegister(const char *name, OpDefCreator creator) {
  g_opsdef_creator.emplace(name, creator);
  g_ops_list.emplace_back(name);
  return 0;
}
OpDef OpDefFactory::OpDefCreate(const char *name) {
  auto it = g_opsdef_creator.find(name);
  if (it != g_opsdef_creator.cend()) {
    return it->second(name);
  }
  return OpDef("default");
}

std::vector<ge::AscendString> &OpDefFactory::GetAllOp(void) {
  return g_ops_list;
}
}  // namespace ops

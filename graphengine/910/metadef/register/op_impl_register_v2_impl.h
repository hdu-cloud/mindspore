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

#ifndef METADEF_CXX_REGISTER_OP_IMPL_REGISTER_V_2_IMPL_H_
#define METADEF_CXX_REGISTER_OP_IMPL_REGISTER_V_2_IMPL_H_
#include "register/op_impl_registry.h"
#include "register/op_impl_registry_base.h"
namespace gert {
class OpImplRegisterV2Impl {
 public:
  OpImplRegistry::OpType op_type;
  OpImplRegistry::OpImplFunctions functions;
  bool is_private_attr_registered = false;
};
}  // namespace gert
#endif  // METADEF_CXX_REGISTER_OP_IMPL_REGISTER_V_2_IMPL_H_

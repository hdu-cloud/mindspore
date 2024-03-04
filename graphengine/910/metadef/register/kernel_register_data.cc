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
#include "kernel_register_data.h"

namespace gert {
namespace {
ge::graphStatus NullCreator(const ge::Node *node, KernelContext *context) {
  (void) node;
  (void) context;
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus NullDestoryer(const ge::Node *node, KernelContext *context) {
  (void) node;
  (void) context;
  return ge::GRAPH_SUCCESS;
}
}  // namespace
KernelRegisterData::KernelRegisterData(const ge::char_t *kernel_type) : kernel_type_(kernel_type) {
  funcs_.outputs_creator = NullCreator;
  funcs_.outputs_creator_func = NullCreator;
  funcs_.outputs_initializer = NullDestoryer;
  funcs_.trace_printer = nullptr;
  critical_section_ = "";
  funcs_.profiling_info_filler = nullptr;
  funcs_.data_dump_info_filler = nullptr;
  funcs_.exception_dump_info_filler = nullptr;
}
}  // namespace gert
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
#include "exe_graph/runtime/runtime_attrs.h"
#include "common/ge_common/debug/ge_log.h"
#include "runtime_attrs_def.h"
#include "graph/def_types.h"
namespace gert {
const void *RuntimeAttrs::GetPointerByIndex(size_t index) const {
  auto attrs = ge::PtrToPtr<uint64_t, const RuntimeAttrsDef>(&placeholder_);
  if (index >= attrs->attr_num) {
    GELOGE(ge::FAILED, "Failed to get attr, the index %zu out of range %zu", index, attrs->attr_num);
    return nullptr;
  }
  return ge::PtrToPtr<const RuntimeAttrsDef, const uint8_t>(attrs) + attrs->offset[index];
}
size_t RuntimeAttrs::GetAttrNum() const {
  return ge::PtrToPtr<uint64_t, const RuntimeAttrsDef>(&placeholder_)->attr_num;
}
}  // namespace gert

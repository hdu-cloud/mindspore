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
#include "cache_desc_stub//runtime_cache_desc.h"
#include "common/checker.h"

namespace ge {
bool RuntimeCacheDesc::IsEqual(const CacheDescPtr &other) const {
  auto rht = dynamic_cast<const RuntimeCacheDesc *>(other.get());
  GE_ASSERT_NOTNULL(rht, "Type error, expect type ge::RuntimeCacheDesc.");
  return (*this == *rht);
}

bool RuntimeCacheDesc::IsMatch(const CacheDescPtr &other) const {
  return IsEqual(other);
}

CacheHashKey RuntimeCacheDesc::GetCacheDescHash() const {
  CacheHashKey hash_key = 0U;
  const char_t sep = ',';
  for (const auto &shape : shapes_) {
    auto seed = HashUtils::MultiHash();
    for (size_t idx = 0U; idx < shape.GetDimNum(); ++idx) {
      seed = HashUtils::HashCombine(seed, shape.GetDim(idx));
    }
    hash_key = HashUtils::HashCombine(seed, sep);
  }
  return hash_key;
}

bool RuntimeCacheDesc::operator==(const RuntimeCacheDesc &rht) const {
  if (shapes_.size() != rht.shapes_.size()) {
    return false;
  }
  for (size_t idx = 0U; idx < shapes_.size(); ++idx) {
    if (shapes_[idx] != rht.shapes_[idx]) {
      return false;
    }
  }
  return true;
}
}  // namespace ge

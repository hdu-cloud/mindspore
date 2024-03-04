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
#include "graph/cache_policy/match_policy_for_exactly_the_same.h"

namespace ge {
CacheItemId MatchPolicyForExactlyTheSame::GetCacheItemId(const CCStatType &cc_state,
                                                         const CacheDescPtr &cache_desc) const {
  const CacheHashKey hash_key = cache_desc->GetCacheDescHash();
  const auto &iter = cc_state.find(hash_key);
  if (iter == cc_state.end() || iter->second.empty()) {
    GELOGD("[CACHE] hash [%lu] not exist.", hash_key);
    return KInvalidCacheItemId;
  }
  const auto &info_vec = iter->second;
  const auto cached_info = std::find_if(info_vec.begin(), info_vec.end(), [&cache_desc](const CacheInfo &cached) {
    return (cache_desc->IsEqual(cached.GetCacheDesc()));
  });
  if (cached_info != info_vec.cend()) {
    return cached_info->GetItemId();
  } else {
    GELOGD("[CACHE] hash [%lu] collision occurred, the same cached desc not found.", hash_key);
    return KInvalidCacheItemId;
  }
}
}  // namespace ge
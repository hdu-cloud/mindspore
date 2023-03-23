/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#include "graph/compile_cache_policy/match_policy_exact_only.h"
#include "graph/compile_cache_policy/compile_cache_hasher.h"
namespace ge {
CacheItemId MatchPolicyExactOnly::GetCacheItemId(const CCStatType &cc_state, const CompileCacheDesc &desc) const {
  const CacheHashKey hash_key = CompileCacheHasher::GetCacheDescHashWithoutShape(desc);
  const auto &iter = cc_state.find(hash_key);
  if (iter == cc_state.end()) {
    GELOGD("can not find without shape hash %lu", hash_key);
    return KInvalidCacheItemId;
  }
  const auto &info_vec = iter->second;
  const auto cached_info = std::find_if(info_vec.begin(), info_vec.end(), [&desc] (const CacheInfo &cached) {
      return (CompileCacheDesc::IsMatchedCompileDesc(cached.GetCompileCacheDesc(), desc));
  });
  if (cached_info != info_vec.end()) {
    return cached_info->GetItemId();
  } else {
    return KInvalidCacheItemId;
  }
}
}
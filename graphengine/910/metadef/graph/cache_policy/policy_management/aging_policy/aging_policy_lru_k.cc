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

#include "graph/cache_policy/aging_policy_lru_k.h"
namespace ge {
std::vector<CacheItemId> AgingPolicyLruK::DoAging(const CacheState &cache_state) const {
  size_t cur_depth = cache_state.GetCacheInfoNum();
  const auto &cc_state = cache_state.GetState();
  GELOGD("[CACHE][AGING] current depth[%zu] cache queue capacity[%zu].", cur_depth, depth_);
  if (cur_depth <= depth_) {
    return {};
  }
  std::pair<CacheItemId, uint64_t> delete_item({KInvalidCacheItemId, UINT64_MAX});
  for (const auto &each_cc_state : cc_state) {
    for (const auto &cache_info : each_cc_state.second) {
      if (cache_info.GetTimerCount() <= delete_item.second) {
        delete_item = {cache_info.GetItemId(), cache_info.GetTimerCount()};
      }
    }
  }
  if (delete_item.first == KInvalidCacheItemId) {
    return {};
  }
  return {delete_item.first};
}

bool AgingPolicyLruK::IsCacheDescAppearKTimes(const CacheHashKey hash_key, const CacheDescPtr &cache_desc) {
  const std::lock_guard<std::mutex> lock(hash_2_cache_descs_and_count_mu_);
  if (hash_2_cache_descs_and_count_.count(hash_key) > 0U) {
    auto &cache_descs_and_count = hash_2_cache_descs_and_count_[hash_key];
    for (auto &cache_desc_and_count : cache_descs_and_count) {
      if (cache_desc->IsEqual(cache_desc_and_count.first)) {
        ++cache_desc_and_count.second;
        return cache_desc_and_count.second >= k_times_;
      }
    }
  }
  hash_2_cache_descs_and_count_[hash_key].emplace_back(std::make_pair(cache_desc, 1U));
  return false;
}
}  // namespace ge

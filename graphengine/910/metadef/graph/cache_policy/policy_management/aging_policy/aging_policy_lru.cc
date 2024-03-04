/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
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
#include "graph/cache_policy/aging_policy_lru.h"
namespace ge {
std::vector<CacheItemId> AgingPolicyLru::DoAging(const CacheState &cache_state) const {
  const auto &cc_state = cache_state.GetState();
  if (cache_state.GetCurTimerCount() <= delete_interval_) {
    GELOGE(ge::PARAM_INVALID, "[Aging][Lru]Delete interval param is invalid. Delete interval is %lu, expect[0, %lu].",
           delete_interval_, cache_state.GetCurTimerCount());
    return {};
  }
  const uint64_t timer_count_lower_bound = cache_state.GetCurTimerCount() - delete_interval_;
  std::vector<CacheItemId> delete_item;
  for (const auto &cache_item : cc_state) {
    const std::vector<CacheInfo> &cache_vec = cache_item.second;
    for (auto iter = cache_vec.begin(); iter != cache_vec.end(); iter++) {
      if ((*iter).GetTimerCount() < timer_count_lower_bound) {
          delete_item.emplace_back((*iter).GetItemId());
      }
    }
  }
  return delete_item;
}
}
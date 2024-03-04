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

#include "graph/cache_policy/cache_policy.h"
#include "graph/debug/ge_util.h"

namespace ge {
std::unique_ptr<CachePolicy> CachePolicy::Create(const MatchPolicyPtr &mp, const AgingPolicyPtr &ap) {
  if (mp == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] param match policy must not be null.");
    return nullptr;
  }
  if (ap == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] param aging policy must not be null.");
    return nullptr;
  }
  auto ccp = ComGraphMakeUnique<CachePolicy>();
  if (ccp == nullptr) {
    GELOGE(GRAPH_FAILED, "Create CachePolicy failed.");
    return nullptr;
  }
  (void)ccp->SetAgingPolicy(ap);
  (void)ccp->SetMatchPolicy(mp);

  GELOGI("[CachePolicy] Create CachePolicy success;");
  return ccp;
}

std::unique_ptr<CachePolicy> CachePolicy::Create(const MatchPolicyType mp_type,
                                                 const AgingPolicyType ap_type, size_t cached_aging_depth) {
  const auto mp = PolicyRegister::GetInstance().GetMatchPolicy(mp_type);
  GE_ASSERT_NOTNULL(mp);
  const auto ap = PolicyRegister::GetInstance().GetAgingPolicy(ap_type);
  GE_ASSERT_NOTNULL(ap);
  ap->SetCachedAgingDepth(cached_aging_depth);
  auto ccp = ComGraphMakeUnique<CachePolicy>();
  GE_ASSERT_NOTNULL(ccp);
  (void)ccp->SetAgingPolicy(ap);
  (void)ccp->SetMatchPolicy(mp);
  GELOGI("[CachePolicy] Create CachePolicy with match_policy: %d, aging_policy: %d success;",
         static_cast<int32_t>(mp_type), static_cast<int32_t>(ap_type));
  return ccp;
}

graphStatus CachePolicy::SetMatchPolicy(const MatchPolicyPtr mp) {
  GE_CHECK_NOTNULL(mp);
  mp_ = mp;
  return GRAPH_SUCCESS;
}

graphStatus CachePolicy::SetAgingPolicy(const AgingPolicyPtr ap) {
  GE_CHECK_NOTNULL(ap);
  ap_ = ap;
  return GRAPH_SUCCESS;
}

CacheItemId CachePolicy::AddCache(const CacheDescPtr &cache_desc) {
  const CacheHashKey main_hash_key = cache_desc->GetCacheDescHash();
  if (!ap_->IsReadyToAddCache(main_hash_key, cache_desc)) {
    GELOGI("Not met the add cache condition with has key:%lu.", main_hash_key);
    return KInvalidCacheItemId;
  }
  const auto cache_item = compile_cache_state_.AddCache(main_hash_key, cache_desc);
  if (cache_item == KInvalidCacheItemId) {
    GELOGE(GRAPH_FAILED, "[Check][Param] AddCache failed: please check the compile cache description.");
    return KInvalidCacheItemId;
  }
  return cache_item;
}

CacheItemId CachePolicy::FindCache(const CacheDescPtr &cache_desc) const {
  if (mp_ == nullptr) {
    GELOGW("match policy is nullptr");
    return KInvalidCacheItemId;
  }
  return mp_->GetCacheItemId(compile_cache_state_.GetState(), cache_desc);
}

std::vector<CacheItemId> CachePolicy::DeleteCache(const DelCacheFunc &func) {
  const auto delete_items = compile_cache_state_.DelCache(func);
  GELOGI("[CachePolicy] [DeleteCache] Delete %zu CacheInfos.", delete_items.size());
  return delete_items;
}

std::vector<CacheItemId> CachePolicy::DeleteCache(const std::vector<CacheItemId> &delete_item) {
  const auto delete_items = compile_cache_state_.DelCache(delete_item);
  GELOGI("[CachePolicy] [DeleteCache] Delete %zu CompileCacheInfo", delete_items.size());
  return delete_items;
}

std::vector<CacheItemId> CachePolicy::DoAging() {
  const auto delete_item = ap_->DoAging(compile_cache_state_);
  (void)compile_cache_state_.DelCache(delete_item);
  return delete_item;
}
}  // namespace ge
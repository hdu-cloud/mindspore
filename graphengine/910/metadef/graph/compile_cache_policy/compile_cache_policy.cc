/**
 * Copyright (c) Huawei Technologies Co., Ltd
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

#include "graph/compile_cache_policy/compile_cache_policy.h"
#include "graph/cache_policy/match_policy_exact_only.h"
#include "graph/cache_policy/aging_policy_lru.h"
#include "debug/ge_util.h"
#include "graph/debug/ge_util.h"

namespace ge {
std::unique_ptr<CompileCachePolicy> CompileCachePolicy::Create(const MatchPolicyPtr mp,
                                                               const AgingPolicyPtr ap) {
  if (mp == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] param match policy must not be null.");
    return nullptr;
  }
  if (ap == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] param aging policy must not be null.");
    return nullptr;
  }
  auto ccp = ComGraphMakeUnique<CompileCachePolicy>();
  (void)ccp->SetAgingPolicy(ap);
  (void)ccp->SetMatchPolicy(mp);

  GELOGI("[CompileCachePolicy] Create CompileCachePolicy success;");
  return ccp;
}

std::unique_ptr<CompileCachePolicy> CompileCachePolicy::Create(const MatchPolicyType mp_type,
                                                               const AgingPolicyType ap_type) {
  const auto mp = PolicyRegister::GetInstance().GetMatchPolicy(mp_type);
  const auto ap = PolicyRegister::GetInstance().GetAgingPolicy(ap_type);
  auto ccp = ComGraphMakeUnique<CompileCachePolicy>();
  (void)ccp->SetAgingPolicy(ap);
  (void)ccp->SetMatchPolicy(mp);
  GELOGI("[CompileCachePolicy] Create CompileCachePolicy with match_policy: %d, aging_policy: %d success;",
         static_cast<int32_t>(mp_type), static_cast<int32_t>(ap_type));
  return ccp;
}

graphStatus CompileCachePolicy::SetMatchPolicy(const MatchPolicyPtr mp) {
  GE_CHECK_NOTNULL(mp);
  mp_ = mp;
  return GRAPH_SUCCESS;
}

graphStatus CompileCachePolicy::SetAgingPolicy(const AgingPolicyPtr ap) {
  GE_CHECK_NOTNULL(ap);
  ap_ = ap;
  return GRAPH_SUCCESS;
}

CacheItemId CompileCachePolicy::AddCache(const CompileCacheDesc &compile_cache_desc) {
  // 为了蓝区air仓老接口兼容上库，air仓修改使用CachePolicy新接口后删除
  CompileCacheDescPtr compile_cache_desc_ptr = make_shared<CompileCacheDesc>(compile_cache_desc);
  if (compile_cache_desc_ptr == nullptr) {
    GELOGW("compile cache desc is nullptr");
    return KInvalidCacheItemId;
  }
  const CacheHashKey main_hash_key = compile_cache_desc.GetCacheDescHash();
  const auto cache_item = compile_cache_state_.AddCache(main_hash_key, compile_cache_desc_ptr);
  if (cache_item == KInvalidCacheItemId) {
    GELOGE(GRAPH_FAILED, "[Check][Param] AddCache failed: please check the compile cache description.");
    return KInvalidCacheItemId;
  }
  return cache_item;
}

CacheItemId CompileCachePolicy::FindCache(const CompileCacheDesc &compile_cache_desc) const {
  if (mp_ == nullptr) {
    GELOGW("match policy is nullptr");
    return KInvalidCacheItemId;
  }
  // 为了蓝区air仓老接口兼容上库，air仓修改使用CachePolicy新接口后删除
  CompileCacheDescPtr compile_cache_desc_ptr = make_shared<CompileCacheDesc>(compile_cache_desc);
  if (compile_cache_desc_ptr == nullptr) {
    GELOGW("compile cache desc is nullptr");
    return KInvalidCacheItemId;
  }
  return mp_->GetCacheItemId(compile_cache_state_.GetState(), compile_cache_desc_ptr);
}

std::vector<CacheItemId> CompileCachePolicy::DeleteCache(const DelCacheFunc &func) {
  const auto delete_items = compile_cache_state_.DelCache(func);
  GELOGI("[CompileCachePolicy] [DeleteCache] Delete %zu CompileCacheInfo", delete_items.size());
  return delete_items;
}

std::vector<CacheItemId> CompileCachePolicy::DeleteCache(const std::vector<CacheItemId> &delete_item) {
  const auto delete_items = compile_cache_state_.DelCache(delete_item);
  GELOGI("[CompileCachePolicy] [DeleteCache] Delete %zu CompileCacheInfo", delete_items.size());
  return delete_items;
}

std::vector<CacheItemId> CompileCachePolicy::DoAging() {
  const auto delete_item = ap_->DoAging(compile_cache_state_);
  (void)compile_cache_state_.DelCache(delete_item);
  return delete_item;
}
}  // namespace ge
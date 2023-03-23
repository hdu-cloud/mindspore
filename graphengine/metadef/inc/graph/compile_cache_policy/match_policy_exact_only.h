/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef GRAPH_COMPILE_CACHE_POLICY_POLICY_MANAGEMENT_MATCH_POLICY_EXACT_ONLY_H_
#define GRAPH_COMPILE_CACHE_POLICY_POLICY_MANAGEMENT_MATCH_POLICY_EXACT_ONLY_H_
#include "match_policy.h"
namespace ge {
class MatchPolicyExactOnly : public MatchPolicy {
public:
  virtual CacheItemId GetCacheItemId(const CCStatType &cc_state, const CompileCacheDesc &desc) const override;
  virtual ~MatchPolicyExactOnly() override = default;
};
}
#endif


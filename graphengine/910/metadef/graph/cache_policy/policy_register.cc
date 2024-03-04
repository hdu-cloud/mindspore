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

#include "graph/cache_policy/policy_register.h"
namespace ge {
PolicyRegister &PolicyRegister::GetInstance() {
  static PolicyRegister instance;
  return instance;
}

MatchPolicyRegister::MatchPolicyRegister(const MatchPolicyType match_policy_type, const MatchPolicyCreator &creator) {
  PolicyRegister::GetInstance().RegisterMatchPolicy(match_policy_type, creator);
}

AgingPolicyRegister::AgingPolicyRegister(const AgingPolicyType aging_policy_type, const AgingPolicyCreator &creator) {
  PolicyRegister::GetInstance().RegisterAgingPolicy(aging_policy_type, creator);
}
}  // namespace ge

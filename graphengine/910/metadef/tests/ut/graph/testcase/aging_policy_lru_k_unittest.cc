/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <gtest/gtest.h>
#include "graph/cache_policy/aging_policy_lru_k.h"
#include "cache_desc_stub/runtime_cache_desc.h"
#include "graph/cache_policy/cache_state.h"

namespace ge {
namespace {
CacheDescPtr CreateRuntimeCacheDesc(const std::vector<gert::Shape> &shapes) {
  auto cache_desc = std::make_shared<RuntimeCacheDesc>();
  cache_desc->SetShapes(shapes);
  return cache_desc;
}
void InsertCacheInfoQueue(CacheState &cache_state, uint16_t depth) {
  for (uint16_t i = 0; i < depth; ++i) {
    int64_t dim_0 = i;
    gert::Shape s{dim_0, 256, 256};
    auto cache_desc = CreateRuntimeCacheDesc({s});
    auto hash_key = cache_desc->GetCacheDescHash();
    (void) cache_state.AddCache(hash_key, cache_desc);
  }
}
}  // namespace
class AgingPolicyLruKUT : public testing::Test {};

TEST_F(AgingPolicyLruKUT, IsReadyToAddCache_ReturnFalse_CacheDescNotAppear2Times) {
  gert::Shape s1{256, 256};
  const std::vector<gert::Shape> shapes1{s1};
  auto cache_desc = CreateRuntimeCacheDesc(shapes1);
  auto hash = cache_desc->GetCacheDescHash();

  AgingPolicyLruK ag;
  EXPECT_EQ(ag.IsReadyToAddCache(hash, cache_desc), false);
}

TEST_F(AgingPolicyLruKUT, IsReadyToAddCache_ReturnTrue_CacheDescAppear2Times) {
  gert::Shape s1{256, 256};
  const std::vector<gert::Shape> shapes1{s1};
  auto cache_desc = CreateRuntimeCacheDesc(shapes1);
  auto hash = cache_desc->GetCacheDescHash();

  AgingPolicyLruK ag;
  EXPECT_EQ(ag.IsReadyToAddCache(hash, cache_desc), false);
  EXPECT_EQ(ag.IsReadyToAddCache(hash, cache_desc), true);
}

TEST_F(AgingPolicyLruKUT, IsReadyToAddCache_ReturnFalse_CacheDescNotMatched) {
  gert::Shape s1{256, 256};
  gert::Shape s2{1, 256, 256};
  const std::vector<gert::Shape> shapes1{s1};
  const std::vector<gert::Shape> shapes2{s2};
  auto cache_desc1 = CreateRuntimeCacheDesc(shapes1);
  auto cache_desc2 = CreateRuntimeCacheDesc(shapes2);
  auto hash2 = cache_desc2->GetCacheDescHash();

  AgingPolicyLruK ag;
  EXPECT_EQ(ag.IsReadyToAddCache(hash2, cache_desc1), false);
  EXPECT_EQ(ag.IsReadyToAddCache(hash2, cache_desc2), false);
}

TEST_F(AgingPolicyLruKUT, IsReadyToAddCache_ReturnTrue_HashCollisionButCacheDescMatched) {
  gert::Shape s1{256, 256};
  gert::Shape s2{1, 256, 256};
  const std::vector<gert::Shape> shapes1{s1};
  const std::vector<gert::Shape> shapes2{s2};
  auto cache_desc1 = CreateRuntimeCacheDesc(shapes1);
  auto cache_desc2 = CreateRuntimeCacheDesc(shapes2);
  auto hash2 = cache_desc2->GetCacheDescHash();

  AgingPolicyLruK ag;
  EXPECT_EQ(ag.IsReadyToAddCache(hash2, cache_desc1), false);
  EXPECT_EQ(ag.IsReadyToAddCache(hash2, cache_desc2), false);
  EXPECT_EQ(ag.IsReadyToAddCache(hash2, cache_desc2), true);
}

TEST_F(AgingPolicyLruKUT, DoAging_NoAgingId_CacheQueueNotReachDepth) {
  CacheState cache_state;
  uint16_t depth = 20;
  AgingPolicyLruK ag(depth);

  auto delete_ids = ag.DoAging(cache_state);
  EXPECT_EQ(delete_ids.size(), 0);

  InsertCacheInfoQueue(cache_state, depth);
  delete_ids = ag.DoAging(cache_state);
  EXPECT_EQ(delete_ids.size(), 0);
}

TEST_F(AgingPolicyLruKUT, DoAging_GetAgingIds_CacheQueueOverDepth) {
  CacheState cache_state;
  AgingPolicyLruK ag(20);
  auto delete_ids = ag.DoAging(cache_state);
  EXPECT_EQ(delete_ids.size(), 0);

  uint16_t depth = 21;
  InsertCacheInfoQueue(cache_state, depth);
  delete_ids = ag.DoAging(cache_state);
  ASSERT_EQ(delete_ids.size(), 1);
  EXPECT_EQ(delete_ids[0], 0);

  depth = 25;
  InsertCacheInfoQueue(cache_state, depth);
  delete_ids = ag.DoAging(cache_state);
  ASSERT_EQ(delete_ids.size(), 1);
  EXPECT_EQ(delete_ids[0], 0);
}
TEST_F(AgingPolicyLruKUT, DoAging_Aging5Times_CacheQueueDepthIs25) {
  CacheState cache_state;
  AgingPolicyLruK ag(20);
  auto delete_ids = ag.DoAging(cache_state);
  EXPECT_EQ(delete_ids.size(), 0);

  int16_t depth = 25;
  InsertCacheInfoQueue(cache_state, depth);

  for (size_t i = 0U; i < depth; ++i) {
    delete_ids = ag.DoAging(cache_state);
    if (i < 5U) {
      ASSERT_EQ(delete_ids.size(), 1);
      EXPECT_EQ(delete_ids[0], i);
    } else {
      EXPECT_EQ(delete_ids.size(), 0);
    }
    cache_state.DelCache(delete_ids);
  }
}
}  // namespace ge
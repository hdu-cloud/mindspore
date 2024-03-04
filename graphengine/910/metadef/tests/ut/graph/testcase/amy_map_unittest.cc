/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "graph/detail/any_map.h"
#include <gtest/gtest.h>

namespace ge {
class AnyMapUt : public testing::Test {};

TEST_F(AnyMapUt, GetNamesSuccess) {
  AnyMap am;
  std::set<std::string> names;
  am.Names(names);
  EXPECT_EQ(names.size(), 0);
  am.Set("name1", 10);
  am.Set("name2", 20);

  am.Names(names);
  EXPECT_EQ(names, std::set<std::string>({"name1", "name2"}));
}

TEST_F(AnyMapUt, TestHasFuncOk) {
  AnyMap am;
  EXPECT_FALSE(am.Has("name1"));
  EXPECT_FALSE(am.Has("name2"));

  am.Set("name1", 10);
  am.Set("name2", 20);

  EXPECT_TRUE(am.Has("name1"));
  EXPECT_TRUE(am.Has("name2"));
}

TEST_F(AnyMapUt, TestSwapFuncOk) {
  AnyMap am1, am2;
  am1.Set("name1", static_cast<int32_t>(10));
  am2.Set("name2", std::vector<int64_t>({10,20,30,40}));
  am1.Swap(am2);

  EXPECT_EQ(am1.Get<int32_t>("name1"), nullptr);
  auto p = am1.Get<std::vector<int64_t>>("name2");
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(*am1.Get<std::vector<int64_t>>("name2"), std::vector<int64_t>({10,20,30,40}));
}

TEST_F(AnyMapUt, GetOk) {
  AnyMap am;
  std::vector<int64_t> data = {1,2,3,4,5,6};
  std::vector<int64_t> ret;

  EXPECT_EQ(am.Get<std::vector<int64_t>>("Test"), nullptr);
  EXPECT_FALSE(am.Get("Test", ret));

  am.Set("Test", data);
  ASSERT_NE(am.Get<std::vector<int64_t>>("Test"), nullptr);
  EXPECT_EQ(*am.Get<std::vector<int64_t>>("Test"), data);

  EXPECT_TRUE(am.Get("Test", ret));
  EXPECT_EQ(ret, data);

}
}
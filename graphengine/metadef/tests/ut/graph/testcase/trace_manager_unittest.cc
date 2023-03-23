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
#include <gtest/gtest.h>
#include <memory>
#include <fstream>
#include <vector>
#include <iostream>
#include "mmpa/mmpa_api.h"
#define protected public
#define private public
#include "inc/common/util/trace_manager/trace_manager.h"

using namespace std;

namespace {
size_t GetFileLinesNum(const std::string fn) {
  std::ifstream f;
  f.open(fn, std::ios::in);
  size_t num = 0U;
  if (f.fail()) {
    return 0U;
  } else {
    std::string s;
    while (std::getline(f, s)) {
      num++;
    }
    f.close();
  }
  return num;
}
}  // namespace

namespace ge {
class UtestTraceManager : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestTraceManager, add_trace_basic_0) {
  auto &instance = TraceManager::GetInstance();
  instance.ClearTraceOwner();
  instance.SetTraceOwner("a", "b", "c");
  instance.trace_index_ = 0;
  EXPECT_EQ(instance.Initialize("."), SUCCESS);
  instance.enabled_ = true;
  instance.ClearTraceOwner();
  instance.SetTraceOwner("a", "b", "c");
  EXPECT_EQ(instance.trace_header_, "a:b");
  EXPECT_EQ(instance.graph_name_, "c");
  instance.AddTrace("0");
  EXPECT_EQ(instance.trace_index_, 1);
  for (int i = 0; i < 10000; i++) {
    instance.AddTrace(std::to_string(i + 1));
  }
  instance.Finalize();
  EXPECT_EQ(instance.current_file_saved_nums_, 10001);
  EXPECT_EQ(GetFileLinesNum(instance.current_saving_file_name_), 10001);
  std::string pre_file_name = instance.current_saving_file_name_;

  EXPECT_EQ(instance.Initialize("."), SUCCESS);
  instance.stopped_ = false;
  instance.current_file_saved_nums_ = 2000000U + 1U;
  for (int i = 0; i < 100; i++) {
    instance.AddTrace(std::to_string(i + 1));
  }
  instance.Finalize();

  EXPECT_NE(pre_file_name, instance.current_saving_file_name_);
  remove(instance.current_saving_file_name_.c_str());
  remove(pre_file_name.c_str());
}
}  // namespace ge

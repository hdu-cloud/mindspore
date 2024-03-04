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

#include "common/hyper_status.h"
#include <gtest/gtest.h>

namespace gert {
class HyperStatusUnittest : public testing::Test {};

TEST_F(HyperStatusUnittest, CreateMessageNullPtr) {
  va_list arg;
  EXPECT_EQ(CreateMessage(nullptr, arg), nullptr);
}

TEST_F(HyperStatusUnittest, CreateErrorStatusOk) {
  auto status = HyperStatus::ErrorStatus("This is a error message %s, int %d", "Hello", 10);
  ASSERT_FALSE(status.IsSuccess());
  EXPECT_EQ(strcmp(status.GetErrorMessage(), "This is a error message Hello, int 10"), 0);
}

TEST_F(HyperStatusUnittest, CreateSuccessStatusOk) {
  auto status = HyperStatus::Success();
  EXPECT_TRUE(status.IsSuccess());
}

TEST_F(HyperStatusUnittest, CopyAssignOk1) {
  auto error = HyperStatus::ErrorStatus("This is a error message %s, int %d", "Hello", 10);
  auto success = HyperStatus::Success();
  success = error;
  EXPECT_FALSE(success.IsSuccess());
  EXPECT_FALSE(error.IsSuccess());
  EXPECT_EQ(strcmp(success.GetErrorMessage(), "This is a error message Hello, int 10"), 0);
  EXPECT_NE(success.GetErrorMessage(), error.GetErrorMessage());
}

TEST_F(HyperStatusUnittest, CopyAssginOk2) {
  auto error = HyperStatus::ErrorStatus("This is a error message %s, int %d", "Hello", 10);
  auto success = HyperStatus::Success();
  error = success;
  EXPECT_TRUE(success.IsSuccess());
  EXPECT_TRUE(error.IsSuccess());
}

TEST_F(HyperStatusUnittest, CopyConstructOk) {
  auto error = HyperStatus::ErrorStatus("This is a error message %s, int %d", "Hello", 10);
  auto success = HyperStatus::Success();
  HyperStatus e1(error);
  HyperStatus s1(success);
  EXPECT_FALSE(e1.IsSuccess());
  EXPECT_TRUE(s1.IsSuccess());
  EXPECT_EQ(strcmp(e1.GetErrorMessage(), "This is a error message Hello, int 10"), 0);
}

TEST_F(HyperStatusUnittest, MoveConstructOk) {
  auto error = HyperStatus::ErrorStatus("This is a error message %s, int %d", "Hello", 10);
  auto success = HyperStatus::Success();
  HyperStatus e1(std::move(error));
  HyperStatus s1(std::move(success));
  EXPECT_FALSE(e1.IsSuccess());
  EXPECT_TRUE(s1.IsSuccess());
  EXPECT_EQ(strcmp(e1.GetErrorMessage(), "This is a error message Hello, int 10"), 0);
}

TEST_F(HyperStatusUnittest, MoveAssginOk1) {
  auto error = HyperStatus::ErrorStatus("This is a error message %s, int %d", "Hello", 10);
  auto success = HyperStatus::Success();
  error = std::move(success);
  EXPECT_TRUE(error.IsSuccess());
}

TEST_F(HyperStatusUnittest, MoveAssginOk2) {
  auto error = HyperStatus::ErrorStatus("This is a error message %s, int %d", "Hello", 10);
  auto success = HyperStatus::Success();
  success = std::move(error);
  ASSERT_FALSE(success.IsSuccess());
  EXPECT_EQ(strcmp(success.GetErrorMessage(), "This is a error message Hello, int 10"), 0);
}
}
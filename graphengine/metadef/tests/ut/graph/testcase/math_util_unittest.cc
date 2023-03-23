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
#include "graph/utils/math_util.h"
#include <gtest/gtest.h>
namespace ge {
class MathUtilUT : public testing::Test {};
TEST_F(MathUtilUT, AddOverflow_NotOverflow) {
  size_t i = 0;
  size_t j = 0;
  size_t ret;
  EXPECT_FALSE(AddOverflow(i, j, ret));
  EXPECT_EQ(ret, 0);

  i = 100;
  j = 200;
  EXPECT_FALSE(AddOverflow(i, j, ret));
  EXPECT_EQ(ret, 300);

  i = 0xFFFFFFFFFFFFFFFF;
  j = 0;
  EXPECT_FALSE(AddOverflow(i, j, ret));
  EXPECT_EQ(ret, 0xFFFFFFFFFFFFFFFF);

  i = 0x7FFFFFFFFFFFFFFF;
  j = 0x8000000000000000;
  EXPECT_FALSE(AddOverflow(i, j, ret));
  EXPECT_EQ(ret, 0xFFFFFFFFFFFFFFFF);
}
TEST_F(MathUtilUT, AddOverflow_Overflow) {

  size_t i = 0xFFFFFFFFFFFFFFFF;
  size_t j = 1;
  size_t ret;
  EXPECT_TRUE(AddOverflow(i, j, ret));

  i = 0x7FFFFFFFFFFFFFFF;
  j = 0x8000000000000001;
  EXPECT_TRUE(AddOverflow(i, j, ret));
}
TEST_F(MathUtilUT, AddOverflow_OverflowUint8) {
  uint8_t i = 255;
  uint8_t j = 0;
  uint8_t ret;
  EXPECT_FALSE(AddOverflow(i, j, ret));

  i = 255;
  j = 1;
  EXPECT_TRUE(AddOverflow(i, j, ret));

  i = 2;
  j = 254;
  EXPECT_TRUE(AddOverflow(i, j, ret));
}

TEST_F(MathUtilUT, AddOverflow_OverflowDiffType) {
  uint16_t i = 255;
  uint8_t j = 0;
  uint8_t ret;
  EXPECT_FALSE(AddOverflow(i, j, ret));
  EXPECT_FALSE(AddOverflow(j, i, ret));

  i = 256;
  j = 0;
  EXPECT_TRUE(AddOverflow(i, j, ret));
  EXPECT_TRUE(AddOverflow(j, i, ret));

  i = 100;
  j = 156;
  EXPECT_TRUE(AddOverflow(i, j, ret));
  EXPECT_TRUE(AddOverflow(j, i, ret));
}

TEST_F(MathUtilUT, AddOverflow_IntUnderflow) {
  int8_t i = -128;
  int8_t j = 0;
  int8_t ret;
  EXPECT_FALSE(AddOverflow(i, j, ret));
  EXPECT_FALSE(AddOverflow(j, i, ret));

  i = -128;
  j = -1;
  EXPECT_TRUE(AddOverflow(i, j, ret));
  EXPECT_TRUE(AddOverflow(j, i, ret));
}

TEST_F(MathUtilUT, AddOverflow_IntDiffTypeUnderflow) {
  int16_t i = -128;
  int8_t j = 0;
  int8_t ret;
  EXPECT_FALSE(AddOverflow(i, j, ret));
  EXPECT_FALSE(AddOverflow(j, i, ret));

  i = -129;
  j = 0;
  EXPECT_TRUE(AddOverflow(i, j, ret));
  EXPECT_TRUE(AddOverflow(j, i, ret));

  i = -128;
  j = -1;
  EXPECT_TRUE(AddOverflow(i, j, ret));
  EXPECT_TRUE(AddOverflow(j, i, ret));
}

TEST_F(MathUtilUT, RoundUp) {
  EXPECT_EQ(RoundUp(10, 8), 16);
  EXPECT_EQ(RoundUp(10, 3), 12);
  EXPECT_EQ(RoundUp(10, 2), 10);
  EXPECT_EQ(RoundUp(10, 1), 10);
}

TEST_F(MathUtilUT, CeilDiv16) {
  EXPECT_EQ(CeilDiv16(0), 0);
  EXPECT_EQ(CeilDiv16(1), 1);
  EXPECT_EQ(CeilDiv16(15), 1);
  EXPECT_EQ(CeilDiv16(16), 1);
  EXPECT_EQ(CeilDiv16(17), 2);
  EXPECT_EQ(CeilDiv16(32), 2);
  EXPECT_EQ(CeilDiv16(33), 3);
}

TEST_F(MathUtilUT, CeilDiv32) {
  EXPECT_EQ(CeilDiv32(0), 0);
  EXPECT_EQ(CeilDiv32(1), 1);
  EXPECT_EQ(CeilDiv32(31), 1);
  EXPECT_EQ(CeilDiv32(32), 1);
  EXPECT_EQ(CeilDiv32(33), 2);
  EXPECT_EQ(CeilDiv32(63), 2);
  EXPECT_EQ(CeilDiv32(64), 2);
  EXPECT_EQ(CeilDiv32(65), 3);
}

TEST_F(MathUtilUT, MulOverflow_NotOverflow) {
  int32_t i;
  EXPECT_FALSE(MulOverflow(10, 20, i));
  EXPECT_EQ(i, 200);

  EXPECT_FALSE(MulOverflow(-10, -20, i));
  EXPECT_EQ(i, 200);

  EXPECT_FALSE(MulOverflow(-10, 20, i));
  EXPECT_EQ(i, -200);

  EXPECT_FALSE(MulOverflow(0, 0, i));
  EXPECT_EQ(i, 0);
}

TEST_F(MathUtilUT, MulOverflow_Overflow) {
  int32_t i;
  EXPECT_TRUE(MulOverflow(std::numeric_limits<int32_t>::max(), 2, i));
  EXPECT_TRUE(MulOverflow(std::numeric_limits<int32_t>::min(), 2, i));
  EXPECT_TRUE(MulOverflow(std::numeric_limits<int32_t>::min(), -1, i));
  EXPECT_TRUE(MulOverflow(2, std::numeric_limits<int32_t>::max(), i));
  EXPECT_TRUE(MulOverflow(2, std::numeric_limits<int32_t>::min(), i));
  EXPECT_TRUE(MulOverflow(-1, std::numeric_limits<int32_t>::min(), i));
  EXPECT_TRUE(MulOverflow(std::numeric_limits<int32_t>::max() / 2 + 1, std::numeric_limits<int32_t>::max() / 2 + 1, i));
  EXPECT_TRUE(MulOverflow(std::numeric_limits<int32_t>::min() / 2 - 1, std::numeric_limits<int32_t>::min() / 2 - 1, i));
}

TEST_F(MathUtilUT, MulOverflow_OverflowUint8) {
  uint8_t i;
  EXPECT_TRUE(MulOverflow(static_cast<uint8_t>(255), static_cast<uint8_t>(2), i));
  EXPECT_TRUE(MulOverflow(static_cast<uint8_t>(2), static_cast<uint8_t>(255), i));
}

TEST_F(MathUtilUT, MulOverflow_OverflowDiffType) {
  uint8_t i;
  EXPECT_TRUE(MulOverflow(300, 1, i));
  EXPECT_TRUE(MulOverflow(1, 300, i));
}
}  // namespace ge
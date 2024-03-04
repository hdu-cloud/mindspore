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
#include "exe_graph/runtime/expand_dims_type.h"
#include <gtest/gtest.h>
namespace gert {
class ExpandDimsTypeUT : public testing::Test {};
TEST_F(ExpandDimsTypeUT, TestDoNotExpandSpecifyAll) {
  auto shape = Shape{2, 3, 5};
  ExpandDimsType edt("000");
  Shape out_shape;
  edt.Expand(shape, out_shape);
  ASSERT_EQ(out_shape, shape);
}
TEST_F(ExpandDimsTypeUT, TestDoNotExpandSpecifyPart1) {
  auto shape = Shape{2, 3, 5};
  ExpandDimsType edt("0");
  Shape out_shape;
  edt.Expand(shape, out_shape);
  ASSERT_EQ(out_shape, shape);
}
TEST_F(ExpandDimsTypeUT, TestDoNotExpandSpecifyPart2) {
  auto shape = Shape{2, 3, 5};
  ExpandDimsType edt("0");
  Shape out_shape;
  edt.Expand(shape, out_shape);
  ASSERT_EQ(out_shape, shape);
}
TEST_F(ExpandDimsTypeUT, TestDoNotExpandSpecifyNone) {
  auto shape = Shape{2, 3, 5};
  ExpandDimsType edt("");
  Shape out_shape;
  edt.Expand(shape, out_shape);
  ASSERT_EQ(out_shape, shape);
}

TEST_F(ExpandDimsTypeUT, ExpandAtHead) {
  auto shape = Shape{2, 16, 16};
  ExpandDimsType edt("11000");
  Shape out_shape;
  edt.Expand(shape, out_shape);

  ASSERT_EQ(5, out_shape.GetDimNum());
  ASSERT_EQ(out_shape, Shape({1, 1, 2, 16, 16}));
}
TEST_F(ExpandDimsTypeUT, ExpandAtHeadSpecifyPart) {
  auto shape = Shape{2, 16};
  ExpandDimsType edt("110");
  Shape out_shape;
  edt.Expand(shape, out_shape);

  ASSERT_EQ(4, out_shape.GetDimNum());
  ASSERT_EQ(out_shape, Shape({1, 1, 2, 16}));
}
TEST_F(ExpandDimsTypeUT, ExpandAtHeadSpecifyNone) {
  auto shape = Shape{2, 16, 16};
  ExpandDimsType edt("11");
  Shape out_shape;
  edt.Expand(shape, out_shape);

  ASSERT_EQ(5, out_shape.GetDimNum());
  ASSERT_EQ(out_shape, Shape({1, 1, 2, 16, 16}));
}
TEST_F(ExpandDimsTypeUT, ExpandAtHeadSpecifyNone1) {
  auto shape = Shape{2, 16, 16};
  ExpandDimsType edt("1");
  Shape out_shape;
  edt.Expand(shape, out_shape);

  ASSERT_EQ(4, out_shape.GetDimNum());
  ASSERT_EQ(out_shape, Shape({1, 2, 16, 16}));
}
TEST_F(ExpandDimsTypeUT, ExpandAtLast) {
  auto shape = Shape{2, 16, 16};
  ExpandDimsType edt("00011");
  Shape out_shape;
  edt.Expand(shape, out_shape);

  ASSERT_EQ(5, out_shape.GetDimNum());
  ASSERT_EQ(out_shape, Shape({2, 16, 16, 1, 1}));
}
TEST_F(ExpandDimsTypeUT, ExpandAtLast3Dim) {
  auto shape = Shape{2};
  ExpandDimsType edt("0111");
  Shape out_shape;
  edt.Expand(shape, out_shape);

  ASSERT_EQ(4, out_shape.GetDimNum());
  ASSERT_EQ(out_shape, Shape({2, 1, 1, 1}));
}
TEST_F(ExpandDimsTypeUT, ExpandHeadAndLast) {
  auto shape = Shape{2, 16, 16};
  ExpandDimsType edt("10001");
  Shape out_shape;
  edt.Expand(shape, out_shape);

  ASSERT_EQ(5, out_shape.GetDimNum());
  ASSERT_EQ(out_shape, Shape({1, 2, 16, 16, 1}));
}
TEST_F(ExpandDimsTypeUT, ExpandMiddle) {
  auto shape = Shape{2, 16, 16};
  ExpandDimsType edt("01010");
  Shape out_shape;
  edt.Expand(shape, out_shape);

  ASSERT_EQ(5, out_shape.GetDimNum());
  ASSERT_EQ(out_shape, Shape({2, 1, 16, 1, 16}));
}
TEST_F(ExpandDimsTypeUT, ExpandMiddleSpecifyPart) {
  auto shape = Shape{2, 16};
  ExpandDimsType edt("011");
  Shape out_shape;
  edt.Expand(shape, out_shape);

  ASSERT_EQ(4, out_shape.GetDimNum());
  ASSERT_EQ(out_shape, Shape({2, 1, 1, 16}));
}
TEST_F(ExpandDimsTypeUT, ExpandDimsMoreThanShape) {
  auto shape = Shape{2, 16};
  ExpandDimsType edt("1000");
  Shape out_shape;
  EXPECT_NE(edt.Expand(shape, out_shape), ge::GRAPH_SUCCESS);
}
TEST_F(ExpandDimsTypeUT, NullInput) {
  auto shape = Shape{2, 16, 16};
  ExpandDimsType edt(nullptr);
  Shape out_shape;
  edt.Expand(shape, out_shape);

  ASSERT_EQ(out_shape, Shape({2, 16, 16}));
}
TEST_F(ExpandDimsTypeUT, Over56Limits) {
  auto shape = Shape{2, 16, 16};
  std::string s;
  for (size_t i = 0; i <= 56; ++i) {
    s.push_back('1');
  }
  ExpandDimsType edt(s.c_str());
  Shape out_shape;
  edt.Expand(shape, out_shape);

  ASSERT_EQ(out_shape, Shape({2, 16, 16}));
}
TEST_F(ExpandDimsTypeUT, ExpandSpecifyPart) {
  auto shape = Shape{2, 16, 16};
  ExpandDimsType edt("100");
  Shape out_shape;
  edt.Expand(shape, out_shape);

  ASSERT_EQ(3, out_shape.GetDimNum());
  ASSERT_EQ(out_shape, Shape({2, 16, 16}));
}
TEST_F(ExpandDimsTypeUT, GetExpandDimMask) {
  ExpandDimsType edt("011");
  ASSERT_EQ(edt.GetFullSize(), 3);
  ASSERT_FALSE(edt.IsExpandIndex(0));
  ASSERT_TRUE(edt.IsExpandIndex(1));
  ASSERT_TRUE(edt.IsExpandIndex(2));
  ASSERT_EQ(edt.GetExpandDimsMask(), 6);
}
}  // namespace gert
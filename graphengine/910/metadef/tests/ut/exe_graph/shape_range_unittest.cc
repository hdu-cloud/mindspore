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
#include "exe_graph/runtime/range.h"
#include <gtest/gtest.h>
namespace gert {
class ShapeRangeUT : public testing::Test {};
TEST_F(ShapeRangeUT, DefaultConstructOk) {
  Range<Shape> sr;
  EXPECT_EQ(sr.GetMax(), nullptr);
  EXPECT_EQ(sr.GetMin(), nullptr);
}

TEST_F(ShapeRangeUT, ConstructFromListOk) {
  Shape min{8, 3, 224, 224};
  Shape max{-1, 10, 2240, 2240};
  Range<Shape> sr(&min, &max);
  auto s = sr.GetMin();
  EXPECT_EQ(s->GetDimNum(), 4);
  EXPECT_EQ(s->GetDim(0), 8);
  EXPECT_EQ(s->GetDim(1), 3);
  EXPECT_EQ(s->GetDim(2), 224);
  EXPECT_EQ(s->GetDim(3), 224);
  s = sr.GetMax();
  EXPECT_EQ(s->GetDimNum(), 4);
  EXPECT_EQ(s->GetDim(0), -1);
  EXPECT_EQ(s->GetDim(1), 10);
  EXPECT_EQ(s->GetDim(2), 2240);
  EXPECT_EQ(s->GetDim(3), 2240);
}

TEST_F(ShapeRangeUT, ConstructFromShapesOk) {
  Shape s1{8, 3, 224, 224};
  Shape s2{-1, 10, 2240, 2240};
  Range<Shape> sr(&s1, &s2);
  auto s = sr.GetMin();
  EXPECT_EQ(s->GetDimNum(), 4);
  EXPECT_EQ(s->GetDim(0), 8);
  EXPECT_EQ(s->GetDim(1), 3);
  EXPECT_EQ(s->GetDim(2), 224);
  EXPECT_EQ(s->GetDim(3), 224);
  s = sr.GetMax();
  EXPECT_EQ(s->GetDimNum(), 4);
  EXPECT_EQ(s->GetDim(0), -1);
  EXPECT_EQ(s->GetDim(1), 10);
  EXPECT_EQ(s->GetDim(2), 2240);
  EXPECT_EQ(s->GetDim(3), 2240);
}

TEST_F(ShapeRangeUT, ConstructFromSameShapeOk) {
  Shape s{8, 3, 224, 224};
  Range<Shape> sr(&s);
  auto min = sr.GetMin();
  EXPECT_EQ(s.GetDimNum(), 4);
  EXPECT_EQ(s.GetDim(0), 8);
  EXPECT_EQ(s.GetDim(1), 3);
  EXPECT_EQ(s.GetDim(2), 224);
  EXPECT_EQ(s.GetDim(3), 224);
  auto max = sr.GetMax();
  EXPECT_EQ(min, max);
}

TEST_F(ShapeRangeUT, EqualOk) {
  Shape s1{8, 3, 224, 224};
  Shape s2{8, 3, 224, 224};
  Range<Shape> sr1(&s1, &s2);
  Range<Shape> sr2(&s1, &s2);
  EXPECT_TRUE(sr1 == sr2);
}

TEST_F(ShapeRangeUT, SetMaxOk) {
  Range<Shape> sr;
  Shape max{7, 3, 224, 224};
  sr.SetMax(&max);
  EXPECT_EQ(*sr.GetMax(), max);
}

TEST_F(ShapeRangeUT, SetMinOk) {
  Range<Shape> sr;
  Shape min{7, 4, -1, -1};
  sr.SetMin(&min);
  EXPECT_EQ(sr.GetMin(), &min);
}

TEST_F(ShapeRangeUT, ModifyMaxOk) {
  Shape max{7, 4, -1, -1};
  Range<Shape> sr;
  sr.SetMax(&max);

  auto max_pr = sr.GetMax();
  (*max_pr)[0] = 8;
  (*max_pr)[1] = -1;
  EXPECT_EQ(sr.GetMax()->GetDim(0), 8);
  EXPECT_EQ(sr.GetMax()->GetDim(1), -1);
}

TEST_F(ShapeRangeUT, ModifyMinOk) {
  Shape min{3, 4, 255, 6};
  Range<Shape> sr;
  sr.SetMin(&min);

  auto min_pr = sr.GetMin();
  (*min_pr)[0] = 1;
  (*min_pr)[1] = 2;
  EXPECT_EQ(sr.GetMin()->GetDim(0), 1);
  EXPECT_EQ(sr.GetMin()->GetDim(1), 2);
}
}  // namespace gert
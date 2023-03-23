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
#include "exe_graph/runtime/infer_shape_range_context.h"
#include <gtest/gtest.h>
#include "faker/kernel_run_context_faker.h"
#include "exe_graph/runtime/storage_shape.h"
namespace gert {
class InferShapeRangeContextUT : public testing::Test {};
TEST_F(InferShapeRangeContextUT, GetInputShapeRangeOk) {
  Shape same_ele{8, 3, 224, 224};
  gert::Range<Shape> in_shape_range1(&same_ele);
  Shape min1{2, 2, 3, 8};
  Shape max1{2, -1, 3, 8};
  gert::Range<Shape> in_shape_range2(&min1, &max1);
  Shape out_shape1{8, 3, 224, 224};
  Shape out_shape2{8, 224, 224, 224};
  gert::Range<Shape> out_shape_range(&out_shape1, &out_shape2);
  auto context_holder = InferShapeRangeContextFaker()
      .IrInputNum(2)
      .NodeIoNum(2, 1)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
      .InputShapeRanges({&in_shape_range1, &in_shape_range2})
      .OutputShapeRanges({&out_shape_range})
      .Build();
  auto context = context_holder.GetContext<InferShapeRangeContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetInputShapeRange(0), nullptr);
  EXPECT_EQ(*context->GetInputShapeRange(0), in_shape_range1);

  ASSERT_NE(context->GetInputShapeRange(1), nullptr);
  EXPECT_EQ(*context->GetInputShapeRange(1), in_shape_range2);

  EXPECT_EQ(context->GetInputShapeRange(2), nullptr);
}

TEST_F(InferShapeRangeContextUT, GetDynamicInputShapeRangeOk) {
  Shape min1{8, 3, 224, 224};
  Shape max1{-1, 3, 224, 224};
  gert::Range<Shape> in_shape_range1(&min1, &max1);
  Shape min2{2, 2, 3, 8};
  Shape max2{2, -1, 3, 8};
  gert::Range<Shape> in_shape_range2(&min2, &max2);
  Shape min3{3, 2, 3, 8};
  Shape max3{3, 2, 9, 8};
  gert::Range<Shape> in_shape_range3(&min3, &max3);
  Shape min4{4, 2, 3, 8};
  Shape max4{4, 2, 3, 16};
  gert::Range<Shape> in_shape_range4(&min4, &max4);
  Shape min5{8, 3, 224, 224};
  Shape max5{-1, 3, 224, 224};
  gert::Range<Shape> out_shape_range(&min5, &max5);
  auto context_holder = InferShapeRangeContextFaker()
      .IrInstanceNum({1, 2, 0, 1})
      .NodeIoNum(4, 1)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
      .InputShapeRanges({&in_shape_range1, &in_shape_range2, &in_shape_range3, &in_shape_range4})
      .OutputShapeRanges({&out_shape_range})
      .Build();
  auto context = context_holder.GetContext<InferShapeRangeContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetOptionalInputShapeRange(0), nullptr);
  EXPECT_EQ(*context->GetOptionalInputShapeRange(0), in_shape_range1);

  ASSERT_NE(context->GetDynamicInputShapeRange(1, 0), nullptr);
  EXPECT_EQ(*context->GetDynamicInputShapeRange(1, 0), in_shape_range2);

  ASSERT_NE(context->GetDynamicInputShapeRange(1, 1), nullptr);
  EXPECT_EQ(*context->GetDynamicInputShapeRange(1, 1), in_shape_range3);

  EXPECT_EQ(context->GetOptionalInputShapeRange(2), nullptr);

  ASSERT_NE(context->GetOptionalInputShapeRange(3), nullptr);
  EXPECT_EQ(*context->GetOptionalInputShapeRange(3), in_shape_range4);
}

TEST_F(InferShapeRangeContextUT, GetOutShapeOk) {
  Shape same_ele{8, 3, 224, 224};
  gert::Range<Shape> in_shape_range1(&same_ele);
  Shape min2{2, 2, 3, 8};
  Shape max2{2, -1, 3, 8};
  gert::Range<Shape> in_shape_range2(&min2, &max2);
  Shape out_min{8, 3, 224, 224};
  Shape out_max{8, 224, 224, 224};
  gert::Range<Shape> out_shape_range(&out_min, &out_max);
  auto context_holder = InferShapeRangeContextFaker()
      .IrInputNum(2)
      .NodeIoNum(2, 1)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
      .InputShapeRanges({&in_shape_range1, &in_shape_range2})
      .OutputShapeRanges({&out_shape_range})
      .Build();
  auto context = context_holder.GetContext<InferShapeRangeContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetOutputShapeRange(0), nullptr);
  EXPECT_EQ(*context->GetOutputShapeRange(0), out_shape_range);

  EXPECT_EQ(context->GetOutputShapeRange(1), nullptr);
}
}  // namespace gert
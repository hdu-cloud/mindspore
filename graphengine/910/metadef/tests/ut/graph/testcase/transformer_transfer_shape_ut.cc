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

#include <gtest/gtest.h>
#define private public
#include "graph/ge_tensor.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "axis_util.h"
#include "expand_dimension.h"
#include "transfer_shape_according_to_format.h"
#include "transfer_range_according_to_format.h"

using namespace ge;

namespace transformer {
class TransformerTransferShapeUT : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

  void RunTransferShape(const ge::Format &origin_format, const ge::Format &format, const ge::DataType &dtype,
                        const bool &expect_ret, const vector<int64_t> &dims, const vector<int64_t> &expect_dim,
                        bool only_test_first_interface = false) {
    std::cout << "RunTransferShape: origin_format=" << origin_format << ", format=" << format << ", dtype=" << dtype
              << ", dim size=" << dims.size() << std::endl;
    ge::GeShape shape(dims);
    ShapeAndFormat shape_and_format_info {shape, origin_format, format, dtype};
    ShapeTransferAccordingToFormat shape_transfer;
    bool ret = shape_transfer.GetShapeAccordingToFormat(shape_and_format_info);
    EXPECT_EQ(ret, expect_ret);
    if (ret) {
      EXPECT_EQ(shape.GetDims(), expect_dim);
    }
    if (only_test_first_interface) {
      return;
    }
    gert::Shape current_shape;
    for (const int64_t &d : dims) {
      current_shape.AppendDim(d);
    }
    gert::Shape ret_shape;
    ret = shape_transfer.TransferShape(origin_format, format, dtype, current_shape, ret_shape);
    if (ret && dims != expect_dim) {
      vector<int64_t> new_dim;
      for (size_t i = 0; i < ret_shape.GetDimNum(); ++i) {
        new_dim.push_back(ret_shape.GetDim(i));
      }
      EXPECT_EQ(new_dim, expect_dim);
    }

    ret = shape_transfer.TransferShape(origin_format, format, dtype, current_shape);
    EXPECT_EQ(ret, expect_ret);
    if (ret) {
      vector<int64_t> new_dim;
      for (size_t i = 0; i < current_shape.GetDimNum(); ++i) {
        new_dim.push_back(current_shape.GetDim(i));
      }
      EXPECT_EQ(new_dim, expect_dim);
    }
    ExtAxisValue ext_axis;
    shape_transfer.InitExtAxisValue(nullptr, ext_axis);
    ge::GeShape src_shape(dims);
    ge::GeShape dst_shape;
    ret = shape_transfer.TransferShape(origin_format, format, dtype, ext_axis, src_shape, dst_shape);
    EXPECT_EQ(ret, expect_ret);
    if (ret && dims != expect_dim) {
      EXPECT_EQ(dst_shape.GetDims(), expect_dim);
    }

    ret = shape_transfer.TransferShape(origin_format, format, dtype, ext_axis, src_shape);
    EXPECT_EQ(ret, expect_ret);
    if (ret) {
      EXPECT_EQ(src_shape.GetDims(), expect_dim);
    }
  }

  void RunTransferShape(const ge::OpDescPtr &op_desc, const ge::Format &origin_format, const ge::Format &format,
                        const ge::DataType &dtype, const bool &expect_ret, const vector<int64_t> &dims,
                        const vector<int64_t> &expect_dim, bool only_test_first_interface = false,
                        const int64_t &m0_val = 16) {
    std::cout << "RunTransferShape: origin_format=" << origin_format << ", format=" << format << ", dtype=" << dtype
              << ", dim size=" << dims.size() << std::endl;
    ge::GeShape shape(dims);
    ShapeAndFormat shape_and_format_info {shape, origin_format, format, dtype};
    ShapeTransferAccordingToFormat shape_transfer;
    bool ret = shape_transfer.GetShapeAccordingToFormat(op_desc, shape_and_format_info);
    EXPECT_EQ(ret, expect_ret);
    if (ret) {
      EXPECT_EQ(shape.GetDims(), expect_dim);
    }
    if (only_test_first_interface) {
      return;
    }

    gert::Shape current_shape;
    for (const int64_t &d : dims) {
      current_shape.AppendDim(d);
    }

    gert::Shape ret_shape;
    ret = shape_transfer.TransferShape(origin_format, format, dtype, current_shape, ret_shape, op_desc);
    if (ret && dims != expect_dim) {
      vector<int64_t> new_dim;
      for (size_t i = 0; i < ret_shape.GetDimNum(); ++i) {
        new_dim.push_back(ret_shape.GetDim(i));
      }
      EXPECT_EQ(new_dim, expect_dim);
    }

    ret = shape_transfer.TransferShape(origin_format, format, dtype, current_shape, op_desc);
    EXPECT_EQ(ret, expect_ret);
    if (ret) {
      vector<int64_t> new_dim;
      for (size_t i = 0; i < current_shape.GetDimNum(); ++i) {
        new_dim.push_back(current_shape.GetDim(i));
      }
      EXPECT_EQ(new_dim, expect_dim);
    }

    ExtAxisValue ext_axis;
    shape_transfer.InitExtAxisValue(op_desc, ext_axis);
    ext_axis[3] = m0_val;
    ge::GeShape src_shape(dims);
    ge::GeShape dst_shape;
    ret = shape_transfer.TransferShape(origin_format, format, dtype, ext_axis, src_shape, dst_shape);
    EXPECT_EQ(ret, expect_ret);
    if (ret && dims != expect_dim) {
      EXPECT_EQ(dst_shape.GetDims(), expect_dim);
    }

    ret = shape_transfer.TransferShape(origin_format, format, dtype, ext_axis, src_shape);
    EXPECT_EQ(ret, expect_ret);
    if (ret) {
      EXPECT_EQ(src_shape.GetDims(), expect_dim);
    }
  }

  void RunTransferShapeWithExtAxis(const ge::OpDescPtr &op_desc, const ge::Format &origin_format, const ge::Format &format,
                        const ge::DataType &dtype, const bool &expect_ret, const vector<int64_t> &dims,
                        const vector<int64_t> &expect_dim, const int64_t &m0_val = 16) {
    std::cout << "RunTransferShape: origin_format=" << origin_format << ", format=" << format << ", dtype=" << dtype
              << ", dim size=" << dims.size() << ", m0 value=" << m0_val << std::endl;

    ShapeTransferAccordingToFormat shape_transfer;
    ExtAxisValue ext_axis;
    shape_transfer.InitExtAxisValue(op_desc, ext_axis);
    ext_axis[3] = m0_val;
    ge::GeShape src_shape(dims);
    ge::GeShape dst_shape;
    bool ret = shape_transfer.TransferShape(origin_format, format, dtype, ext_axis, src_shape, dst_shape);
    EXPECT_EQ(ret, expect_ret);
    if (ret && dims != expect_dim) {
      EXPECT_EQ(dst_shape.GetDims(), expect_dim);
    }

    ret = shape_transfer.TransferShape(origin_format, format, dtype, ext_axis, src_shape);
    EXPECT_EQ(ret, expect_ret);
    if (ret) {
      EXPECT_EQ(src_shape.GetDims(), expect_dim);
    }
  }
};

TEST_F(TransformerTransferShapeUT, transfer_shape_verify_param) {
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NHWC, DT_FLOAT16, true, {}, {});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_ND, DT_INT8, true, {3, 4, 5, 6}, {3, 4, 5, 6});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_NHWC, DT_FLOAT, true, {3, 4, 5, 6}, {3, 4, 5, 6});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_NCDHW, DT_INT32, true, {3, 4, 5, 6}, {3, 4, 5, 6});

  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NHWC, DT_UNDEFINED, false, {3, 4, 5, 6}, {3, 4, 5, 6});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NHWC, DT_MAX, false, {3, 4, 5, 6}, {3, 4, 5, 6});
  RunTransferShape(ge::FORMAT_RESERVED, ge::FORMAT_NHWC, DT_FLOAT16, false, {3, 4, 5, 6}, {3, 4, 5, 6});
  RunTransferShape(ge::FORMAT_END, ge::FORMAT_NHWC, DT_FLOAT16, false, {3, 4, 5, 6}, {3, 4, 5, 6});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_RESERVED, DT_FLOAT16, false, {3, 4, 5, 6}, {3, 4, 5, 6});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_END, DT_FLOAT16, false, {3, 4, 5, 6}, {3, 4, 5, 6});
}

TEST_F(TransformerTransferShapeUT, transfer_shape_from_nchw) {
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NCHW, DT_FLOAT16, true, {}, {});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NHWC, DT_FLOAT16, true, {}, {});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NHWC, DT_FLOAT, true, {5}, {5});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NCHW, DT_INT64, true, {5, 6}, {5, 6});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NCHW, DT_UINT8, true, {5, 6, 7}, {5, 6, 7});

  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NCHW, DT_UINT8, true, {5, 6, 7, 8}, {5, 6, 7, 8});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NHWC, DT_INT8, true, {5, 6, 7, 8}, {5, 7, 8, 6});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_HWCN, DT_UINT16, true, {5, 6, 7, 8}, {7, 8, 6, 5});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_CHWN, DT_INT16, true, {5, 6, 7, 8}, {6, 7, 8, 5});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NCHW, DT_UINT32, true, {5, 6, 7, 8}, {5, 6, 7, 8});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NHWC, DT_INT32, true, {5, 6, 7, 8}, {5, 7, 8, 6});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_HWCN, DT_FLOAT, true, {5, 6, 7, 8}, {7, 8, 6, 5});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_CHWN, DT_FLOAT16, true, {5, 6, 7, 8}, {6, 7, 8, 5});

  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_UINT8, true, {8, 512, 5, 5}, {8, 16, 5, 5, 32});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT8, true, {8, 512, 5, 5}, {8, 16, 5, 5, 32});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_UINT16, true, {8, 512, 5, 5}, {8, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT16, true, {8, 512, 5, 5}, {8, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_UINT32, true, {8, 512, 5, 5}, {8, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT32, true, {8, 512, 5, 5}, {8, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_FLOAT, true, {8, 512, 5, 5}, {8, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_FLOAT16, true, {8, 512, 5, 5}, {8, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_UINT1, true, {8, 512, 5, 5}, {8, 2, 5, 5, 256});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_UINT2, true, {8, 512, 5, 5}, {8, 4, 5, 5, 128});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT2, true, {8, 512, 5, 5}, {8, 4, 5, 5, 128});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT4, true, {8, 512, 5, 5}, {8, 8, 5, 5, 64});

  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_UINT8, true, {18, 512, 5, 5}, {16, 5, 5, 2, 32, 32});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_INT8, true, {18, 512, 5, 5}, {16, 5, 5, 2, 32, 32});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_UINT16, true, {18, 512, 5, 5}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_INT16, true, {18, 512, 5, 5}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_UINT32, true, {18, 512, 5, 5}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_INT32, true, {18, 512, 5, 5}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_FLOAT, true, {18, 512, 5, 5}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_FLOAT16, true, {18, 512, 5, 5}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_UINT1, true, {18, 512, 5, 5}, {2, 5, 5, 2, 256, 256});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_UINT2, true, {18, 512, 5, 5}, {4, 5, 5, 2, 128, 128});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_INT2, true, {18, 512, 5, 5}, {4, 5, 5, 2, 128, 128});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, DT_INT4, true, {18, 512, 5, 5}, {8, 5, 5, 2, 64, 64});

  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_UINT8, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_INT8, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_UINT16, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_INT16, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_UINT32, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_INT32, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_FLOAT, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_FLOAT16, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_UINT1, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_UINT2, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_INT2, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_INT4, true, {8, 512, 5, 5}, {8, 128, 5, 5, 4});

  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_UINT8, true, {48, 512, 5, 5}, {400, 3, 16, 32});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_INT8, true, {48, 512, 5, 5}, {400, 3, 16, 32});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_UINT16, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_INT16, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_UINT32, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_INT32, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_UINT1, true, {48, 512, 5, 5}, {50, 3, 16, 256});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_UINT2, true, {48, 512, 5, 5}, {100, 3, 16, 128});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_INT2, true, {48, 512, 5, 5}, {100, 3, 16, 128});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_INT4, true, {48, 512, 5, 5}, {200, 3, 16, 64});

  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_UINT8, true, {48, 3, 5, 5}, {4, 3, 16, 32});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_INT8, true, {48, 3, 5, 5}, {4, 3, 16, 32});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_UINT16, true, {48, 3, 5, 5}, {7, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_INT16, true, {48, 3, 5, 5}, {7, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_UINT32, true, {48, 3, 5, 5}, {7, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_INT32, true, {48, 3, 5, 5}, {7, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_FLOAT, true, {48, 3, 5, 5}, {7, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_FLOAT16, true, {48, 3, 5, 5}, {7, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_UINT1, true, {48, 3, 5, 5}, {1, 3, 16, 256});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_UINT2, true, {48, 3, 5, 5}, {1, 3, 16, 128});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_INT2, true, {48, 3, 5, 5}, {1, 3, 16, 128});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_INT4, true, {48, 3, 5, 5}, {2, 3, 16, 64});

  int32_t group = 16;
  ge::Format target_format = static_cast<ge::Format>(GetFormatFromSub(static_cast<int32_t>(ge::FORMAT_FRACTAL_Z), group));
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_UINT8, true, {48, 512, 5, 5}, {6400, 3, 16, 32});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_INT8, true, {48, 512, 5, 5}, {6400, 3, 16, 32});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_UINT16, true, {48, 512, 5, 5}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_INT16, true, {48, 512, 5, 5}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_UINT32, true, {48, 512, 5, 5}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_INT32, true, {48, 512, 5, 5}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_FLOAT, true, {48, 512, 5, 5}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_FLOAT16, true, {48, 512, 5, 5}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_UINT1, true, {48, 512, 5, 5}, {800, 3, 16, 256});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_UINT2, true, {48, 512, 5, 5}, {1600, 3, 16, 128});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_INT2, true, {48, 512, 5, 5}, {1600, 3, 16, 128});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_INT4, true, {48, 512, 5, 5}, {3200, 3, 16, 64});
}

TEST_F(TransformerTransferShapeUT, transfer_shape_from_nchw_unknow_shape) {
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT16, true, {-1, 512, 5, 5}, {-1, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT16, true, {-1, 512, -1, 5}, {-1, 32, -1, 5, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT16, true, {8, -1, 5, 5}, {8, -1, 5, 5, 16});

  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_INT16, true, {-1, 33, 5, 5}, {-1, 9, 5, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_INT16, true, {-1, 33, -1, 5}, {-1, 9, -1, 5, 4});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0_C04, DT_INT16, true, {8, -1, 5, 5}, {8, -1, 5, 5, 4});

  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {-1, 33, 5, 5}, {75, -1, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, -1, 5, 5}, {-1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, -1, -1, 5}, {-1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, -1, 5, -1}, {-1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, 512, -1, 5}, {-1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, 512, 5, -1}, {-1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, 512, -1, -1}, {-1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, -1, -1, -1}, {-1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {-1, -1, -1, -1}, {-1, -1, 16, 16});

  int32_t group = 16;
  ge::Format target_format = static_cast<ge::Format>(GetFormatFromSub(static_cast<int32_t>(ge::FORMAT_FRACTAL_Z), group));
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_FLOAT16, true, {48, 512, -1, 5}, {-1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_FLOAT16, true, {-1, 512, 5, 5}, {800, -1, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_FLOAT16, true, {48, -1, 5, 5}, {-1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, target_format, DT_FLOAT16, true, {-1, -1, 5, 5}, {-1, -1, 16, 16});

  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_FLOAT16, true, {-1, 3, 5, 5}, {7, -1, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_FLOAT16, true, {48, -1, 5, 5}, {7, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_FLOAT16, true, {48, -1, -1, 5}, {-1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_FLOAT16, true, {48, -1, 5, -1}, {-1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_FLOAT16, true, {48, 3, -1, 5}, {-1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_FLOAT16, true, {48, 3, 5, -1}, {-1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_FLOAT16, true, {48, 3, -1, -1}, {-1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_FLOAT16, true, {48, -1, -1, -1}, {-1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z_C04, DT_FLOAT16, true, {-1, -1, -1, -1}, {-1, -1, 16, 16});
}

TEST_F(TransformerTransferShapeUT, transfer_shape_from_hwcn) {
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NCHW, DT_FLOAT16, true, {}, {});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_HWCN, DT_FLOAT16, true, {}, {});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NHWC, DT_FLOAT, true, {5}, {5});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NCHW, DT_INT64, true, {5, 6}, {5, 6});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NCHW, DT_UINT8, true, {5, 6, 7}, {5, 6, 7});

  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NCHW, DT_UINT8, true, {7, 8, 6, 5}, {5, 6, 7, 8});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NHWC, DT_INT8, true, {7, 8, 6, 5}, {5, 7, 8, 6});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_HWCN, DT_UINT16, true, {7, 8, 6, 5}, {7, 8, 6, 5});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_CHWN, DT_INT16, true, {7, 8, 6, 5}, {6, 7, 8, 5});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NCHW, DT_UINT32, true, {7, 8, 6, 5}, {5, 6, 7, 8});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NHWC, DT_INT32, true, {7, 8, 6, 5}, {5, 7, 8, 6});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_HWCN, DT_FLOAT, true, {7, 8, 6, 5}, {7, 8, 6, 5});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_CHWN, DT_FLOAT16, true, {7, 8, 6, 5}, {6, 7, 8, 5});

  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_UINT8, true, {5, 5, 512, 8}, {8, 16, 5, 5, 32});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_INT8, true, {5, 5, 512, 8}, {8, 16, 5, 5, 32});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_UINT16, true, {5, 5, 512, 8}, {8, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_INT16, true, {5, 5, 512, 8}, {8, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_UINT32, true, {5, 5, 512, 8}, {8, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_INT32, true, {5, 5, 512, 8}, {8, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_FLOAT, true, {5, 5, 512, 8}, {8, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_FLOAT16, true, {5, 5, 512, 8}, {8, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_UINT1, true, {5, 5, 512, 8}, {8, 2, 5, 5, 256});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_UINT2, true, {5, 5, 512, 8}, {8, 4, 5, 5, 128});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_INT2, true, {5, 5, 512, 8}, {8, 4, 5, 5, 128});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, DT_INT4, true, {5, 5, 512, 8}, {8, 8, 5, 5, 64});

  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_UINT8, true, {5, 5, 512, 18}, {16, 5, 5, 2, 32, 32});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_INT8, true, {5, 5, 512, 18}, {16, 5, 5, 2, 32, 32});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_UINT16, true, {5, 5, 512, 18}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_INT16, true, {5, 5, 512, 18}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_UINT32, true, {5, 5, 512, 18}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_INT32, true, {5, 5, 512, 18}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_FLOAT, true, {5, 5, 512, 18}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_FLOAT16, true, {5, 5, 512, 18}, {32, 5, 5, 2, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_UINT1, true, {5, 5, 512, 18}, {2, 5, 5, 2, 256, 256});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_UINT2, true, {5, 5, 512, 18}, {4, 5, 5, 2, 128, 128});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_INT2, true, {5, 5, 512, 18}, {4, 5, 5, 2, 128, 128});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, DT_INT4, true, {5, 5, 512, 18}, {8, 5, 5, 2, 64, 64});

  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_UINT8, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_INT8, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_UINT16, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_INT16, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_UINT32, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_INT32, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_FLOAT, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_FLOAT16, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_UINT1, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_UINT2, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_INT2, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0_C04, DT_INT4, true, {5, 5, 512, 8}, {8, 128, 5, 5, 4});

  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_UINT8, true, {5, 5, 512, 48}, {400, 3, 16, 32});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_INT8, true, {5, 5, 512, 48}, {400, 3, 16, 32});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_UINT16, true, {5, 5, 512, 48}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_INT16, true, {5, 5, 512, 48}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_UINT32, true, {5, 5, 512, 48}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_INT32, true, {5, 5, 512, 48}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_FLOAT, true, {5, 5, 512, 48}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {5, 5, 512, 48}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_UINT1, true, {5, 5, 512, 48}, {50, 3, 16, 256});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_UINT2, true, {5, 5, 512, 48}, {100, 3, 16, 128});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_INT2, true, {5, 5, 512, 48}, {100, 3, 16, 128});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, DT_INT4, true, {5, 5, 512, 48}, {200, 3, 16, 64});

  int32_t group = 16;
  ge::Format target_format = static_cast<ge::Format>(GetFormatFromSub(static_cast<int32_t>(ge::FORMAT_FRACTAL_Z), group));
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_UINT8, true, {5, 5, 512, 48}, {6400, 3, 16, 32});
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_INT8, true, {5, 5, 512, 48}, {6400, 3, 16, 32});
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_UINT16, true, {5, 5, 512, 48}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_INT16, true, {5, 5, 512, 48}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_UINT32, true, {5, 5, 512, 48}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_INT32, true, {5, 5, 512, 48}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_FLOAT, true, {5, 5, 512, 48}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_FLOAT16, true, {5, 5, 512, 48}, {12800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_UINT1, true, {5, 5, 512, 48}, {800, 3, 16, 256});
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_UINT2, true, {5, 5, 512, 48}, {1600, 3, 16, 128});
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_INT2, true, {5, 5, 512, 48}, {1600, 3, 16, 128});
  RunTransferShape(ge::FORMAT_HWCN, target_format, DT_INT4, true, {5, 5, 512, 48}, {3200, 3, 16, 64});
}

TEST_F(TransformerTransferShapeUT, transfer_shape_from_ncdhw) {
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_FLOAT16, true, {}, {});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_FLOAT16, true, {}, {});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_FLOAT, true, {5}, {5});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_INT64, true, {5, 6}, {5, 6});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_UINT8, true, {5, 6, 7}, {5, 6, 7});

  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_UINT8, true, {8, 512, 9, 5, 5}, {8, 9, 16, 5, 5, 32});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_INT8, true, {8, 512, 9, 5, 5}, {8, 9, 16, 5, 5, 32});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_UINT16, true, {8, 512, 9, 5, 5}, {8, 9, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_INT16, true, {8, 512, 9, 5, 5}, {8, 9, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_UINT32, true, {8, 512, 9, 5, 5}, {8, 9, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_INT32, true, {8, 512, 9, 5, 5}, {8, 9, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_FLOAT, true, {8, 512, 9, 5, 5}, {8, 9, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_FLOAT16, true, {8, 512, 9, 5, 5}, {8, 9, 32, 5, 5, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_UINT1, true, {8, 512, 9, 5, 5}, {8, 9, 2, 5, 5, 256});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_UINT2, true, {8, 512, 9, 5, 5}, {8, 9, 4, 5, 5, 128});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_INT2, true, {8, 512, 9, 5, 5}, {8, 9, 4, 5, 5, 128});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, DT_INT4, true, {8, 512, 9, 5, 5}, {8, 9, 8, 5, 5, 64});

  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_UINT8, true, {48, 512, 3, 5, 5}, {1200, 3, 16, 32});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_INT8, true, {48, 512, 3, 5, 5}, {1200, 3, 16, 32});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_UINT16, true, {48, 512, 3, 5, 5}, {2400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_INT16, true, {48, 512, 3, 5, 5}, {2400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_UINT32, true, {48, 512, 3, 5, 5}, {2400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_INT32, true, {48, 512, 3, 5, 5}, {2400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_FLOAT, true, {48, 512, 3, 5, 5}, {2400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_FLOAT16, true, {48, 512, 3, 5, 5}, {2400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_UINT1, true, {48, 512, 3, 5, 5}, {150, 3, 16, 256});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_UINT2, true, {48, 512, 3, 5, 5}, {300, 3, 16, 128});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_INT2, true, {48, 512, 3, 5, 5}, {300, 3, 16, 128});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, DT_INT4, true, {48, 512, 3, 5, 5}, {600, 3, 16, 64});

  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_UINT8, true, {90, 512, 3, 5, 5}, {450, 16, 16, 32});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_INT8, true, {90, 512, 3, 5, 5}, {450, 16, 16, 32});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_UINT16, true, {90, 512, 3, 5, 5}, {450, 32, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_INT16, true, {90, 512, 3, 5, 5}, {450, 32, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_UINT32, true, {90, 512, 3, 5, 5}, {450, 32, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_INT32, true, {90, 512, 3, 5, 5}, {450, 32, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_FLOAT, true, {90, 512, 3, 5, 5}, {450, 32, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_FLOAT16, true, {90, 512, 3, 5, 5}, {450, 32, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_UINT1, true, {90, 512, 3, 5, 5}, {450, 2, 16, 256});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_UINT2, true, {90, 512, 3, 5, 5}, {450, 4, 16, 128});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_INT2, true, {90, 512, 3, 5, 5}, {450, 4, 16, 128});
  RunTransferShape(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE, DT_INT4, true, {90, 512, 3, 5, 5}, {450, 8, 16, 64});

  int32_t group = 16;
  ge::Format target_format = static_cast<ge::Format>(GetFormatFromSub(static_cast<int32_t>(ge::FORMAT_FRACTAL_Z_3D), group));
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_UINT8, true, {48, 512, 3, 5, 5}, {19200, 3, 16, 32});
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_INT8, true, {48, 512, 3, 5, 5}, {19200, 3, 16, 32});
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_UINT16, true, {48, 512, 3, 5, 5}, {38400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_INT16, true, {48, 512, 3, 5, 5}, {38400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_UINT32, true, {48, 512, 3, 5, 5}, {38400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_INT32, true, {48, 512, 3, 5, 5}, {38400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_FLOAT, true, {48, 512, 3, 5, 5}, {38400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_FLOAT16, true, {48, 512, 3, 5, 5}, {38400, 3, 16, 16});
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_UINT1, true, {48, 512, 3, 5, 5}, {2400, 3, 16, 256});
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_UINT2, true, {48, 512, 3, 5, 5}, {4800, 3, 16, 128});
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_INT2, true, {48, 512, 3, 5, 5}, {4800, 3, 16, 128});
  RunTransferShape(ge::FORMAT_NCDHW, target_format, DT_INT4, true, {48, 512, 3, 5, 5}, {9600, 3, 16, 64});
}

TEST_F(TransformerTransferShapeUT, transfer_shape_from_4d_to_6hd) {
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NDC1HWC0, DT_FLOAT16, true, {4, 33, 7, 7}, {4, 1, 3, 7, 7, 16});
  RunTransferShape(ge::FORMAT_NHWC, ge::FORMAT_NDC1HWC0, DT_FLOAT16, true, {4, 7, 7, 33}, {4, 1, 3, 7, 7, 16});
  RunTransferShape(ge::FORMAT_HWCN, ge::FORMAT_NDC1HWC0, DT_FLOAT16, true, {7, 7, 33, 4}, {4, 1, 3, 7, 7, 16});
}

TEST_F(TransformerTransferShapeUT, transfer_shape_from_nd_to_nz) {
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT16, true, {34}, {1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT16, true, {34, 1}, {1, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT, true, {18, 34}, {3, 2, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT8, true, {1, 18, 34}, {1, 2, 2, 16, 32});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT8, true, {1, 18, 34}, {1, 2, 2, 16, 32});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT16, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT16, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT32, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT32, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT16, true, {1, 18, 34}, {1, 3, 2, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT1, true, {1, 18, 134}, {1, 1, 2, 16, 256});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT2, true, {1, 18, 134}, {1, 2, 2, 16, 128});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT2, true, {1, 18, 134}, {1, 2, 2, 16, 128});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT4, true, {1, 18, 134}, {1, 3, 2, 16, 64});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT16, true, {-2}, {-2}, true);
  RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_NZ, DT_FLOAT16, true, {8, 1000}, {63, 1, 16, 16});

  transformer::TransferShapeUtils::m0_list_.fill(1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT16, true, {34}, {1, 34, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT16, true, {34, 1}, {1, 34, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT, true, {18, 34}, {3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT8, true, {1, 18, 34}, {1, 2, 18, 1, 32}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT8, true, {1, 18, 34}, {1, 2, 18, 1, 32}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT16, true, {1, 18, 34}, {1, 3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT16, true, {1, 18, 34}, {1, 3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT32, true, {1, 18, 34}, {1, 3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT32, true, {1, 18, 34}, {1, 3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT, true, {1, 18, 34}, {1, 3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_FLOAT16, true, {1, 18, 34}, {1, 3, 18, 1, 16}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT1, true, {1, 18, 134}, {1, 1, 18, 1, 256}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_UINT2, true, {1, 18, 134}, {1, 2, 18, 1, 128}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT2, true, {1, 18, 134}, {1, 2, 18, 1, 128}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, DT_INT4, true, {1, 18, 134}, {1, 3, 18, 1, 64}, 1);
  RunTransferShapeWithExtAxis(nullptr, ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_NZ, DT_FLOAT16, true, {8, 1000}, {63, 8, 1, 16}, 1);
  transformer::TransferShapeUtils::m0_list_.fill(16);
}

TEST_F(TransformerTransferShapeUT, transfer_shape_from_nd_to_fz) {
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT8, true, {18, 34}, {1, 3, 16, 32});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT8, true, {18, 34}, {1, 3, 16, 32});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT16, true, {18, 34}, {2, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT16, true, {18, 34}, {2, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT32, true, {18, 34}, {2, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT32, true, {18, 34}, {2, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_FLOAT, true, {18, 34}, {2, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {18, 34}, {2, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT1, true, {188, 23}, {1, 2, 16, 256});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT2, true, {188, 23}, {2, 2, 16, 128});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT2, true, {188, 23}, {2, 2, 16, 128});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT4, true, {188, 23}, {3, 2, 16, 64});

  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT8, true, {48, 512, 5, 5}, {400, 3, 16, 32});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT8, true, {48, 512, 5, 5}, {400, 3, 16, 32});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT16, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT16, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT32, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT32, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_FLOAT, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_FLOAT16, true, {48, 512, 5, 5}, {800, 3, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT1, true, {48, 512, 5, 5}, {50, 3, 16, 256});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_UINT2, true, {48, 512, 5, 5}, {100, 3, 16, 128});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT2, true, {48, 512, 5, 5}, {100, 3, 16, 128});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, DT_INT4, true, {48, 512, 5, 5}, {200, 3, 16, 64});

  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_UINT8, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_INT8, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_UINT16, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_INT16, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_UINT32, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_INT32, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_FLOAT, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_FLOAT16, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_UINT1, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_UINT2, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_INT2, true, {48, 80, 5, 5}, {6, 4, 16, 16});
  RunTransferShape(ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_LSTM, DT_INT4, true, {48, 80, 5, 5}, {6, 4, 16, 16});
}

TEST_F(TransformerTransferShapeUT, transfer_shape_from_nd_to_zn_rnn) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test", "test");
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {128}, {128});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {65, 128}, {65, 128});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT1, true, {65, 128}, {65, 128});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT2, true, {65, 128}, {65, 128});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT4, true, {65, 128}, {65, 128});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT8, true, {65, 128}, {65, 128});

  (void)ge::AttrUtils::SetInt(op_desc, "input_size", 30);
  (void)ge::AttrUtils::SetInt(op_desc, "hidden_size", 40);
  (void)ge::AttrUtils::SetInt(op_desc, "state_size", -1);
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {128}, {128});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {70, 128}, {5, 9, 16, 16});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT1, true, {70, 128}, {5, 3, 16, 256});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT2, true, {70, 128}, {5, 3, 16, 128});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT4, true, {70, 128}, {5, 3, 16, 64});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT8, true, {70, 128}, {5, 6, 16, 32});

  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {9, 40, 128},
                   {9, 3, 9, 16, 16});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT1, true, {9, 40, 128}, {9, 3, 3, 16, 256});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT2, true, {9, 40, 128}, {9, 3, 3, 16, 128});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT4, true, {9, 40, 128}, {9, 3, 3, 16, 64});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT8, true, {9, 40, 128}, {9, 3, 6, 16, 32});

  (void)ge::AttrUtils::SetInt(op_desc, "state_size", 70);
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {128}, {128});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {70, 128}, {5, 9, 16, 16});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT1, true, {70, 128}, {5, 3, 16, 256});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT2, true, {70, 128}, {5, 3, 16, 128});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT4, true, {70, 128}, {5, 3, 16, 64});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT8, true, {70, 128}, {5, 6, 16, 32});

  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16, true, {9, 100, 128},
                   {9, 7, 9, 16, 16});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT1, true, {9, 100, 128},
                   {9, 7, 3, 16, 256});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_UINT2, true, {9, 100, 128},
                   {9, 7, 3, 16, 128});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT4, true, {9, 100, 128}, {9, 7, 3, 16, 64});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_FRACTAL_ZN_RNN, DT_INT8, true, {9, 100, 128}, {9, 7, 6, 16, 32});
}

TEST_F(TransformerTransferShapeUT, transfer_shape_from_nd_to_nd_rnn_bias) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test", "test");
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {}, {});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {150}, {2400});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {18, 80}, {18, 1280});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_UINT1, true, {18, 80}, {18, 20480});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_UINT2, true, {18, 80}, {18, 10240});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_INT4, true, {18, 80}, {18, 5120});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_INT8, true, {18, 80}, {18, 2560});

  (void)ge::AttrUtils::SetInt(op_desc, "hidden_size", 64);
  (void)ge::AttrUtils::SetInt(op_desc, "input_size", 1);
  (void)ge::AttrUtils::SetInt(op_desc, "state_size", 1);
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {}, {});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {150}, {128});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {18, 80}, {18, 64});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_UINT1, true, {18, 80}, {18, 256});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_UINT2, true, {18, 80}, {18, 128});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_INT4, true, {18, 80}, {18, 64});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_INT8, true, {18, 80}, {18, 64});

  (void)ge::AttrUtils::SetInt(op_desc, "hidden_size", 0);
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {150}, {150});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_INT8, true, {18, 80}, {18, 80});
  RunTransferShape(op_desc, ge::FORMAT_ND, ge::FORMAT_ND_RNN_BIAS, DT_FLOAT16, true, {-2}, {-2}, true);
}
TEST_F(TransformerTransferShapeUT, transfer_shape_from_nyuva) {
    ShapeTransferAccordingToFormat shape_transfer;
    gert::Shape current_shape;
    vector<int64_t> dims = {42, 63, 3};
    vector<int64_t> expect_dim = {48, 64, 3};
    for (const int64_t &d : dims) {
      current_shape.AppendDim(d);
    }
    bool ret = shape_transfer.TransferShape(ge::FORMAT_NYUV, ge::FORMAT_NYUV_A, DT_INT8, current_shape);
    EXPECT_EQ(ret, true);
    if (ret) {
      vector<int64_t> new_dim;
      for (size_t i = 0; i < current_shape.GetDimNum(); ++i) {
        new_dim.push_back(current_shape.GetDim(i));
      }
      EXPECT_EQ(new_dim, expect_dim);
    }
}
}  // namespace ges

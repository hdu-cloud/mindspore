/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <algorithm>

#include "minddata/dataset/kernels/ir/vision/softdvpp_decode_random_crop_resize_jpeg_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/soft_dvpp/soft_dvpp_decode_random_crop_resize_jpeg_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {

namespace vision {

#ifndef ENABLE_ANDROID

// SoftDvppDecodeRandomCropResizeJpegOperation
SoftDvppDecodeRandomCropResizeJpegOperation::SoftDvppDecodeRandomCropResizeJpegOperation(std::vector<int32_t> size,
                                                                                         std::vector<float> scale,
                                                                                         std::vector<float> ratio,
                                                                                         int32_t max_attempts)
    : size_(size), scale_(scale), ratio_(ratio), max_attempts_(max_attempts) {}

SoftDvppDecodeRandomCropResizeJpegOperation::~SoftDvppDecodeRandomCropResizeJpegOperation() = default;

std::string SoftDvppDecodeRandomCropResizeJpegOperation::Name() const {
  return kSoftDvppDecodeRandomCropResizeJpegOperation;
}

Status SoftDvppDecodeRandomCropResizeJpegOperation::ValidateParams() {
  // size
  RETURN_IF_NOT_OK(ValidateVectorSize("SoftDvppDecodeRandomCropResizeJpeg", size_));
  for (int32_t i = 0; i < size_.size(); i++) {
    if (size_[i] % 2 == 1) {
      std::string err_msg = "SoftDvppDecodeRandomCropResizeJpeg: size[" + std::to_string(i) +
                            "] must be even values, got: " + std::to_string(size_[i]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  // scale
  RETURN_IF_NOT_OK(ValidateVectorScale("SoftDvppDecodeRandomCropResizeJpeg", scale_));
  // ratio
  RETURN_IF_NOT_OK(ValidateVectorRatio("SoftDvppDecodeRandomCropResizeJpeg", ratio_));
  // max_attempts
  if (max_attempts_ < 1) {
    std::string err_msg = "SoftDvppDecodeRandomCropResizeJpeg: max_attempts must be greater than or equal to 1, got: " +
                          std::to_string(max_attempts_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> SoftDvppDecodeRandomCropResizeJpegOperation::Build() {
  int32_t height = size_[0];
  int32_t width = size_[0];
  // User specified the width value.
  if (size_.size() == 2) {
    width = size_[1];
  }

  auto tensor_op = std::make_shared<SoftDvppDecodeRandomCropResizeJpegOp>(height, width, scale_[0], scale_[1],
                                                                          ratio_[0], ratio_[1], max_attempts_);
  return tensor_op;
}

Status SoftDvppDecodeRandomCropResizeJpegOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["size"] = size_;
  args["scale"] = scale_;
  args["ratio"] = ratio_;
  args["max_attempts"] = max_attempts_;
  *out_json = args;
  return Status::OK();
}

#endif

}  // namespace vision
}  // namespace dataset
}  // namespace mindspore

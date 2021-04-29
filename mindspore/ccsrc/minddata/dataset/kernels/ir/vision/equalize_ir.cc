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

#include "minddata/dataset/kernels/ir/vision/equalize_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/equalize_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {

namespace vision {

#ifndef ENABLE_ANDROID

// EqualizeOperation
EqualizeOperation::~EqualizeOperation() = default;

std::string EqualizeOperation::Name() const { return kEqualizeOperation; }

Status EqualizeOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> EqualizeOperation::Build() { return std::make_shared<EqualizeOp>(); }

#endif

}  // namespace vision
}  // namespace dataset
}  // namespace mindspore

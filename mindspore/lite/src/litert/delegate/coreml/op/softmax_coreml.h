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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_OP_SOFTMAX_COREML_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_OP_SOFTMAX_COREML_H_

#include <vector>
#include <string>
#include "src/litert/delegate/coreml/op/coreml_op.h"
namespace mindspore::lite {
class SoftmaxCoreMLOp : public CoreMLOp {
 public:
  SoftmaxCoreMLOp(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                  const std::vector<mindspore::MSTensor> &out_tensors, std::string name)
      : CoreMLOp(primitive, in_tensors, out_tensors, name) {}

  int InitParams() override;

  int BuildLayer() override;

  int HandleAxis() override;

 private:
  const schema::Softmax *softmax_prim_;
  int axis_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_OP_SOFTMAX_COREML_H_

/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#include "register/infer_axis_slice_registry.h"
#include "external/graph/types.h"
#include "graph/operator_factory_impl.h"

namespace ge {
InferAxisTypeInfoFuncRegister::InferAxisTypeInfoFuncRegister(const char_t *const operator_type,
                                                             const InferAxisTypeInfoFunc &infer_axis_type_info_func) {
  (void)OperatorFactoryImpl::RegisterInferAxisTypeInfoFunc(operator_type, infer_axis_type_info_func);
}

InferAxisSliceFuncRegister::InferAxisSliceFuncRegister(const char_t *const operator_type,
                                                       const InferAxisSliceFunc &infer_axis_slice_func) {
  (void)OperatorFactoryImpl::RegisterInferAxisSliceFunc(operator_type, infer_axis_slice_func);
}
}  // namespace ge

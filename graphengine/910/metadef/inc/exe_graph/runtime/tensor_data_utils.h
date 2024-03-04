/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#ifndef METADEF_CXX_INC_EXE_GRAPH_TENSOR_DATA_UTILS_H_
#define METADEF_CXX_INC_EXE_GRAPH_TENSOR_DATA_UTILS_H_
#include "exe_graph/runtime/tensor_data.h"
#include "graph/types.h"

namespace gert {
inline const ge::char_t *GetPlacementStr(const TensorPlacement placement) {
  static const ge::char_t *placement_str[static_cast<int32_t>(kTensorPlacementEnd) + 1] = {"DeviceHbm", "HostDDR",
                                                                                           "HostDDR", "Unknown"};
  if ((placement >= kTensorPlacementEnd) || (placement < kOnDeviceHbm)) {
    return placement_str[kTensorPlacementEnd];
  }
  return placement_str[placement];
}
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_TENSOR_DATA_UTILS_H_
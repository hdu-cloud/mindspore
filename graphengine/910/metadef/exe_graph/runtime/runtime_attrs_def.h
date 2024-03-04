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

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_RUNTIME_ATTR_CDEF_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_RUNTIME_ATTR_CDEF_H_
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  size_t attr_num;
  uint8_t reserved_[40];  // Reserved field, 32+8, do not directly use when only 8-byte left
  size_t offset[0];
} RuntimeAttrsDef;
#ifdef __cplusplus
}
#endif

#endif  // AIR_CXX_RUNTIME_V2_KERNEL_RUNTIME_ATTR_CDEF_H_

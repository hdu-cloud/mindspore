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

#ifndef COMMON_UTILS_TRANSFORMER_INC_AXIS_CONSTANTS_H_
#define COMMON_UTILS_TRANSFORMER_INC_AXIS_CONSTANTS_H_

namespace transformer {
extern const size_t DIM_SIZE_TWO;
extern const size_t DIM_SIZE_FOUR;
extern const size_t DIM_SIZE_FIVE;
extern const size_t DIM_SIZE_SIX;

extern const size_t EXT_INDEX_INPUT_SIZE;
extern const size_t EXT_INDEX_HIDEEN_SIZE;
extern const size_t EXT_INDEX_STATE_SIZE;

extern const int32_t AXIS_NCHW_DIM_N;
extern const int32_t AXIS_NCHW_DIM_C;
extern const int32_t AXIS_NCHW_DIM_H;
extern const int32_t AXIS_NCHW_DIM_W;

extern const int32_t AXIS_NHWC_DIM_N;
extern const int32_t AXIS_NHWC_DIM_H;
extern const int32_t AXIS_NHWC_DIM_W;
extern const int32_t AXIS_NHWC_DIM_C;

extern const int32_t AXIS_HWCN_DIM_H;
extern const int32_t AXIS_HWCN_DIM_W;
extern const int32_t AXIS_HWCN_DIM_C;
extern const int32_t AXIS_HWCN_DIM_N;

extern const int32_t AXIS_CHWN_DIM_C;
extern const int32_t AXIS_CHWN_DIM_H;
extern const int32_t AXIS_CHWN_DIM_W;
extern const int32_t AXIS_CHWN_DIM_N;

extern const int32_t NDHWC_DIM_N;
extern const int32_t NDHWC_DIM_D;
extern const int32_t NDHWC_DIM_H;
extern const int32_t NDHWC_DIM_W;
extern const int32_t NDHWC_DIM_C;

extern const int32_t NCDHW_DIM_N;
extern const int32_t NCDHW_DIM_C;
extern const int32_t NCDHW_DIM_D;
extern const int32_t NCDHW_DIM_H;
extern const int32_t NCDHW_DIM_W;

extern const int32_t DHWCN_DIM_D;
extern const int32_t DHWCN_DIM_H;
extern const int32_t DHWCN_DIM_W;
extern const int32_t DHWCN_DIM_C;
extern const int32_t DHWCN_DIM_N;

extern const int32_t DHWNC_DIM_D;
extern const int32_t DHWNC_DIM_H;
extern const int32_t DHWNC_DIM_W;
extern const int32_t DHWNC_DIM_N;
extern const int32_t DHWNC_DIM_C;

extern const int32_t AXIS_NC1HWC0_DIM_N;
extern const int32_t AXIS_NC1HWC0_DIM_C1;
extern const int32_t AXIS_NC1HWC0_DIM_H;
extern const int32_t AXIS_NC1HWC0_DIM_W;
extern const int32_t AXIS_NC1HWC0_DIM_C0;

extern const int32_t AXIS_C1HWNCoC0_DIM_C1;
extern const int32_t AXIS_C1HWNCoC0_DIM_H;
extern const int32_t AXIS_C1HWNCoC0_DIM_W;
extern const int32_t AXIS_C1HWNCoC0_DIM_N;
extern const int32_t AXIS_C1HWNCoC0_DIM_Co;
} // namespace transformer

#endif // COMMON_UTILS_TRANSFORMER_INC_AXIS_CONSTANTS_H_

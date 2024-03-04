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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_DEFORMABLE_OFFSETS_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_DEFORMABLE_OFFSETS_IMPL_CUH_

#include <cuda_runtime.h>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

CUDA_LIB_EXPORT cudaError_t GenPositionGrid(const uint kernel_h, const uint kernel_w, const uint stride_h,
                                            const uint stride_w, const uint dilations_h, const uint dilations_w,
                                            const uint pad_l, const uint pad_t, const uint output_w, const uint num,
                                            int32_t *position_grid, const uint32_t device_id, cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT cudaError_t DeformableOffsets(const T *input, const T *offsets, const int32_t *position_grid, uint n,
                                              uint c, uint input_h, uint input_w, uint dfm_group, uint kernel_h,
                                              uint kernel_w, uint output_h, uint output_w, T *output,
                                              uint32_t device_id, cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_DEFORMABLE_OFFSETS_IMPL_CUH_

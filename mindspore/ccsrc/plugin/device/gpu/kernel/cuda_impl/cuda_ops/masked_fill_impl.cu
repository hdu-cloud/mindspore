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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/masked_fill_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void ElewiseMaskedFillKernel(size_t size, const T *input, const bool *mask, T value, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = mask[pos] ? value : input[pos];
  }
}

__device__ __forceinline__ size_t Index(const size_t &index, const size_t &dim) { return dim == 1 ? 0 : index; }

template <typename T>
__global__ void BroadcastMaskedFillKernel(const size_t l0, const size_t l1, const size_t l2, const size_t l3,
                                          const size_t l4, const size_t l5, const size_t l6, const size_t r0,
                                          const size_t r1, const size_t r2, const size_t r3, const size_t r4,
                                          const size_t r5, const size_t r6, const size_t d0, const size_t d1,
                                          const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                          const size_t d6, const T *input, const bool *mask, T value, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < d0 * d1 * d2 * d3 * d4 * d5 * d6;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / (d1 * d2 * d3 * d4 * d5 * d6) % d0;
    size_t j = pos / (d2 * d3 * d4 * d5 * d6) % d1;
    size_t k = pos / (d3 * d4 * d5 * d6) % d2;
    size_t l = pos / (d4 * d5 * d6) % d3;
    size_t m = pos / (d5 * d6) % d4;
    size_t n = pos / d6 % d5;
    size_t o = pos % d6;

    size_t l_index = Index(i, l0) * l1 * l2 * l3 * l4 * l5 * l6;
    l_index += Index(j, l1) * l2 * l3 * l4 * l5 * l6;
    l_index += Index(k, l2) * l3 * l4 * l5 * l6;
    l_index += Index(l, l3) * l4 * l5 * l6;
    l_index += Index(m, l4) * l5 * l6;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    size_t r_index = Index(i, r0) * r1 * r2 * r3 * r4 * r5 * r6;
    r_index += Index(j, r1) * r2 * r3 * r4 * r5 * r6;
    r_index += Index(k, r2) * r3 * r4 * r5 * r6;
    r_index += Index(l, r3) * r4 * r5 * r6;
    r_index += Index(m, r4) * r5 * r6;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    output[pos] = mask[r_index] ? value : input[l_index];
  }
}

template <typename T>
void ElewiseMaskedFill(const size_t input_size, const T *input, const bool *mask, T value, T *output,
                       cudaStream_t cuda_stream) {
  ElewiseMaskedFillKernel<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(input_size, input, mask, value,
                                                                                   output);
}

template <typename T>
void BroadcastMaskedFill(const std::vector<size_t> &input_shape, const std::vector<size_t> &mask_shape,
                         const std::vector<size_t> &output_shape, const T *input, const bool *mask, T value, T *output,
                         cudaStream_t cuda_stream) {
  size_t size = 1;
  for (auto d : output_shape) {
    size *= d;
  }
  BroadcastMaskedFillKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(
    input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4], input_shape[5], input_shape[6],
    mask_shape[0], mask_shape[1], mask_shape[2], mask_shape[3], mask_shape[4], mask_shape[5], mask_shape[6],
    output_shape[0], output_shape[1], output_shape[2], output_shape[3], output_shape[4], output_shape[5],
    output_shape[6], input, mask, value, output);
}

template CUDA_LIB_EXPORT void ElewiseMaskedFill<half>(const size_t input_size, const half *input, const bool *mask,
                                                      half value, half *output, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseMaskedFill<float>(const size_t input_size, const float *input, const bool *mask,
                                                       float value, float *output, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseMaskedFill<int8_t>(const size_t input_size, const int8_t *input, const bool *mask,
                                                        int8_t value, int8_t *output, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseMaskedFill<int32_t>(const size_t input_size, const int32_t *input,
                                                         const bool *mask, int32_t value, int32_t *output,
                                                         cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastMaskedFill<half>(const std::vector<size_t> &input_shape,
                                                        const std::vector<size_t> &mask_shape,
                                                        const std::vector<size_t> &output_shape, const half *input,
                                                        const bool *mask, half value, half *output,
                                                        cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastMaskedFill<float>(const std::vector<size_t> &input_shape,
                                                         const std::vector<size_t> &mask_shape,
                                                         const std::vector<size_t> &output_shape, const float *input,
                                                         const bool *mask, float value, float *output,
                                                         cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastMaskedFill<int8_t>(const std::vector<size_t> &input_shape,
                                                          const std::vector<size_t> &mask_shape,
                                                          const std::vector<size_t> &output_shape, const int8_t *input,
                                                          const bool *mask, int8_t value, int8_t *output,
                                                          cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastMaskedFill<int32_t>(const std::vector<size_t> &input_shape,
                                                           const std::vector<size_t> &mask_shape,
                                                           const std::vector<size_t> &output_shape,
                                                           const int32_t *input, const bool *mask, int32_t value,
                                                           int32_t *output, cudaStream_t stream);
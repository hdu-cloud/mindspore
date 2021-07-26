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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/vatom/v1_coordinate_refresh_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"

__global__ void v1_Coordinate_Refresh(const int virtual_numbers, const VIRTUAL_TYPE_1 *v_info,
                                      const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, VECTOR *coordinate) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < virtual_numbers) {
    VIRTUAL_TYPE_1 v_temp = v_info[i];
    int atom_v = v_temp.virtual_atom;
    int atom_1 = v_temp.from_1;
    int atom_2 = v_temp.from_2;
    float a = v_temp.a;
    VECTOR rv1 = a * Get_Periodic_Displacement(uint_crd[atom_2], uint_crd[atom_1], scaler);
    coordinate[atom_v] = coordinate[atom_1] + rv1;
  }
}

void v1CoordinateRefresh(int atom_numbers, int virtual_numbers, const float *v_info_f, const int *uint_crd_f,
                         const float *scaler_f, float *crd_f, cudaStream_t stream) {
  Reset_List<<<ceilf(static_cast<float>(3. * atom_numbers) / 128), 128, 0, stream>>>(3 * atom_numbers, crd_f, 0.);
  size_t thread_per_block = 128;
  size_t block_per_grid = ceilf(static_cast<float>(virtual_numbers) / 128);
  UNSIGNED_INT_VECTOR *uint_crd =
    const_cast<UNSIGNED_INT_VECTOR *>(reinterpret_cast<const UNSIGNED_INT_VECTOR *>(uint_crd_f));

  VECTOR *scaler = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(scaler_f));
  VECTOR *crd = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(crd_f));
  VIRTUAL_TYPE_1 *v_info = const_cast<VIRTUAL_TYPE_1 *>(reinterpret_cast<const VIRTUAL_TYPE_1 *>(v_info_f));

  v1_Coordinate_Refresh<<<block_per_grid, thread_per_block, 0, stream>>>(virtual_numbers, v_info, uint_crd, scaler[0],
                                                                         crd);

  return;
}

void v1CoordinateRefresh(int atom_numbers, int virtual_numbers, const float *v_info_f, const int *uint_crd_f,
                         const float *scaler_f, float *crd_f, cudaStream_t stream);

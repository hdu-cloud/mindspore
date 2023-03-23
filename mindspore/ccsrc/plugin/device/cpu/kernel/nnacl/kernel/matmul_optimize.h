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

#ifndef MINDSPORE_NNACL_EXPERIMENT_MATMUL_OPTIMIZE_H_
#define MINDSPORE_NNACL_EXPERIMENT_MATMUL_OPTIMIZE_H_

#include "nnacl/kernel.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/experimental/ms_core.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct MatmulOptStru {
  KernelBase *base;
  MatMulParameter param;
  int row_tile;
  int col_tile;
} MatmulOptStru;

void MatmulOpt_prepare(MatmulOptStru *matmul);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_NNACL_EXPERIMENT_MATMUL_OPTIMIZE_H_

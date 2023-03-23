#ifdef ENABLE_AVX512
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

#include "src/litert/kernel/cpu/fp32/matmul_fp32_avx512.h"
#include "src/litert/kernel/cpu/fp32/matmul_fp32_base.h"
#include "nnacl/fp32/matmul_avx512_fp32.h"
#include "nnacl/fp32/matmul_avx512_mask_fp32.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/fp32/pack_fp32.h"

namespace mindspore::kernel {
void MatmulFp32AVX512CPUKernel::InitGlobalVariable() {
  matrix_a_.need_pack = true;
  matrix_b_.need_pack = true;
  matrix_a_pack_fun_ = params_->a_transpose_ ? RowMajor2ColMajorParallel : RowMajor2RowMajorParallel;
  matrix_b_pack_fun_ = params_->b_transpose_ ? RowMajor2Col64MajorParallel : RowMajor2Row64MajorParallel;
  matrix_a_.need_pack = params_->a_transpose_;
  row_tile_ = C1NUM;
  col_tile_ = C16NUM;
  col_min_unit_ = C64NUM;

  if (params_->row_ == 1) {
    if (!params_->b_const_ && params_->col_ <= C128NUM) {
      out_need_aligned_ = true;
    }
  } else if (params_->col_ == 1) {
    out_need_aligned_ = true;
  } else {
    out_need_aligned_ = false;
  }
}

int MatmulFp32AVX512CPUKernel::PackMatrixAImplOpt() {
  MS_LOG(ERROR) << "Matmul: don't support optimized-packing, only support single-thread currently.";
  return RET_ERROR;
}

int MatmulFp32AVX512CPUKernel::ParallelRunByBatch(int task_id) const {
  int start_batch = task_id * batch_stride_;
  int end_batch = MSMIN(params_->batch, start_batch + batch_stride_);
  int func_flag{0};
  if (params_->row_ == 1) {
    func_flag += (!params_->b_const_ && params_->col_ <= C128NUM) ? C2NUM : C1NUM;
  }

  for (int index = start_batch; index < end_batch; ++index) {
    const float *a = matrix_a_.pack_ptr + a_offset_[index] * params_->row_align_ * params_->deep_;
    const float *b = matrix_b_.pack_ptr + b_offset_[index] * params_->deep_ * params_->col_align_;
    float *c = output_data_ + index * params_->row_ * col_step_;

    auto bias = (matrix_c_.pack_ptr == nullptr) ? nullptr : matrix_c_.pack_ptr;
    if (func_flag == 0) {
      if (out_need_aligned_) {
        MatMulAvx512Fp32(a, b, c, bias, params_->act_type_, params_->deep_, col_step_, params_->col_align_,
                         params_->row_);
      } else {
        MatMulMaskAvx512Fp32(a, b, c, bias, params_->act_type_, params_->deep_, col_step_, params_->col_,
                             params_->row_);
      }
    } else if (func_flag == C1NUM) {
      if (out_need_aligned_) {
        MatVecMulAvx512Fp32(a, b, c, bias, params_->act_type_, params_->deep_, col_step_, params_->col_align_);
      } else {
        MatVecMulMaskAvx512Fp32(a, b, c, bias, params_->act_type_, params_->deep_, col_step_, params_->col_);
      }
    } else {
      MatVecMulNoPackFp32(a, b, c, bias, params_->act_type_, params_->deep_, col_step_, col_step_);
    }
  }
  return RET_OK;
}

int MatmulFp32AVX512CPUKernel::ParallelRunByRow(int task_id) const {
  if (task_id < 0 || task_id >= thread_count_) {
    MS_LOG(ERROR) << "task_id " << task_id << " is out of range, node is " << name_;
    return RET_ERROR;
  }
  int start_row = split_points_[task_id];
  int end_row = row_num_;
  if (task_id < (thread_count_ - 1)) {
    end_row = split_points_[task_id + 1];
  }
  int row_num = end_row - start_row;
  if (row_num <= 0) {
    return RET_OK;
  }
  const float *input = matrix_a_.pack_ptr + start_row * params_->deep_;
  float *output = output_data_ + start_row * col_step_;
  if (params_->col_ == 1) {
    float bias = 0;
    if (matrix_c_.pack_ptr != nullptr) {
      bias = matrix_c_.pack_ptr[0];
    }
    gemmIsNotPackFun(input, matrix_b_.pack_ptr, output, &bias, row_num, params_->deep_, params_->act_type_);
  } else {
    if (out_need_aligned_) {
      MatMulAvx512Fp32(input, matrix_b_.pack_ptr, output, matrix_c_.pack_ptr, params_->act_type_, params_->deep_,
                       params_->col_align_, params_->col_align_, row_num);
    } else {
      MatMulMaskAvx512Fp32(input, matrix_b_.pack_ptr, output, matrix_c_.pack_ptr, params_->act_type_, params_->deep_,
                           params_->col_, params_->col_, row_num);
    }
  }
  return RET_OK;
}

int MatmulFp32AVX512CPUKernel::ParallelRunByOC(int task_id) const {
  if (task_id < 0 || task_id >= thread_count_) {
    MS_LOG(ERROR) << "task_id " << task_id << " is out of range, node is " << name_;
    return RET_ERROR;
  }
  int start_oc = split_points_[task_id];
  int end_oc = col_step_;
  if (task_id < (thread_count_ - 1)) {
    end_oc = split_points_[task_id + 1];
  }
  int compute_oc = end_oc - start_oc;
  if (compute_oc <= 0) {
    return RET_OK;
  }
  int func_flag{0};
  if (params_->row_ == 1) {
    func_flag += (!params_->b_const_ && params_->col_ <= C128NUM) ? C2NUM : C1NUM;
  }
  int b_stride = func_flag == C2NUM ? 1 : params_->deep_;
  for (int i = 0; i < params_->batch; ++i) {
    auto a = matrix_a_.pack_ptr + a_offset_[i] * params_->row_align_ * params_->deep_;
    auto b = matrix_b_.pack_ptr + b_offset_[i] * params_->deep_ * params_->col_align_ + start_oc * b_stride;
    auto c = output_data_ + i * params_->row_ * col_step_ + start_oc;
    auto bias = (matrix_c_.pack_ptr == nullptr) ? nullptr : matrix_c_.pack_ptr + start_oc;
    if (func_flag == 0) {
      if (out_need_aligned_) {
        MatMulAvx512Fp32(a, b, c, bias, params_->act_type_, params_->deep_, compute_oc, params_->col_align_,
                         params_->row_);
      } else {
        MatMulMaskAvx512Fp32(a, b, c, bias, params_->act_type_, params_->deep_, compute_oc, params_->col_,
                             params_->row_);
      }
    } else if (func_flag == C1NUM) {
      if (out_need_aligned_) {
        MatVecMulAvx512Fp32(a, b, c, bias, params_->act_type_, params_->deep_, compute_oc, params_->col_align_);
      } else {
        MatVecMulMaskAvx512Fp32(a, b, c, bias, params_->act_type_, params_->deep_, compute_oc, params_->col_);
      }
    } else {
      MatVecMulNoPackFp32(a, b, c, bias, params_->act_type_, params_->deep_, compute_oc, col_step_);
    }
  }
  return RET_OK;
}

bool MatmulFp32AVX512CPUKernel::CheckThreadCuttingByRow() {
  if (b_batch_ != C1NUM) {
    return false;
  }
  if (row_num_ < op_parameter_->thread_num_) {
    return false;
  }
  if (params_->col_ == 1) {
    row_min_unit_ = C8NUM;
    return true;
  }
  if (params_->row_ == 1 && !params_->b_const_ && params_->col_ <= C128NUM) {
    return false;
  }
  row_min_unit_ = C6NUM;
  if (col_step_ < C48NUM) {
    row_min_unit_ = C12NUM;
  } else if (col_step_ < C64NUM) {
    row_min_unit_ = C8NUM;
  }
  return MSMIN(row_num_ / row_min_unit_, op_parameter_->thread_num_) >
         MSMIN(col_step_ / col_min_unit_, op_parameter_->thread_num_);
}
}  // namespace mindspore::kernel
#endif

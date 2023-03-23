/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_MATMUL_BASE_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_MATMUL_BASE_FP16_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include <vector>
#include "src/litert/lite_kernel.h"
#include "src/common/common.h"
#include "nnacl/matmul_parameter.h"

namespace mindspore::kernel {
class MatmulBaseFP16CPUKernel : public LiteKernel {
 public:
  explicit MatmulBaseFP16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                   const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    params_ = reinterpret_cast<MatMulParameter *>(op_parameter_);
  }
  ~MatmulBaseFP16CPUKernel() override;
  int Prepare() override;
  int ReSize() override;
  int Run() override;

 public:
  int RunImpl(int task_id);

 protected:
  virtual int InitAShape() = 0;
  virtual int InitBShape() = 0;
  bool IsRepack() { return is_repack_; }
  bool is_repack_ = false;

 private:
  int InitBias();
  void ResizeParameter();
  int InitBufferA();
  int InitBufferB();
  void InitMatrixA(const void *src_ptr);
  void InitMatrixB(const void *src_ptr, TypeId data_type);
  void FreeResizeBufA();
  void FreeResizeBufB();

 protected:
  MatMulParameter *params_ = nullptr;
  int row_tile_ = 0;
  int a_batch_ = 1;
  int b_batch_ = 1;
  std::vector<int> a_offset_;
  std::vector<int> b_offset_;

 private:
  int thread_stride_ = 0;
  int thread_count_ = 0;
  int bias_count_ = 0;
  bool vec_matmul_ = false;
  float16_t *a_pack_ptr_ = nullptr;
  float16_t *b_pack_ptr_ = nullptr;
  float16_t *src_b_ = nullptr;
  float16_t *bias_ptr_ = nullptr;
  float16_t *batch_a_ptr_ = nullptr;
  float16_t *batch_b_ptr_ = nullptr;
  float16_t *batch_c_ptr_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_MATMUL_BASE_FP16_H_

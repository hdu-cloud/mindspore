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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_DYNAMIC_GRU_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_DYNAMIC_GRU_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <memory>
#include <utility>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"

namespace mindspore {
namespace kernel {
class DynamicGruGpuKernelMod : public NativeGpuKernelMod {
 public:
  DynamicGruGpuKernelMod()
      : batch_size_(0),
        max_seq_len_(0),
        input_size_(0),
        hidden_size_(0),
        num_layers_(0),
        has_bias_(false),
        bidirectional_(false),
        states_init_(false),
        is_null_input_(false),
        is_train_(true),
        dropout_(0),
        weight_size_(0),
        reserved_size_(0),
        x_desc_(nullptr),
        x_desc_max_(nullptr),
        hx_desc_(nullptr),
        cx_desc_(nullptr),
        w_desc_(nullptr),
        dropout_desc_(nullptr),
        y_desc_(nullptr),
        hy_desc_(nullptr),
        cy_desc_(nullptr),
        rnn_desc_(nullptr),
        handle_(nullptr),
        cudnn_data_type_(CUDNN_DATA_FLOAT) {}

  ~DynamicGruGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyRNNDescriptor(rnn_desc_), "Destroy rnn_desc failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyDropoutDescriptor(dropout_desc_), "Destroy dropout_desc failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(hx_desc_), "Destroy hx_desc failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(cx_desc_), "Destroy cx_desc failed");
    if (y_desc_.get() != nullptr) {
      CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyRNNDataDescriptor(*(y_desc_.get())), "Destroy y_desc failed");
    }
    if (x_desc_.get() != nullptr) {
      CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyRNNDataDescriptor(*(x_desc_.get())), "Destroy x_desc failed");
    }
#if CUDNN_VERSION < 8000
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(hy_desc_), "Destroy hy_desc failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(cy_desc_), "Destroy cy_desc failed");
    if (w_desc_ != nullptr) {
      CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyFilterDescriptor(w_desc_), "Destroy w_desc_failed");
    }
    if (x_desc_max_ != nullptr) {
      for (size_t i = 0; i < IntToSize(max_seq_len_); ++i) {
        CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyTensorDescriptor(x_desc_max_[i]), "destroy x_desc_max failed");
      }
    }
#else
    if (x_desc_max_.get() != nullptr) {
      CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyRNNDataDescriptor(*(x_desc_max_.get())),
                                         "Destroy x_desc_max_ failed");
    }
#endif
  }

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  void InitResource() override {
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&hx_desc_), "Create hx_desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&cx_desc_), "Create cx_desc failed");
#if CUDNN_VERSION < 8000
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&hy_desc_), "Create hy_desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&cy_desc_), "Create cy_desc failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnCreateFilterDescriptor(&w_desc_), "Create w_desc failed");
#endif
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateDropoutDescriptor(&dropout_desc_), "Create dropout_desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateRNNDescriptor(&rnn_desc_), "Create rnn_desc failed");
  }

 private:
  void ResetResource() noexcept;
  void CreateRNNDataDescGrp();
#if CUDNN_VERSION < 8000
  void CreateFilterDesc();
#endif
  void CreateTensorNdDesc();
  void SetRNNDesc();
  void CheckWeightSize(const std::vector<KernelTensorPtr> &inputs);
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);

  using DynamicGruGpuFunc =
    std::function<bool(DynamicGruGpuKernelMod *, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                       const std::vector<AddressPtr> &, void *)>;
  static std::vector<std::pair<KernelAttr, DynamicGruGpuFunc>> func_list_;
  DynamicGruGpuFunc kernel_func_;
  cudaStream_t cuda_stream_;

  int batch_size_;
  std::vector<int32_t> seq_lens_;
  int max_seq_len_;
  int input_size_;
  int hidden_size_;
  int num_layers_;

  bool has_bias_;
  bool bidirectional_;
  bool states_init_;
  bool is_null_input_;
  bool is_train_;
  float dropout_;

  size_t weight_size_;
  size_t reserved_size_;

  size_t input_type_size_;  // sizeof(T)

  // input desc
  std::unique_ptr<cudnnRNNDataDescriptor_t> x_desc_;
#if CUDNN_VERSION < 8000
  std::unique_ptr<cudnnTensorDescriptor_t[]> x_desc_max_;
#else
  std::unique_ptr<cudnnRNNDataDescriptor_t> x_desc_max_;
#endif
  cudnnTensorDescriptor_t hx_desc_;
  cudnnTensorDescriptor_t cx_desc_;
  cudnnFilterDescriptor_t w_desc_;
  cudnnDropoutDescriptor_t dropout_desc_;
  std::unique_ptr<cudnnRNNDataDescriptor_t> y_desc_;
  cudnnTensorDescriptor_t hy_desc_;
  cudnnTensorDescriptor_t cy_desc_;
  cudnnRNNDescriptor_t rnn_desc_;
  cudnnHandle_t handle_;
  cudnnDataType_t cudnn_data_type_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_DYNAMIC_GRU_GPU_KERNEL_H_

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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_REDUCESCATTER_TENSORRT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_REDUCESCATTER_TENSORRT_H_
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include "src/litert/delegate/tensorrt/op/tensorrt_op.h"
#include "src/litert/delegate/tensorrt/op/tensorrt_plugin.h"
#include "src/litert/delegate/tensorrt/tensorrt_utils.h"
#include "src/litert/delegate/tensorrt/distribution/distribution_collective.h"

namespace mindspore::lite {
constexpr char *REDUCESCATTER_PLUGIN_NAME{"ReduceScatterPlugin"};
class ReduceScatterTensorRT : public TensorRTOp {
 public:
  ReduceScatterTensorRT(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                        const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name,
                        const schema::QuantType &quant_type)
      : TensorRTOp(primitive, in_tensors, out_tensors, name, quant_type) {}

  ~ReduceScatterTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors) override;
};

class ReduceScatterPlugin : public TensorRTPlugin {
 public:
  ReduceScatterPlugin(const std::string name, schema::ReduceMode red_mode, int rank, uint32_t device_id)
      : TensorRTPlugin(name, std::string(REDUCESCATTER_PLUGIN_NAME), device_id), red_mode_(red_mode), rank_(rank) {}

  ReduceScatterPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(REDUCESCATTER_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    red_mode_ = static_cast<const schema::ReduceMode *>(fields[0].data)[0];
    rank_ = static_cast<const int *>(fields[1].data)[0];
  }

  ReduceScatterPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(REDUCESCATTER_PLUGIN_NAME)) {
    DeserializeValue(&serialData, &serialLength, &red_mode_, sizeof(schema::ReduceMode));
    DeserializeValue(&serialData, &serialLength, &rank_, sizeof(int));
  }

  ReduceScatterPlugin() = delete;

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
                                          nvinfer1::IExprBuilder &exprBuilder) noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;

 private:
  int rank_{0};
  schema::ReduceMode red_mode_;
};
class ReduceScatterPluginCreater : public TensorRTPluginCreater<ReduceScatterPlugin> {
 public:
  ReduceScatterPluginCreater() : TensorRTPluginCreater(std::string(REDUCESCATTER_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_REDUCESCATTER_TENSORRT_H_

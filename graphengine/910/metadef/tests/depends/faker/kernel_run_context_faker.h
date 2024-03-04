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

#ifndef AIR_CXX_TESTS_UT_GE_RUNTIME_V2_FAKER_KERNEL_RUN_CONTEXT_FACKER_H_
#define AIR_CXX_TESTS_UT_GE_RUNTIME_V2_FAKER_KERNEL_RUN_CONTEXT_FACKER_H_
#include <memory>
#include <vector>
#include <cstring>
#include "exe_graph/runtime/kernel_run_context.h"
#include "exe_graph/runtime/context_extend.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/lowering/buffer_pool.h"
#include "graph/any_value.h"
#include "graph/node.h"
#include "runtime/kernel_run_context_builder.h"

namespace gert {
struct FakeKernelContextHolder {
  template<typename T>
  T *GetContext() {
    return reinterpret_cast<T*>(holder.context_);
  }
  ComputeNodeInfo *MutableComputeNodeInfo() {
    return reinterpret_cast<ComputeNodeInfo *>(holder.compute_node_extend_holder_.get());
  }
  size_t kernel_input_num;
  size_t kernel_output_num;
  KernelContextHolder holder;
};
FakeKernelContextHolder BuildKernelRunContext(size_t input_num, size_t output_num);

class KernelRunContextFaker {
 public:
  KernelRunContextFaker() = default;
  KernelRunContextFaker &KernelIONum(size_t input_num, size_t output_num);
  KernelRunContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  KernelRunContextFaker &IrInputNum(size_t input_num);
  KernelRunContextFaker &IrOutputNum(size_t input_num);
  KernelRunContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num);
  KernelRunContextFaker &IrOutputInstanceNum(std::vector<uint32_t> instance_num);
  KernelRunContextFaker &NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                     ge::Format storage_format);
  KernelRunContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                      ge::Format storage_format);
  KernelRunContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value);
  KernelRunContextFaker &Inputs(std::vector<void *> inputs);
  KernelRunContextFaker &Outputs(std::vector<void *> outputs);

  FakeKernelContextHolder Build() const;

 private:
  ge::OpDescPtr FakeOp() const;

 private:
  size_t kernel_input_num_;
  size_t kernel_output_num_;
  size_t node_input_num_;
  size_t node_output_num_;
  std::vector<uint32_t> ir_instance_num_;
  std::vector<uint32_t> ir_output_instance_num_{};
  std::vector<CompileTimeTensorDesc> node_input_tds_;
  std::vector<CompileTimeTensorDesc> node_output_tds_;
  std::vector<void *> inputs_;
  std::vector<void *> outputs_;
  std::vector<std::pair<std::string, ge::AnyValue>> attrs_;
};

class InferShapeContextFaker {
 public:
  InferShapeContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  InferShapeContextFaker &IrInputNum(size_t input_num) {
    base_faker_.IrInputNum(input_num);
    return *this;
  }
  InferShapeContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num) {
    base_faker_.IrInstanceNum(std::move(instance_num));
    return *this;
  }
  InferShapeContextFaker &NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                      ge::Format storage_format) {
    base_faker_.NodeInputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  InferShapeContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                       ge::Format storage_format) {
    base_faker_.NodeOutputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  InferShapeContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
    base_faker_.NodeAttrs(std::move(keys_to_value));
    return *this;
  }

  InferShapeContextFaker &InputShapes(std::vector<void *> input_shapes);
  InferShapeContextFaker &OutputShapes(std::vector<void *> output_shapes);

  FakeKernelContextHolder Build() const;

 private:
  enum InputsAppend { kInputsInferShapeFunc, kInputsAppendEnd };

 private:
  KernelRunContextFaker base_faker_;
};

class InferShapeRangeContextFaker {
 public:
  InferShapeRangeContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  InferShapeRangeContextFaker &IrInputNum(size_t input_num) {
    base_faker_.IrInputNum(input_num);
    return *this;
  }
  InferShapeRangeContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num) {
    base_faker_.IrInstanceNum(std::move(instance_num));
    return *this;
  }
  InferShapeRangeContextFaker &NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                      ge::Format storage_format) {
    base_faker_.NodeInputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  InferShapeRangeContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                       ge::Format storage_format) {
    base_faker_.NodeOutputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  InferShapeRangeContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
    base_faker_.NodeAttrs(std::move(keys_to_value));
    return *this;
  }

  InferShapeRangeContextFaker &InputShapeRanges(std::vector<void *> input_shape_ranges);
  InferShapeRangeContextFaker &OutputShapeRanges(std::vector<void *> output_shape_ranges);

  FakeKernelContextHolder Build() const;

 private:
  enum InputsAppend { kInputsInferShapeRangeFunc, kInputsAppendEnd };

 private:
  KernelRunContextFaker base_faker_;
};

class InferDataTypeContextFaker {
 public:
  InferDataTypeContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  InferDataTypeContextFaker &IrInputNum(size_t input_num) {
    base_faker_.IrInputNum(input_num);
    return *this;
  }
  InferDataTypeContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num) {
    base_faker_.IrInstanceNum(std::move(instance_num));
    return *this;
  }
  InferDataTypeContextFaker &NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                      ge::Format storage_format) {
    base_faker_.NodeInputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  InferDataTypeContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                       ge::Format storage_format) {
    base_faker_.NodeOutputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  InferDataTypeContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
    base_faker_.NodeAttrs(std::move(keys_to_value));
    return *this;
  }

  InferDataTypeContextFaker &InputDataTypes(std::vector<void *> input_datatypes);
  InferDataTypeContextFaker &OutputDataTypes(std::vector<void *> output_datatypes);

  FakeKernelContextHolder Build() const;

 private:
  enum InputsAppend { kInputsInferDataTypeFunc, kInputsAppendEnd };

 private:
  std::vector<void *> inputs_;
  std::vector<void *> outputs_;
  KernelRunContextFaker base_faker_;
};

class TilingContextFaker {
 public:
  TilingContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  TilingContextFaker &IrInputNum(size_t input_num) {
    base_faker_.IrInputNum(input_num);
    return *this;
  }
  TilingContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num) {
    base_faker_.IrInstanceNum(std::move(instance_num));
    return *this;
  }
  TilingContextFaker &NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format, ge::Format storage_format) {
    base_faker_.NodeInputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  TilingContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                   ge::Format storage_format) {
    base_faker_.NodeOutputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  TilingContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
    base_faker_.NodeAttrs(std::move(keys_to_value));
    return *this;
  }
  TilingContextFaker &InputShapes(std::vector<gert::StorageShape *> input_shapes);
  TilingContextFaker &OutputShapes(std::vector<gert::StorageShape *> output_shapes);
  TilingContextFaker &CompileInfo(void *compile_info);
  TilingContextFaker &PlatformInfo(void *platform_info);
  TilingContextFaker &TilingData(void *tiling_data);
  TilingContextFaker &Workspace(ContinuousVector *workspace);

  FakeKernelContextHolder Build() const;

 private:
  void UpdateInputs();

 private:
  enum InputsAppend { kInputsCompileInfo, kInputsPlatformInfo, kInputsTilingFunc, kInputsAppendEnd };

  KernelRunContextFaker base_faker_;
  std::vector<gert::StorageShape *> input_shapes_;
  std::vector<gert::StorageShape *> output_shapes_;
  std::vector<void *> outputs_ {TilingContext::kOutputNum};

  void *compile_info_;
  void *platform_info_;
};
}  // namespace gert
#endif  //AIR_CXX_TESTS_UT_GE_RUNTIME_V2_FAKER_KERNEL_RUN_CONTEXT_FACKER_H_

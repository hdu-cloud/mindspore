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
#include "kernel_run_context_faker.h"
#include "graph/compute_graph.h"
#include "exe_graph/lowering/bg_kernel_context_extend.h"
#include "exe_graph/runtime/tiling_context.h"

namespace gert {
FakeKernelContextHolder BuildKernelRunContext(size_t input_num, size_t output_num) {
  return KernelRunContextFaker().KernelIONum(input_num, output_num).Build();
}
KernelRunContextFaker &KernelRunContextFaker::KernelIONum(size_t input_num, size_t output_num) {
  kernel_input_num_ = input_num;
  kernel_output_num_ = output_num;
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::NodeIoNum(size_t input_num, size_t output_num) {
  node_input_num_ = input_num;
  node_output_num_ = output_num;
  node_input_tds_.resize(input_num);
  node_output_tds_.resize(output_num);
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::IrInputNum(size_t input_num) {
  ir_instance_num_ = std::vector<uint32_t>(input_num, 1);
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::IrInstanceNum(std::vector<uint32_t> instance_num) {
  ir_instance_num_ = std::move(instance_num);
  return *this;
}

ge::OpDescPtr KernelRunContextFaker::FakeOp() const {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  size_t input_index = 0;
  for (size_t ir_index = 0; ir_index < ir_instance_num_.size(); ++ir_index) {
    auto ir_ins_num = ir_instance_num_[ir_index];
    auto prefix = "x_" + std::to_string(ir_index) + "_";
    op_desc->AppendIrInput(prefix, ge::kIrInputDynamic);
    for (size_t i = 0; i < ir_ins_num; ++i, ++input_index) {
      auto td = ge::GeTensorDesc();
      if (node_input_tds_.size() > input_index) {
        td.SetOriginFormat(node_input_tds_[input_index].GetOriginFormat());
        td.SetFormat(node_input_tds_[input_index].GetStorageFormat());
        td.SetDataType(node_input_tds_[input_index].GetDataType());
        td.SetOriginDataType(node_input_tds_[input_index].GetDataType());
      }
      op_desc->AddInputDesc(prefix + std::to_string(i), td);
    }
  }
  for (size_t i = 0; i < node_output_num_; ++i) {
    auto td = ge::GeTensorDesc();
    if (node_output_tds_.size() > i) {
      td.SetOriginFormat(node_output_tds_[i].GetOriginFormat());
      td.SetFormat(node_output_tds_[i].GetStorageFormat());
      td.SetDataType(node_output_tds_[i].GetDataType());
      td.SetOriginDataType(node_output_tds_[i].GetDataType());
    }
    op_desc->AddOutputDesc("y" + std::to_string(i), td);
  }
  for (const auto &attr : attrs_) {
    op_desc->AppendIrAttrName(attr.first);
    op_desc->SetAttr(attr.first, attr.second);
  }
  return op_desc;
}

FakeKernelContextHolder KernelRunContextFaker::Build() const {
  FakeKernelContextHolder fake_holder;
  fake_holder.kernel_input_num = kernel_input_num_;
  fake_holder.kernel_output_num = kernel_output_num_;
  KernelRunContextBuilder kernel_context_builder;
  auto op_desc = FakeOp();
  if (inputs_.size() != kernel_input_num_ || outputs_.size() != kernel_output_num_) {
    std::vector<void *> inputs(kernel_input_num_, nullptr);
    std::vector<void *> outputs(kernel_output_num_, nullptr);
    fake_holder.holder = kernel_context_builder.Inputs(inputs).Outputs(outputs).Build(op_desc);
    return fake_holder;
  }
  fake_holder.holder = kernel_context_builder.Inputs(inputs_).Outputs(outputs_).Build(op_desc);
  return fake_holder;
}
KernelRunContextFaker &KernelRunContextFaker::NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                                          ge::Format storage_format) {
  node_input_tds_[index].SetDataType(dt);
  node_input_tds_[index].SetOriginFormat(origin_format);
  node_input_tds_[index].SetStorageFormat(storage_format);
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                                           ge::Format storage_format) {
  node_output_tds_[index].SetDataType(dt);
  node_output_tds_[index].SetOriginFormat(origin_format);
  node_output_tds_[index].SetStorageFormat(storage_format);
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::Inputs(std::vector<void *> inputs) {
  inputs_ = std::move(inputs);
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::Outputs(std::vector<void *> outputs) {
  outputs_ = std::move(outputs);
  return *this;
}
KernelRunContextFaker &
KernelRunContextFaker::NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
  attrs_ = std::move(keys_to_value);
  return *this;
}
InferShapeContextFaker &InferShapeContextFaker::NodeIoNum(size_t input_num, size_t output_num) {
  base_faker_.KernelIONum(input_num + kInputsAppendEnd, output_num);
  base_faker_.NodeIoNum(input_num, output_num);
  return *this;
}
InferShapeContextFaker &InferShapeContextFaker::InputShapes(std::vector<void *> input_shapes) {
  std::vector<void *> inputs(std::move(input_shapes));
  inputs.push_back(nullptr);  // infershape func
  base_faker_.Inputs(std::move(inputs));
  return *this;
}
InferShapeContextFaker &InferShapeContextFaker::OutputShapes(std::vector<void *> output_shapes) {
  base_faker_.Outputs(std::move(output_shapes));
  return *this;
}
FakeKernelContextHolder InferShapeContextFaker::Build() const {
  return base_faker_.Build();
}
InferShapeRangeContextFaker &InferShapeRangeContextFaker::NodeIoNum(size_t input_num, size_t output_num) {
  base_faker_.KernelIONum(input_num + kInputsAppendEnd, output_num);
  base_faker_.NodeIoNum(input_num, output_num);
  return *this;
}
InferShapeRangeContextFaker &InferShapeRangeContextFaker::InputShapeRanges(std::vector<void *> input_shape_ranges) {
  std::vector<void *> inputs(std::move(input_shape_ranges));
  inputs.push_back(nullptr);  // infershaperange func
  base_faker_.Inputs(std::move(inputs));
  return *this;
}
InferShapeRangeContextFaker &InferShapeRangeContextFaker::OutputShapeRanges(std::vector<void *> output_shape_ranges) {
  base_faker_.Outputs(std::move(output_shape_ranges));
  return *this;
}
FakeKernelContextHolder InferShapeRangeContextFaker::Build() const {
    return base_faker_.Build();
}
InferDataTypeContextFaker &InferDataTypeContextFaker::NodeIoNum(size_t input_num, size_t output_num) {
  base_faker_.KernelIONum(input_num + kInputsAppendEnd, output_num);
  base_faker_.NodeIoNum(input_num, output_num);
  return *this;
}
InferDataTypeContextFaker &InferDataTypeContextFaker::InputDataTypes(std::vector<void *> input_datatypes) {
  std::vector<void *> inputs(std::move(input_datatypes));
  inputs_ = inputs;
  base_faker_.Inputs(std::move(inputs));
  return *this;
}
InferDataTypeContextFaker &InferDataTypeContextFaker::OutputDataTypes(std::vector<void *> output_datatypes) {
  outputs_ = output_datatypes;
  base_faker_.Outputs(std::move(output_datatypes));
  return *this;
}
FakeKernelContextHolder InferDataTypeContextFaker::Build() const {
  auto context_holder =  base_faker_.Build();
  auto origin_context = context_holder.GetContext<KernelContext>();
  for (size_t i = 0U; i < inputs_.size(); ++i) {
    memcpy_s(origin_context->MutableInputPointer<void *>(i), sizeof(void *), inputs_[i], sizeof(ge::DataType));
  }
  for (size_t i = 0U; i < outputs_.size(); ++i) {
    memcpy_s(origin_context->GetOutputPointer<void *>(i), sizeof(void *), outputs_[i], sizeof(ge::DataType));
  }
  return context_holder;
}

TilingContextFaker &TilingContextFaker::NodeIoNum(size_t input_num, size_t output_num) {
  base_faker_.KernelIONum(input_num + output_num + kInputsAppendEnd, gert::TilingContext::kOutputNum);
  base_faker_.NodeIoNum(input_num, output_num);
  return *this;
}
TilingContextFaker &TilingContextFaker::InputShapes(std::vector<gert::StorageShape *> input_shapes) {
  input_shapes_ = std::move(input_shapes);
  UpdateInputs();
  return *this;
}
TilingContextFaker &TilingContextFaker::OutputShapes(std::vector<gert::StorageShape *> output_shapes) {
  output_shapes_ = std::move(output_shapes);
  UpdateInputs();
  return *this;
}
TilingContextFaker &TilingContextFaker::CompileInfo(void *compile_info) {
  compile_info_ = compile_info;
  UpdateInputs();
  return *this;
}
TilingContextFaker &TilingContextFaker::TilingData(void *tiling_data) {
  outputs_[TilingContext::kOutputTilingData] = tiling_data;
  base_faker_.Outputs(outputs_);
  return *this;
}
TilingContextFaker &TilingContextFaker::Workspace(ContinuousVector *workspace) {
  outputs_[TilingContext::kOutputWorkspace] = workspace;
  base_faker_.Outputs(outputs_);
  return *this;
}
FakeKernelContextHolder TilingContextFaker::Build() const {
  return base_faker_.Build();
}
void TilingContextFaker::UpdateInputs() {
  std::vector<void *> inputs;
  for (const auto input_shape : input_shapes_) {
    inputs.push_back(input_shape);
  }
  for (const auto output_shape : output_shapes_) {
    inputs.push_back(output_shape);
  }
  inputs.push_back(compile_info_);  // kInputsCompileInfo
  inputs.push_back(nullptr);        // kInputsTilingFunc
  base_faker_.Inputs(std::move(inputs));
}
}  // namespace gert
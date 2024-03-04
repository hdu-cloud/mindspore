/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "faker/node_faker.h"
#include "graph/compute_graph.h"
#include "common/checker.h"
namespace gert {
using namespace ge;
ComputeNodeFaker &ComputeNodeFaker::IoNum(size_t input_num, size_t output_num, ge::DataType data_type) {
  inputs_desc_.resize(input_num, ge::GeTensorDesc(GeShape({10,10,10,10}), ge::FORMAT_ND, data_type));
  outputs_desc_.resize(output_num, ge::GeTensorDesc(GeShape({10,10,10,10}), ge::FORMAT_ND, data_type));
  return *this;
}
ge::NodePtr ComputeNodeFaker::Build() {
  auto op_desc = std::make_shared<OpDesc>(name_, type_);
  GE_ASSERT_NOTNULL(op_desc);
  for (size_t i = 0U; i < inputs_desc_.size(); ++i) {
    auto &input_desc = inputs_desc_[i];
    if (i < input_names_.size()) {
      op_desc->AddInputDesc(input_names_[i], input_desc);
      op_desc->AppendIrInput(input_names_[i], ge::kIrInputRequired);
    } else {
      op_desc->AddInputDesc(input_desc);
    }
  }

  for (size_t i = 0U; i < outputs_desc_.size(); ++i) {
    auto &desc = outputs_desc_[i];
    if (i < output_names_.size()) {
      op_desc->AddOutputDesc(output_names_[i], desc);
      op_desc->AppendIrOutput(desc.GetName(), ge::kIrOutputRequired);
    } else {
      op_desc->AddOutputDesc(desc);
    }
  }

  for (const auto &attr : attr_keys_to_value_) {
    op_desc->SetAttr(attr.first, attr.second);
  }

  return graph_->AddNode(op_desc);
}
ComputeNodeFaker &ComputeNodeFaker::NameAndType(std::string name, std::string type) {
  name_ = std::move(name);
  type_ = std::move(type);
  return *this;
}
ComputeNodeFaker &ComputeNodeFaker::InputNames(vector<std::string> names) {
  if (names.size() != inputs_desc_.size()) {
    throw std::invalid_argument("The size of names and input num not match");
  }
  input_names_ = std::move(names);
  return *this;
}
ComputeNodeFaker &ComputeNodeFaker::OutputNames(vector<std::string> names) {
  if (names.size() != outputs_desc_.size()) {
    throw std::invalid_argument("The size of names and output num not match");
  }
  output_names_ = std::move(names);
  return *this;
}
}
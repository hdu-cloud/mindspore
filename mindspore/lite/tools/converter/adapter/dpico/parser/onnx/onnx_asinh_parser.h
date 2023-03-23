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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_PARSER_ONNX_ONNX_ASINH_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_PARSER_ONNX_ONNX_ASINH_PARSER_H_

#include <memory>
#include "include/registry/node_parser.h"
#include "ops/base_operator.h"
#include "ops/custom.h"

namespace mindspore {
namespace lite {
using BaseOperatorPtr = std::shared_ptr<ops::BaseOperator>;
class OnnxAsinhParser : public converter::NodeParser {
 public:
  OnnxAsinhParser() : NodeParser() {}
  ~OnnxAsinhParser() override = default;

  ops::BaseOperatorPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_PARSER_ONNX_ONNX_ASINH_PARSER_H_

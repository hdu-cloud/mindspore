/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "external/graph/operator.h"
#include "external/graph/operator_factory.h"
#include "external/graph/attr_value.h"
#include "graph/compute_graph.h"
#include "graph/ge_context.h"
#include "graph/runtime_inference_context.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/constant_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_utils_ex.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/shape_refiner.h"
#include "graph/opsproto_manager.h"

namespace ge {
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Operator OpDescUtils::CreateOperatorFromNode(ge::ConstNodePtr node_ptr) {
  return Operator("default");
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Operator OpDescUtils::CreateOperatorFromOpDesc(OpDescPtr op_desc) {
  return Operator("default");
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr OpDescUtils::GetOpDescFromOperator(const Operator &oprt) {
  return nullptr;
}

OpDescPtr OpDescUtils::CreateConstOp(const GeTensorPtr &tensor_ptr) {
  return nullptr;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
std::vector<GeTensorPtr> OpDescUtils::MutableWeights(const ge::NodePtr node) {
  return std::vector<ge::GeTensorPtr>();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraphPtr GraphUtilsEx::GetComputeGraph(const ge::Graph &graph) {
  return nullptr;
}

ConstGeTensorPtr TensorAdapter::AsGeTensorPtr(const Tensor &tensor) {
  GeTensorPtr ge_tensor;
  return ge_tensor;
}

GeTensorPtr TensorAdapter::AsGeTensorPtr(Tensor &tensor) {
  GeTensorPtr ge_tensor;
  return ge_tensor;
}

const GeTensor TensorAdapter::AsGeTensor(const Tensor &tensor) {
  return GeTensor();
}

GeTensor TensorAdapter::AsGeTensor(Tensor &tensor) {
  return GeTensor();
}

const Tensor TensorAdapter::AsTensor(const GeTensor &ge_tensor) {
  const Tensor tensor;
  return tensor;
}

Tensor TensorAdapter::AsTensor(GeTensor &ge_tensor) {
  const Tensor tensor;
  return tensor;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus ShapeRefiner::InferShapeAndType(const NodePtr &node, const bool before_subgraph) {
  return GRAPH_SUCCESS;
}

OpsProtoManager *OpsProtoManager::Instance() {
  static OpsProtoManager instance;
  return &instance;
}

bool OpsProtoManager::Initialize(const std::map<std::string, std::string> &options) {
  return true;
}

void OpsProtoManager::Finalize() {
}

GeTensor::GeTensor() {
}

TensorType::TensorType(DataType dt) {
}

TensorType::TensorType(const std::initializer_list<DataType> &initial_types) {
}
} // namespace ge

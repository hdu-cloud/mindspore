/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "graph/debug/ge_log.h"
namespace fe {
const std::string ATTR_NAME_UNKNOWN_SHAPE_OP = "_unknown_shape";
bool UnknownShapeUtils::IsUnKnownShapeTensor(const ge::OpDesc &op_desc) {
  for (auto &tenosr_desc_ptr : op_desc.GetAllInputsDescPtr()) {
    if (tenosr_desc_ptr == nullptr) {
      continue;
    }
    if (tenosr_desc_ptr->GetShape().IsUnknownShape()) {
      return true;
    }
  }

  for (auto &tenosr_desc_ptr : op_desc.GetAllOutputsDescPtr()) {
    if (tenosr_desc_ptr == nullptr) {
      continue;
    }
    if (tenosr_desc_ptr->GetShape().IsUnknownShape()) {
      return true;
    }
  }

  return false;
}

bool UnknownShapeUtils::IsUnknownShapeOp(const ge::OpDesc &op_desc) {
  bool unknown_shape_status = false;
  if (ge::AttrUtils::GetBool(op_desc, ATTR_NAME_UNKNOWN_SHAPE_OP, unknown_shape_status)) {
    return unknown_shape_status;
  }
  if (op_desc.GetAllInputsSize() != 0 || op_desc.GetOutputsSize() != 0) {
    unknown_shape_status = IsUnKnownShapeTensor(op_desc);
  }
  ge::OpDesc *no_const_op_desc = const_cast<ge::OpDesc *>(&op_desc);
  (void)ge::AttrUtils::SetBool(*no_const_op_desc, ATTR_NAME_UNKNOWN_SHAPE_OP, unknown_shape_status);
  GELOGD("Op[%s, %s] Set attr unknown_shape [%d].", op_desc.GetName().c_str(), op_desc.GetType().c_str(),
         unknown_shape_status);
  return unknown_shape_status;
}

bool UnknownShapeUtils::IsContainUnknownDimNum(const ge::OpDesc &op_desc) {
  for (auto &ptr : op_desc.GetAllInputsDescPtr()) {
    if (ptr->GetShape().IsUnknownDimNum()) {
      GELOGD("Op[name:%s,type:%s] has input tensor whose shape contains -2.", op_desc.GetName().c_str(),
             op_desc.GetType().c_str());
      return true;
    }
  }

  for (auto &ptr : op_desc.GetAllOutputsDescPtr()) {
    if (ptr->GetShape().IsUnknownDimNum()) {
      GELOGD("Op[name:%s,type:%s] has output tensor whose shape contains -2.", op_desc.GetName().c_str(),
             op_desc.GetType().c_str());
      return true;
    }
  }

  return false;
}

bool UnknownShapeUtils::IsUnknownShapeValue(const int64_t &value) {
  if (value == ge::UNKNOWN_DIM || value == ge::UNKNOWN_DIM_NUM) {
    return true;
  }
  return false;
}
}  // namespace fe
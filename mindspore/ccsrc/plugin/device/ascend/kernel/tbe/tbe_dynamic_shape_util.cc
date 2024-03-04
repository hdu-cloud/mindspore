/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/tbe/tbe_dynamic_shape_util.h"
#include <memory>
#include <string>
#include <utility>
#include <set>
#include "include/backend/optimizer/helper.h"

namespace mindspore::kernel::tbe {
bool TbeDynamicShapeUtil::GetDynamicShapeAttr(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (anf_node->isa<CNode>()) {
    auto cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    return GetDynamicShapeAttr(cnode);
  }
  return false;
}

bool TbeDynamicShapeUtil::GetDynamicShapeAttr(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto is_dynamic_shape = common::AnfAlgo::IsDynamicShape(cnode);
  return is_dynamic_shape;
}

std::shared_ptr<OpInfo> TbeDynamicShapeUtil::FindOp(const std::string &op_name, const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (anf_node->isa<CNode>()) {
    auto cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    return FindOp(op_name, cnode);
  }
  return nullptr;
}

std::shared_ptr<OpInfo> TbeDynamicShapeUtil::FindOp(const std::string &op_name, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto is_dynamic_shape = GetDynamicShapeAttr(cnode) || common::AnfAlgo::HasNodeAttr(kAttrMutableKernel, cnode);
  auto op_info = mindspore::kernel::OpLib::FindOp(op_name, OpImplyType::kImplyTBE, is_dynamic_shape);
  // If have no dynamic shape op, get static shape op
  if (op_info != nullptr && !op_info->dynamic_shape_support() && is_dynamic_shape) {
    MS_LOG(INFO) << "Node(" << cnode->fullname_with_scope() << ") not support dynamic shape:" << cnode->DebugString();
  }
  return op_info;
}

std::shared_ptr<OpInfo> TbeDynamicShapeUtil::FindOp(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = common::AnfAlgo::GetCNodeName(cnode);
  return FindOp(op_name, cnode);
}

inline std::string GetPrimitiveName(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return "";
  }

  return AnfUtils::GetCNodeName(node);
}

inline void GetRangeByShape(const AnfNodePtr &anf_node, const ShapeVector &shape, RangePair *range) {
  constexpr int64_t kConv2DMaxShape = 2048;
  auto name = GetPrimitiveName(anf_node);
  for (auto val : shape) {
    if (val < 0) {
      // for "Conv2Dxxx" operators, the upper bound of range can not exceed 4096
      if (name.find("Conv2D") != std::string::npos) {
        (void)range->emplace_back(std::make_pair(1L, kConv2DMaxShape));
      } else {
        (void)range->emplace_back(std::make_pair(1L, -1L));
      }
    } else {
      (void)range->emplace_back(std::make_pair(val, val));
    }
  }
}

ShapeVector TbeDynamicShapeUtil::UpdateShape(const AnfNodePtr &node, const std::string &format,
                                             const ShapeVector &shape, size_t index, bool is_input) {
  MS_EXCEPTION_IF_NULL(node);
  const std::set<std::string> op_names = {kTransDataOpName};
  if (!node->isa<CNode>() || op_names.find(common::AnfAlgo::GetCNodeName(node)) == op_names.end()) {
    return shape;
  }
  std::string sp_format = format;
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  if (kernel_info->select_kernel_build_info() != nullptr) {
    auto in_format = AnfAlgo::GetInputFormat(node, 0);
    auto out_format = AnfAlgo::GetOutputFormat(node, 0);
    sp_format = IsOneOfHWSpecialFormat(in_format) ? in_format : out_format;
  }

  const auto &pad_idx =
    is_input ? AnfAlgo::GetInputReshapeType(node, index) : AnfAlgo::GetOutputReshapeType(node, index);
  if (format == kOpFormat_NCHW && shape.size() < kDim4 && IsOneOfDynRankNeedPadShape(sp_format)) {
    return trans::PaddingShape(shape, sp_format, pad_idx);
  }
  return shape;
}

RangePair TbeDynamicShapeUtil::GetInputDynamicRange(const AnfNodePtr &anf_node, size_t index,
                                                    const std::string &def_format, const std::string &ori_format,
                                                    const TypeId &type) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(anf_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto format =
    kernel_info->select_kernel_build_info() == nullptr ? def_format : AnfAlgo::GetInputFormat(anf_node, index);
  auto data_type =
    kernel_info->select_kernel_build_info() == nullptr ? type : AnfAlgo::GetInputDeviceDataType(anf_node, index);

  std::string reshape_type = AnfAlgo::GetInputReshapeType(anf_node, index);
  trans::ShapeRangeTransfer shapeRangeTransfer;
  RangePair ret;

  auto prev_node = common::AnfAlgo::GetPrevNodeOutput(anf_node, index);
  MS_EXCEPTION_IF_NULL(prev_node.first);
  auto shape = common::AnfAlgo::GetOutputInferShape(prev_node.first, prev_node.second);
  if (anf_node->isa<CNode>()) {
    shape = UpdateShape(anf_node, ori_format, shape, index, true);
  }
  GetRangeByShape(anf_node, shape, &ret);

  return shapeRangeTransfer.GetRealRange(ret, format, data_type, reshape_type);
}

RangePair TbeDynamicShapeUtil::GetOutputDynamicRange(const AnfNodePtr &anf_node, size_t index,
                                                     const std::string &def_format, const std::string &ori_format,
                                                     const TypeId &type) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(anf_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto format =
    kernel_info->select_kernel_build_info() == nullptr ? def_format : AnfAlgo::GetOutputFormat(anf_node, index);
  auto data_type =
    kernel_info->select_kernel_build_info() == nullptr ? type : AnfAlgo::GetOutputDeviceDataType(anf_node, index);
  std::string reshape_type =
    kernel_info->select_kernel_build_info() == nullptr ? "" : AnfAlgo::GetOutputReshapeType(anf_node, index);
  trans::ShapeRangeTransfer shapeRangeTransfer;
  RangePair ret;

  auto shape = common::AnfAlgo::GetOutputInferShape(anf_node, index);
  if (anf_node->isa<CNode>()) {
    shape = UpdateShape(anf_node, ori_format, shape, index, false);
  }
  GetRangeByShape(anf_node, shape, &ret);

  return shapeRangeTransfer.GetRealRange(ret, format, data_type, reshape_type);
}
}  // namespace mindspore::kernel::tbe

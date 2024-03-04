/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_STRIDED_SLICE_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_STRIDED_SLICE_INFO_H_

#include <string>

#include <memory>
#include <vector>

#include "utils/hash_map.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
constexpr size_t STRIDE_SLICE_CNODE_BEGIN_INDEX = 2;
constexpr size_t STRIDE_SLICE_CNODE_END_INDEX = 3;
class StridedSliceInfo : public OperatorInfo {
 public:
  StridedSliceInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                   const PrimitiveAttrs &attrs)
      : OperatorInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<StridedSliceCost>()) {}
  ~StridedSliceInfo() override = default;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;
  std::shared_ptr<Strategies> GenerateBatchStrategies() override;
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;

 protected:
  Status GetAttrs() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferMirrorOps() override;
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status GetMask(const std::string &mask_name, int64_t *mask_value);
  void ChangeCNodeBegin();
  void ChangeCNodeEnd();
  void ChangeMakeTupleConstant(const CNodePtr &cnode, size_t make_tuple_index);

 private:
  std::vector<int64_t> begin_;
  std::vector<int64_t> end_;
  std::vector<int64_t> strides_;
  int64_t begin_mask_ = 0;
  int64_t end_mask_ = 0;
  int64_t ellipsis_mask_ = 0;
  int64_t new_axis_mask_ = 0;
  int64_t shrink_axis_mask_ = 0;
  bool skip_redistribution_ = false;
  std::vector<bool> begin_mask_bitmap_;
  std::vector<bool> end_mask_bitmap_;
  std::vector<bool> ellipsis_mask_bitmap_;
  std::vector<bool> new_axis_mask_bitmap_;
  std::vector<bool> shrink_axis_mask_bitmap_;
  Shape input_shape_in_process_;
  void ComputeBeginMask();
  void ComputeEndMask();
  void ComputeEllipsisMask();
  void ComputeNewAxisMask();
  void AdjustShrinkAxisMask();
  Status CheckInputStrategy(const Shape &strategy);
};

using StridedSliceInfoPtr = std::shared_ptr<StridedSliceInfo>;
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_STRIDED_SLICE_INFO_H_

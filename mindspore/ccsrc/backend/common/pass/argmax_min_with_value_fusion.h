/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_ARGMAX_WITH_VALUE_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_ARGMAX_WITH_VALUE_FUSION_H_

#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class ArgMaxWithValueFusion : public PatternProcessPass {
 public:
  explicit ArgMaxWithValueFusion(bool multigraph = true) : PatternProcessPass("argmax_with_value_fusion", multigraph) {}
  ~ArgMaxWithValueFusion() override = default;

  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;
};

class ArgMinWithValueFusion : public PatternProcessPass {
 public:
  explicit ArgMinWithValueFusion(bool multigraph = true) : PatternProcessPass("argmin_with_value_fusion", multigraph) {}
  ~ArgMinWithValueFusion() override = default;

  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_ARGMAX_WITH_VALUE_FUSION_H_

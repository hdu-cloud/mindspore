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

#include "frontend/operator/bprop/bprop_irbuilder.h"
#include "include/common/utils/utils.h"

namespace mindspore::expander::bprop {
REG_BPROP_BUILDERS_BEGIN(GradImplementationsOps)
REG_BPROP_BUILDER("Load").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto u_monad = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  return {dout, ib->ZerosLike(u_monad)};
});
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop

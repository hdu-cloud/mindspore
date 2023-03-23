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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_TF_FAKE_QUANT_ADJUST_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_TF_FAKE_QUANT_ADJUST_H_

#include <string>
#include <vector>
#include "backend/common/optimizer/pass.h"
#include "backend/common/optimizer/optimizer.h"
#include "tools/converter/quantizer/quant_param_holder.h"

namespace mindspore {
namespace lite {
class TFFakeQuantAdjust {
 public:
  bool Adjust(const FuncGraphPtr &func_graph);

 private:
  bool SetQuantParam(const CNodePtr &cnode, const CNodePtr &post_cnode, size_t index);
  bool RemoveFakeQuant(const FuncGraphPtr &func_graph);
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_TF_FAKE_QUANT_PARSER_H_

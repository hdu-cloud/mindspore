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

#include "inc/graph/axis_type_info.h"

namespace ge {
void AxisTypeInfo::AddInputCutInfo(CutInfo &input_cut_info) {
  relate_inputs_.emplace_back(input_cut_info);
}

void AxisTypeInfo::AddOutputCutInfo(CutInfo &output_cut_info) {
  relate_outputs_.emplace_back(output_cut_info);
}

graphStatus AxisTypeInfo::GetInputCutInfo(const size_t index, CutInfo &input_cut_info) const {
  return DoGetCutInfo(relate_inputs_, index, input_cut_info);
}

graphStatus AxisTypeInfo::GetOutputCutInfo(const size_t index, CutInfo &output_cut_info) const {
  return DoGetCutInfo(relate_outputs_, index, output_cut_info);
}

void AxisTypeInfo::AddInputValueCutInfo(const CutInfo &cut_info) {
  relate_input_values_.emplace_back(cut_info);
}

void AxisTypeInfo::AddOutputValueCutInfo(const CutInfo &cut_info) {
  relate_output_values_.emplace_back(cut_info);
}

graphStatus AxisTypeInfo::GetInputValueCutInfo(const size_t index, CutInfo &cut_info) const {
  return DoGetCutInfo(relate_input_values_, index, cut_info);
}

graphStatus AxisTypeInfo::GetOutputValueCutInfo(const size_t index, CutInfo &cut_info) const {
  return DoGetCutInfo(relate_output_values_, index, cut_info);
}

graphStatus AxisTypeInfo::DoGetCutInfo(const std::vector<CutInfo> &cut_infos,
                                       const size_t index,
                                       CutInfo &cut_info) {
  if (cut_infos.size() <= index) {
    return GRAPH_FAILED;
  }
  cut_info = cut_infos[index];
  return GRAPH_SUCCESS;
}
}
